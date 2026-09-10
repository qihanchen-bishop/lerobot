"""Online ACT residual bridge using the training feature and timing contracts."""

import hashlib
import json
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

# Direct GUI script execution adds mycode/, but not its package parent, to sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from mycode.slow_fast_residual import (
    ActionPlan, MultiRateExecutor, OnlineWindow, ResidualConfig, ResidualCorrector,
    context_record, observation_valid, validate_joints,
)
from mycode.train_slow_fast_residual import BasePolicyAdapter


def checkpoint_identity(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Residual checkpoint does not exist: {path}")
    stat = path.stat()
    return _checkpoint_digest(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=16)
def _checkpoint_digest(path, modified, size):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ACTResidualRuntime:
    def __init__(self, path, mode, policy, pre, post, state_names, action_names, settings):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        spec, contract = saved["spec"], saved["contract"]
        cfg = ResidualConfig(**spec["cfg"])
        expected_experiment = {"unet": "UNET-SEM-V5-FS", "qtoken": "UNET-SEM-V5-FS-QTOKEN"}
        experiment = getattr(policy, "experiment", None)
        if mode != spec["mode"] or (
            cfg.variant == "act" and experiment is not None
        ) or (cfg.variant != "act" and experiment != expected_experiment.get(cfg.variant)):
            raise ValueError("Residual mode/variant does not match this ACT checkpoint")
        validate_joints(state_names, action_names, contract["joints"])
        if contract["unit"] != "dataset_native" or policy.config.action_target != "dataset_action":
            raise ValueError("Residual requires dataset-native absolute ACT actions")
        if (settings["fps"] != cfg.fps or settings["replan_interval_steps"] != cfg.slow_stride
                or settings["prediction_steps"] != policy.config.chunk_size
                or settings["fusion_steps"] or settings["auto_replan"] or settings["use_amp"]
                or settings["execution_mode"] != "asynchronous"):
            raise ValueError(f"Residual requires FPS={cfg.fps}, replan={cfg.slow_stride}, full chunk, "
                             "Async, fusion=0, auto-replan off, AMP off")
        if cfg.slow_stride + cfg.slow_delay + cfg.fast_delay + cfg.correction_steps > policy.config.chunk_size:
            raise ValueError("ACT chunk is too short for the residual timing contract")
        keys = list(policy.config.image_features) if cfg.variant == "act" else list(policy.rgb_keys)
        if keys != ["observation.images.front", "observation.images.side"]:
            raise ValueError("Residual requires ordered front/side ACT inputs")
        self.device = next(policy.parameters()).device
        self.cfg = cfg
        self.model = ResidualCorrector(spec["feature_dim"], spec["context_dim"], spec["action_dim"],
                                       saved["model"]["bounds"], cfg, mode)
        self.model.load_state_dict(saved["model"], strict=True)
        self.model.to(self.device).eval()
        self.window = OnlineWindow(self.model)
        # Share the frozen ACT and its processors; features() is exactly the training implementation.
        self.adapter = object.__new__(BasePolicyAdapter)
        self.adapter.cfg, self.adapter.device = cfg, self.device
        self.adapter.policy = policy
        self.adapter.act = policy if cfg.variant == "act" else policy.act_policy
        self.adapter.pre, self.adapter.post, self.adapter.keys = pre, post, keys
        self.adapter.query_capture = None
        self.adapter.latest_plan_features = None
        if cfg.variant == "qtoken":
            from mycode.slow_fast_semantic import QueryCapture
            self.adapter.query_capture = QueryCapture(self.adapter.act.model)
        self.plan_features = None
        self.semantic_frames = {}
        self.mask_frames = {}
        self.scale = torch.tensor(contract["scale"], device=self.device)
        if not torch.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("Invalid residual normalization scale")
        self.executor = MultiRateExecutor(cfg, self.model.bounds)
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="act-residual")
        self.pending = None
        self.origin = time.perf_counter()
        self.last_slow = self.last_fast = -float("inf")
        self.previous_q = self.previous_action = None
        self.previous_time = None
        self.previous_plan_id = None
        self.state_history = deque(maxlen=max(cfg.fast_stride * 4, 32))
        self.plan_id = 0
        self.metadata = {"spec": spec, "contract": contract, "cache_id": saved["cache_id"],
                         "checkpoint_sha256": checkpoint_identity(path),
                         "clock": "elapsed_wall_time", "view_timestamps": "opencv_capture_completion"}

    def close(self):
        self.worker.shutdown(wait=True, cancel_futures=True)
        if self.adapter.query_capture is not None:
            self.adapter.query_capture.close()

    def latest_inference_semantic_input(self):
        return dict(self.semantic_frames)

    def latest_inference_mask_preview(self):
        return dict(self.mask_frames)

    def _predict_plan(self, rgbs, state):
        actions = self.adapter.predict(rgbs, state)
        policy = self.adapter.policy
        semantic = getattr(policy, "latest_inference_semantic_input", lambda: {})()
        masks = getattr(policy, "latest_inference_mask_preview", lambda: {})()
        return actions, self.adapter.latest_plan_features, semantic, masks

    def _infer_fast(self, rgbs, q, context, stamp):
        with torch.inference_mode():
            feature = self.adapter.features(rgbs, q[None])[0]
            if self.cfg.variant == "qtoken":
                if self.plan_features is None:
                    raise RuntimeError("Current plan has no Qtoken descriptor")
                feature = torch.cat((feature, self.plan_features[0].to(feature.device)))
            return self.window.update(feature, context, stamp)

    def step(self, batch, view_times):
        now = time.perf_counter() - self.origin
        times = [t - self.origin for t in view_times]
        stamp = min(times)
        q = batch["observation.state"].reshape(-1).to(self.device)
        if self.previous_action is None:
            self.previous_action = q.clone()
        report = {"time_s": now, "view_times_s": times, "event": "hold", "applied": False}
        if self.pending is not None:
            future, kind, submitted, owner, target, captured = self.pending
            delay = self.cfg.slow_delay if kind == "plan" else self.cfg.fast_delay
            if future.done() and now >= submitted + delay / self.cfg.fps:
                value = future.result()
                if kind == "plan":
                    actions, features, semantic, masks = value
                    plan = ActionPlan(owner, submitted, now, actions[0], self.cfg.fps)
                    accepted = self.executor.accept_plan(plan, now)
                    if accepted:
                        self.plan_features = features
                        self.semantic_frames, self.mask_frames = semantic, masks
                else:
                    accepted = value is not None and self.executor.accept_residual(
                        owner, value, target, captured, now)
                report.update(event=kind, accepted=bool(accepted), pipeline_ms=(now-submitted)*1000)
                self.pending = None
        valid = observation_valid(times, now, self.cfg)
        if valid and (not self.state_history or stamp > self.state_history[-1][0]):
            self.state_history.append((stamp, q.clone()))
        if self.pending is None and valid:
            rgbs = [batch[k] for k in self.adapter.keys]
            if now - self.last_slow >= self.cfg.slow_stride / self.cfg.fps:
                self.plan_id += 1
                self.pending = (self.worker.submit(self._predict_plan, rgbs, q[None]),
                                "plan", stamp, self.plan_id, stamp, times)
                self.last_slow = now
                report["scheduled"] = "plan"
            elif self.executor.plan is not None and now-self.last_fast >= self.cfg.fast_stride/self.cfg.fps:
                plan = self.executor.plan
                target_past = stamp - self.cfg.fast_stride / self.cfg.fps
                prev_q = min(self.state_history, key=lambda item: abs(item[0]-target_past))[1]
                dt = stamp-self.previous_time if self.previous_time is not None else self.cfg.fast_stride/self.cfg.fps
                context = context_record(q, prev_q, self.previous_action, plan, stamp, dt,
                                         self.scale, self.cfg, self.previous_plan_id != plan.plan_id)
                self.pending = (self.worker.submit(self._infer_fast, rgbs, q, context, stamp),
                                "residual", stamp, plan.plan_id, stamp+self.cfg.fast_delay/self.cfg.fps, times)
                self.previous_q, self.previous_time = q.clone(), stamp
                self.previous_plan_id = plan.plan_id
                self.last_fast = now
                report["scheduled"] = "residual"
        command = self.executor.command(now)
        if command is None:
            if self.executor.plan is not None:
                _, valid_plan = self.executor.plan.at(now, 1)
                if not bool(valid_plan[0]):
                    raise RuntimeError("Residual ACT plan expired before a replacement arrived")
            command = self.previous_action.clone()
        if self.executor.plan is not None:
            nominal, _ = self.executor.plan.at(now, 1)
            delta = command - nominal[0]
            report.update(plan_id=self.executor.plan.plan_id, base=nominal[0].tolist(),
                          residual=delta.tolist(), applied=bool(delta.abs().max() > 1e-7))
        self.previous_action = command.clone()
        report.update(command=command.tolist(), observation_valid=valid,
                      history_length=len(self.window.records))
        return command[None], report

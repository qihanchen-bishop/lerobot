"""Offline residual BC runner. Run cache, smoke, train and verify in a dedicated tmux.

Paths are CLI inputs. The cache is a fingerprinted, episode-sharded artifact;
no generated labels or future images are used as policy inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader, Subset

from mycode.slow_fast_residual import (
    ActionPlan,
    MultiRateExecutor,
    OnlineWindow,
    ResidualConfig,
    ResidualCorrector,
    ResidualSequenceDataset,
    context_record,
    diagnostic_metrics,
    residual_loss,
    validate_joints,
)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def save_pt(path: Path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BasePolicyAdapter:
    """Uses the original policy and saved processors, with gradients disabled."""

    def __init__(self, checkpoint: Path, cfg: ResidualConfig, device: str):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.cfg, self.device = cfg, device
        self.checkpoint = checkpoint.resolve()
        if cfg.variant == "act":
            config = PreTrainedConfig.from_pretrained(str(checkpoint))
            config.pretrained_backbone_weights = None
            config.device = device
            self.policy = ACTPolicy.from_pretrained(str(checkpoint), config=config)
            self.pre, self.post = make_pre_post_processors(
                policy_cfg=config,
                pretrained_path=str(checkpoint),
                preprocessor_overrides={"device_processor": {"device": device}},
            )
            self.act = self.policy
            self.keys = list(config.image_features)
        else:
            # Existing inference utilities also support direct script imports.
            script_directory = str(Path(__file__).resolve().parent)
            if script_directory not in sys.path:
                sys.path.insert(0, script_directory)
            from mycode.mask_act_inference import load_mask_act_for_inference

            self.policy, self.pre, self.post, self.details = load_mask_act_for_inference(
                checkpoint, Path(__file__).resolve().parents[1]
            )
            config = self.policy.config
            if str(config.device) != device:
                raise ValueError("Semantic adapter device must match the existing inference loader")
            expected = "UNET-SEM-V5-FS" if cfg.variant == "unet" else "UNET-SEM-V5-FS-QTOKEN"
            if self.details["experiment"].upper() != expected:
                raise ValueError(f"Expected {expected}, got {self.details['experiment']}")
            self.act = self.policy.act_policy
            self.keys = list(self.policy.rgb_keys)
        self.config = config
        self.policy.to(device).eval().requires_grad_(False)
        if len(self.keys) != 2 or config.action_target != "dataset_action":
            raise ValueError("v1 requires a two-RGB-view dataset_action checkpoint")
        if cfg.slow_stride + cfg.slow_delay + cfg.fast_delay + cfg.correction_steps > config.chunk_size:
            raise ValueError("Chunk too short for configured delay and replanning interval")
        self.query_capture = None
        self.latest_plan_features = None
        if cfg.variant == "qtoken":
            from mycode.slow_fast_semantic import QueryCapture

            self.query_capture = QueryCapture(self.act.model)

    def normalized(self, rgbs: list[torch.Tensor], state: torch.Tensor):
        return self.pre({"observation.state": state, **dict(zip(self.keys, rgbs, strict=True))})

    @torch.no_grad()
    def predict(self, rgbs, state):
        if self.query_capture is not None:
            self.query_capture.clear()
        action = self.post(self.policy.predict_action_chunk(self.normalized(rgbs, state))).to(self.device)
        if self.query_capture is not None:
            self.latest_plan_features = self.query_capture.descriptor(state).detach().cpu()
        return action

    @torch.no_grad()
    def features(self, rgbs, state):
        batch = self.normalized(rgbs, state)
        pooled = []
        for key in self.keys:
            small = F.interpolate(
                batch[key],
                size=(self.cfg.image_height, self.cfg.image_width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            feature = self.act.model.backbone(small)["feature_map"]
            pooled.append(F.adaptive_avg_pool2d(feature, (2, 3)).flatten(1))
        if self.cfg.variant == "unet":
            from mycode.slow_fast_semantic import semantic_descriptor

            # Exactly the existing segmenter preprocessing; no ground-truth masks.
            raw = self.policy._get_rgb_inputs(batch, denormalize=True)
            logits, _ = self.policy.predict_view_logits_and_latent_from_rgbs(raw)
            probabilities = self.policy.semantic_probabilities(logits)
            for p, classes in zip(probabilities, self.policy.view_mask_suffixes, strict=True):
                pooled.append(semantic_descriptor(p, list(classes), state))
        return torch.cat(pooled, dim=-1)


def load_metadata(root: Path):
    info = json.loads((root / "meta/info.json").read_text())
    records = []
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        records.extend(pq.read_table(path).to_pylist())
    records.sort(key=lambda row: row["episode_index"])
    if [r["episode_index"] for r in records] != list(range(info["total_episodes"])):
        raise ValueError("Missing or duplicate episode metadata")
    if sum(r["length"] for r in records) != info["total_frames"]:
        raise ValueError("Episode frame counts disagree")
    return info, records


def episode_table(root: Path, info: dict, record: dict):
    path = root / info["data_path"].format(
        chunk_index=record["data/chunk_index"], file_index=record["data/file_index"]
    )
    table = pq.read_table(path).to_pandas()
    table = table[table.episode_index == record["episode_index"]].sort_values("frame_index")
    if len(table) != record["length"] or table.frame_index.tolist() != list(range(len(table))):
        raise ValueError(f"Bad frame indices in {path}")
    times = torch.tensor(table.timestamp.to_numpy(), dtype=torch.float64)
    if len(times) > 1 and not torch.allclose(
        times.diff(), torch.full_like(times[1:], 1 / info["fps"]), atol=1e-4
    ):
        raise ValueError("Dataset timing is not uniformly sampled; resampling must be explicit")
    states = torch.tensor(np.stack(table["observation.state"]), dtype=torch.float32)
    actions = torch.tensor(np.stack(table.action), dtype=torch.float32)
    if not torch.isfinite(states).all() or not torch.isfinite(actions).all():
        raise ValueError("Nonfinite state/action")
    return states, actions, times


def video_frames(root, info, record, key, wanted):
    prefix = f"videos/{key}"
    path = root / info["video_path"].format(
        video_key=key, chunk_index=record[f"{prefix}/chunk_index"], file_index=record[f"{prefix}/file_index"]
    )
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened() or abs(capture.get(cv2.CAP_PROP_FPS) - info["fps"]) > 0.01:
        raise ValueError(f"Invalid video/fps: {path}")
    offset = round(record[f"{prefix}/from_timestamp"] * info["fps"])
    expected = record[f"{prefix}/to_timestamp"] - record[f"{prefix}/from_timestamp"]
    if abs(expected - record["length"] / info["fps"]) > 0.05:
        raise ValueError(f"Video/episode duration mismatch: {path}")
    if capture.get(cv2.CAP_PROP_FRAME_COUNT) < offset + record["length"]:
        raise ValueError(f"Truncated video: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, offset)
    frames = []
    selection = set(wanted)
    try:
        for index in range(record["length"]):
            ok, bgr = capture.read()
            if not ok:
                raise ValueError(f"Decode failed: {path}, frame {index}")
            if index in selection:
                frames.append(torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
    finally:
        capture.release()
    return torch.stack(frames)


def fingerprint(args, cfg, info):
    root, checkpoint = Path(args.root), Path(args.checkpoint)
    source = []
    for directory in (root / "meta", root / "data", root / "videos"):
        for p in sorted(directory.rglob("*")):
            if p.is_file():
                s = p.stat()
                source.append((str(p.relative_to(root)), s.st_size, s.st_mtime_ns))
    weights = checkpoint / ("model.safetensors" if cfg.variant == "act" else "training_state.pt")
    payload = {
        "schema": 2,
        "precision": "float32_tf32_disabled",
        "config": asdict(cfg),
        "checkpoint": str(checkpoint.resolve()),
        "weights_sha256": sha(weights),
        "source": source,
        "total_frames": info["total_frames"],
    }
    for filename in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
        if (checkpoint / filename).exists():
            payload[filename] = sha(checkpoint / filename)
    if cfg.variant != "act":
        payload["run_config"] = sha(checkpoint.parent / "mask_act_run_config.json")
    payload["id"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def split_records(records, seed):
    groups = {}
    for r in records:
        groups.setdefault(tuple(r["tasks"]), []).append(r["episode_index"])
    rng = random.Random(seed)
    train, val = [], []
    for ids in groups.values():
        rng.shuffle(ids)
        count = max(1, round(len(ids) * 0.2))
        val.extend(ids[:count])
        train.extend(ids[count:])
    return {
        "train": sorted(train),
        "val": sorted(val),
        "seed": seed,
        "claim": "Residual-head-held-out episodes only; frozen base may have seen all episodes",
        "stratification": "task; initial-grid metadata unavailable",
    }


def prepare_cache(args, cfg):
    root, cache = Path(args.root), Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    info, records = load_metadata(root)
    if info["fps"] != cfg.fps:
        raise ValueError("FPS mismatch")
    joints = info["features"]["action"]["names"]
    validate_joints(info["features"]["observation.state"]["names"], joints, joints)
    manifest = fingerprint(args, cfg, info)
    manifest_path = cache / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["id"] != manifest["id"]:
        raise ValueError("Cache fingerprint mismatch; use a new cache directory")
    if (cache / "complete.json").exists():
        load_cache(cache, cfg)
        print("Complete fingerprint-matched cache verified; no re-encoding", flush=True)
        return
    write_json(manifest_path, manifest)
    write_json(cache / "split.json", split_records(records, cfg.seed))
    # Dataset-native, checkpoint-independent loss scaling; no validation samples.
    split = json.loads((cache / "split.json").read_text())
    train_actions = torch.cat(
        [episode_table(root, info, r)[1] for r in records if r["episode_index"] in split["train"]]
    )
    scale = train_actions.std(0).clamp_min(1e-3)
    write_json(
        cache / "action_contract.json",
        {
            "joints": joints,
            "unit": cfg.unit,
            "note": "SO default calibrated -100..100 command coordinates; not angular degrees",
            "scale": scale.tolist(),
            "action_target": "dataset_action",
            "state_source": "follower",
            "supervision_source": "recorded leader command",
        },
    )
    adapter = BasePolicyAdapter(Path(args.checkpoint), cfg, args.device)
    if adapter.config.robot_state_feature.shape[0] != len(joints):
        raise ValueError("Checkpoint joint dimension mismatch")
    times_slow, times_fast = [], []
    for record in records:
        ep_id = record["episode_index"]
        destination = cache / f"episode_{ep_id:03d}.pt"
        if destination.exists():
            loaded = torch.load(destination, weights_only=False)
            if loaded["fingerprint"] != manifest["id"]:
                raise ValueError("Incomplete/stale cache shard")
            continue
        started = time.perf_counter()
        q, actions, stamps = episode_table(root, info, record)
        selected = list(range(0, len(q), cfg.fast_stride))
        plan_frames = list(range(0, len(q), cfg.slow_stride))
        all_frames = sorted(set(selected + plan_frames))
        images = [video_frames(root, info, record, key, all_frames) for key in adapter.keys]
        lookup = {frame: index for index, frame in enumerate(all_frames)}
        all_features = []
        for start in range(0, len(selected), args.encode_batch):
            chosen = selected[start : start + args.encode_batch]
            ids = [lookup[t] for t in chosen]
            rgbs = [x[ids].to(args.device).float() / 255 for x in images]
            torch.cuda.synchronize() if args.device == "cuda" else None
            clock = time.perf_counter()
            features = adapter.features(rgbs, q[chosen].to(args.device))
            torch.cuda.synchronize() if args.device == "cuda" else None
            times_fast.append((time.perf_counter() - clock) / len(chosen))
            all_features.append(features.cpu())
        features = torch.cat(all_features)
        plans, plan_features = {}, {}
        for frame in plan_frames:
            rgbs = [x[lookup[frame] : lookup[frame] + 1].to(args.device).float() / 255 for x in images]
            torch.cuda.synchronize() if args.device == "cuda" else None
            clock = time.perf_counter()
            prediction = adapter.predict(rgbs, q[frame : frame + 1].to(args.device))[0].cpu()
            torch.cuda.synchronize() if args.device == "cuda" else None
            times_slow.append(time.perf_counter() - clock)
            plans[frame] = ActionPlan(
                frame,
                float(stamps[frame]),
                float(stamps[frame]) + cfg.slow_delay / cfg.fps,
                prediction,
                cfg.fps,
            )
            if adapter.latest_plan_features is not None:
                plan_features[frame] = adapter.latest_plan_features[0].clone()
        collected = {k: [] for k in ("features", "context", "base", "target", "valid", "age", "time")}
        previous_frame, previous_plan = None, None
        for row, frame in enumerate(selected):
            ready = [p for p in plan_frames if p + cfg.slow_delay <= frame]
            if not ready:
                continue
            p = ready[-1]
            plan = plans[p]
            now = float(stamps[frame])
            dt = (
                cfg.fast_stride / cfg.fps
                if previous_frame is None
                else float(stamps[frame] - stamps[previous_frame])
            )
            ctx = context_record(
                q[frame],
                q[max(0, frame - cfg.fast_stride)],
                actions[frame - 1] if frame else q[frame],
                plan,
                now,
                dt,
                scale,
                cfg,
                switched=previous_plan != p,
            )
            nominal, valid = plan.at(now + cfg.fast_delay / cfg.fps, cfg.correction_steps)
            targets = torch.arange(frame + cfg.fast_delay, frame + cfg.fast_delay + cfg.correction_steps)
            valid &= targets < len(q)
            if not valid.any():
                continue
            for key, value in {
                "features": torch.cat((features[row], plan_features[p]))
                if p in plan_features
                else features[row],
                "context": ctx,
                "base": nominal,
                "target": actions[targets.clamp(max=len(q) - 1)],
                "valid": valid,
                "age": torch.tensor(now - plan.observed_at),
                "time": torch.tensor(now, dtype=torch.float64),
            }.items():
                collected[key].append(value)
            previous_frame, previous_plan = frame, p
        result = {k: torch.stack(v) for k, v in collected.items()}
        if not all(torch.isfinite(v).all() for v in result.values()):
            raise ValueError("Nonfinite cache")
        result.update(episode_id=ep_id, fingerprint=manifest["id"])
        print(
            json.dumps(
                {
                    "cache_episode": ep_id,
                    "samples": len(result["features"]),
                    "seconds": round(time.perf_counter() - started, 2),
                }
            ),
            flush=True,
        )
        if ep_id == 0:
            rgbs = [x[:1].to(args.device).float() / 255 for x in images]
            state = q[:1].to(args.device)
            normalized = adapter.normalized(rgbs, state)
            original = adapter.post(adapter.policy.predict_action_chunk(normalized)).to(args.device)
            torch.testing.assert_close(original, adapter.predict(rgbs, state), atol=0, rtol=0)
            direct = adapter.features(rgbs, state).cpu()
            torch.testing.assert_close(direct, features[:1], atol=3e-4, rtol=3e-4)
            assert not any(p.requires_grad or p.grad is not None for p in adapter.policy.parameters())
            write_json(
                cache / "adapter_validation.json",
                {
                    "zero_base_max_diff": 0,
                    "cache_direct_max_diff": (direct - features[:1]).abs().max().item(),
                    "frozen_parameters": True,
                    "feature_dim": result["features"].shape[-1],
                    "plan_feature_dim": len(plan_features[0]) if plan_features else 0,
                },
            )
        save_pt(destination, result)
    single_latency = []
    # Measure batch-one latency separately; throughput timings are not latency.
    for _ in range(20):
        torch.cuda.synchronize() if args.device == "cuda" else None
        clock = time.perf_counter()
        adapter.features(rgbs, q[:1].to(args.device))
        torch.cuda.synchronize() if args.device == "cuda" else None
        single_latency.append(time.perf_counter() - clock)
    write_json(
        cache / "complete.json",
        {
            "episodes": len(records),
            "fingerprint": manifest["id"],
            "slow_latency_p50_s": float(np.median(times_slow)) if times_slow else None,
            "slow_latency_p95_s": float(np.quantile(times_slow, 0.95)) if times_slow else None,
            "fast_batched_per_frame_p95_s": float(np.quantile(times_fast, 0.95)) if times_fast else None,
            "fast_single_p95_s": float(np.quantile(single_latency[3:], 0.95)),
            "timing_caveat": "Batched feature timing is throughput, not realtime latency",
        },
    )


def load_cache(cache: Path, cfg):
    manifest = json.loads((cache / "manifest.json").read_text())
    complete = json.loads((cache / "complete.json").read_text())
    if manifest["config"] != asdict(cfg) or complete["fingerprint"] != manifest["id"]:
        raise ValueError("Cache/config mismatch")
    if cfg.variant == "act":
        for path in Path(manifest["checkpoint"]).glob("policy_*"):
            if path.stat().st_mtime_ns > (cache / "manifest.json").stat().st_mtime_ns:
                raise ValueError("Saved ACT preprocessing changed after cache creation; rebuild the cache")
    split = json.loads((cache / "split.json").read_text())
    contract = json.loads((cache / "action_contract.json").read_text())
    data = {}
    for name in ("train", "val"):
        data[name] = [torch.load(cache / f"episode_{i:03d}.pt", weights_only=False) for i in split[name]]
        if any(ep["fingerprint"] != manifest["id"] for ep in data[name]):
            raise ValueError("Cache shard mismatch")
    scale = torch.tensor(contract["scale"])
    residuals = torch.cat([(e["target"] - e["base"])[e["valid"].bool()].abs() for e in data["train"]])
    bounds = torch.quantile(residuals, 0.95, dim=0).clamp(0.05, 5.0)
    return data, scale, bounds, manifest, contract


@torch.no_grad()
def evaluate(model, loader, device, scale, bounds):
    model.eval()
    collected = {k: [] for k in ("base", "target", "valid", "age", "residual")}
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        r = model(batch["features"], batch["context"], batch["lengths"])
        for key in collected:
            collected[key].append((r if key == "residual" else batch[key]).cpu())
    values = {k: torch.cat(v) for k, v in collected.items()}
    return diagnostic_metrics(
        values["base"],
        values["target"],
        values["residual"],
        values["valid"],
        scale.cpu(),
        bounds.cpu(),
        values["age"],
    )


def build_model(spec, bounds):
    spec = dict(spec)
    spec["cfg"] = ResidualConfig(**spec["cfg"])
    return ResidualCorrector(**spec, bounds=bounds)


def train(args, cfg):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data, scale, bounds, manifest, contract = load_cache(Path(args.cache), cfg)
    if Path(args.checkpoint).resolve() != Path(manifest["checkpoint"]).resolve():
        raise ValueError("Training checkpoint does not match the cached frozen policy")
    history = 1 if args.mode == "mlp" else cfg.history
    train_set = ResidualSequenceDataset(data["train"], cfg, history)
    val_set = ResidualSequenceDataset(data["val"], cfg, history)
    if args.smoke:
        train_set = Subset(train_set, list(range(min(64, len(train_set)))))
        val_set = train_set
    sample = train_set[0]
    model = ResidualCorrector(
        sample["features"].shape[-1], sample["context"].shape[-1], len(scale), bounds, cfg, args.mode
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    contract.update(
        residual_bounds=bounds.tolist(), bounds_note="min(train P95, 5 native units); offline only"
    )
    config = {"args": vars(args), "model": model.spec, "cache_id": manifest["id"], "contract": contract}
    config_path = out / "config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text())
        if old["model"] != config["model"] or old["cache_id"] != config["cache_id"]:
            raise ValueError("Output directory belongs to another experiment")
        if not args.resume:
            raise ValueError("Output exists; use --resume or a new output directory")
    else:
        write_json(config_path, config)
    start, best, stale = 0, math_inf(), 0
    if args.resume and (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location=args.device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, best, stale = state["epoch"] + 1, state["best"], state["stale"]
        torch.set_rng_state(state["rng"].cpu())
        if stale >= args.patience:
            start = args.epochs  # Re-verify an early-stopped run without training it again.
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    scale = scale.to(args.device)
    initial = evaluate(model, val_loader, args.device, scale, bounds)
    if not (out / "initial.json").exists():
        write_json(out / "initial.json", initial)
    for epoch in range(start, args.epochs):
        model.train()
        losses = []
        started = time.perf_counter()
        for batch in loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            r = model(batch["features"], batch["context"], batch["lengths"])
            loss = residual_loss(r, batch, scale)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = evaluate(model, val_loader, args.device, scale, bounds)
        score = metrics["corrected_normalized_mae"]
        improved = score < best - 1e-6
        best, stale = (score, 0) if improved else (best, stale + 1)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "grad_norm": float(norm),
            "seconds": time.perf_counter() - started,
            **metrics,
        }
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in (
                        "epoch",
                        "train_loss",
                        "seconds",
                        "base_normalized_mae",
                        "corrected_normalized_mae",
                        "residual_rms",
                    )
                }
            ),
            flush=True,
        )
        saved = {
            "model": model.state_dict(),
            "spec": model.spec,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best": best,
            "stale": stale,
            "rng": torch.get_rng_state(),
            "cache_id": manifest["id"],
            "metrics": metrics,
            "contract": contract,
        }
        save_pt(out / "last.pt", saved)
        if improved:
            save_pt(out / "best.pt", saved)
        if (epoch + 1) % 5 == 0:
            save_pt(out / f"epoch_{epoch + 1:03d}.pt", saved)
        if stale >= args.patience:
            break
    saved = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    reloaded = build_model(saved["spec"], saved["model"]["bounds"])
    reloaded.load_state_dict(saved["model"], strict=True)
    reloaded.to(args.device).eval()
    metrics = evaluate(reloaded, val_loader, args.device, scale, bounds)
    if abs(metrics["corrected_normalized_mae"] - saved["metrics"]["corrected_normalized_mae"]) > 1e-6:
        raise ValueError("Reload metrics changed")
    # The online path uses the exact same right-padded fixed window semantics.
    episode = data["val"][0]
    window = OnlineWindow(reloaded)
    diff = 0.0
    dataset = ResidualSequenceDataset([episode], cfg, history)
    for i in range(min(24, len(episode["features"]))):
        f, c = episode["features"][i].to(args.device), episode["context"][i].to(args.device)
        online = window.update(f, c, float(episode["time"][i]))
        example = dataset[i]
        with torch.no_grad():
            offline = reloaded(
                example["features"][None].to(args.device),
                example["context"][None].to(args.device),
                example["lengths"][None].to(args.device),
            )[0]
        diff = max(diff, float((online - offline).abs().max()))
    if diff > 1e-4:
        raise ValueError(f"Online/offline mismatch {diff}")
    if (
        args.smoke
        and metrics["corrected_normalized_mae"]
        >= json.loads((out / "initial.json").read_text())["corrected_normalized_mae"]
    ):
        raise ValueError("Small-sample overfit did not improve")
    write_json(
        out / "result.json",
        {
            "status": "verified",
            "metrics": metrics,
            "best_epoch": saved["epoch"],
            "reload_ok": True,
            "online_offline_max_diff": diff,
            "scope": "Offline expert-state evaluation; no robot executed; not full-system held-out data",
            "train_samples": len(train_set),
            "val_samples": len(val_set),
        },
    )


def math_inf():
    return float("inf")


@torch.no_grad()
def shadow(args, cfg):
    """Replay real RGB through both networks and the scheduler; never send commands."""
    output = Path(args.output)
    saved = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(saved["spec"], saved["model"]["bounds"])
    model.load_state_dict(saved["model"])
    model.to(args.device).eval()
    adapter = BasePolicyAdapter(Path(args.checkpoint), cfg, args.device)
    window = OnlineWindow(model)
    executor = MultiRateExecutor(cfg, model.bounds.cpu())
    root = Path(args.root)
    info, records = load_metadata(root)
    split = json.loads((Path(args.cache) / "split.json").read_text())
    record = records[split["val"][0]]
    q, target, stamps = episode_table(root, info, record)
    count = min(120, len(q))
    images = [video_frames(root, info, record, k, list(range(count))) for k in adapter.keys]
    scale = torch.tensor(saved["contract"]["scale"])
    warm_rgbs = [im[:1].to(args.device).float() / 255 for im in images]
    for _ in range(3):
        adapter.predict(warm_rgbs, q[:1].to(args.device))
        adapter.features(warm_rgbs, q[:1].to(args.device))
        model(
            torch.zeros(1, cfg.history, saved["spec"]["feature_dim"], device=args.device),
            torch.zeros(1, cfg.history, saved["spec"]["context_dim"], device=args.device),
            torch.tensor([cfg.history], device=args.device),
        )
    ready_plan = None
    plan_features = {}
    previous_command = q[0]
    last_fast, last_plan_id = None, None
    fast_times, slow_times, rows = [], [], []
    accepted_residuals = 0
    for frame in range(count):
        now = float(stamps[frame])
        if frame % cfg.slow_stride == 0:
            rgbs = [im[frame : frame + 1].to(args.device).float() / 255 for im in images]
            torch.cuda.synchronize() if args.device == "cuda" else None
            start = time.perf_counter()
            prediction = adapter.predict(rgbs, q[frame : frame + 1].to(args.device))[0].cpu()
            torch.cuda.synchronize() if args.device == "cuda" else None
            slow_times.append(time.perf_counter() - start)
            ready_plan = ActionPlan(
                frame, now, now + max(cfg.slow_delay / cfg.fps, slow_times[-1]), prediction, cfg.fps
            )
            if adapter.latest_plan_features is not None:
                plan_features[frame] = adapter.latest_plan_features[0].clone()
        if ready_plan is not None and ready_plan.ready_at <= now + 1e-6:
            executor.accept_plan(ready_plan, now + 1e-6)
            ready_plan = None
        p = executor.plan
        if p is not None and frame % cfg.fast_stride == 0:
            rgbs = [im[frame : frame + 1].to(args.device).float() / 255 for im in images]
            torch.cuda.synchronize() if args.device == "cuda" else None
            start = time.perf_counter()
            features = adapter.features(rgbs, q[frame : frame + 1].to(args.device))[0]
            if p.plan_id in plan_features:
                features = torch.cat((features, plan_features[p.plan_id].to(args.device)))
            dt = cfg.fast_stride / cfg.fps if last_fast is None else now - last_fast
            context = context_record(
                q[frame],
                q[max(0, frame - cfg.fast_stride)],
                previous_command,
                p,
                now,
                dt,
                scale,
                cfg,
                p.plan_id != last_plan_id,
            ).to(args.device)
            residual = window.update(features, context, now)
            torch.cuda.synchronize() if args.device == "cuda" else None
            elapsed = time.perf_counter() - start
            fast_times.append(elapsed)
            accepted_residuals += int(
                executor.accept_residual(
                    p.plan_id, residual.cpu(), now + cfg.fast_delay / cfg.fps, [now, now], now + elapsed
                )
            )
            last_fast, last_plan_id = now, p.plan_id
        command = executor.command(now)
        if command is not None:
            previous_command = command
            nominal = executor.plan.at(now, 1)[0][0]
            rows.append(
                {
                    "frame": frame,
                    "observation_time": now,
                    "plan_id": executor.plan.plan_id,
                    "plan_age": now - executor.plan.observed_at,
                    "command": command.tolist(),
                    "nominal": nominal.tolist(),
                    "applied_residual": (command - nominal).tolist(),
                    "expert": target[frame].tolist(),
                }
            )
    with (output / "shadow_commands.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    report = {
        "episode": record["episode_index"],
        "commands": len(rows),
        "commands_with_nonzero_residual": sum(
            max(abs(x) for x in row["applied_residual"]) > 1e-6 for row in rows
        ),
        "robot_io": False,
        "accepted_residual_updates": accepted_residuals,
        "attempted_residual_updates": len(fast_times),
        "overdue_residuals_rejected": len(fast_times) - accepted_residuals,
        "expired_prefix_slots": executor.expired_prefix_slots,
        "scope": "Expert observation replay, not closed-loop physical simulation",
        "fast_p50_s": float(np.median(fast_times)),
        "fast_p95_s": float(np.quantile(fast_times, 0.95)),
        "slow_p95_s": float(np.quantile(slow_times, 0.95)),
        "fast_delay_overrun_fraction": float(np.mean(np.array(fast_times) > cfg.fast_delay / cfg.fps)),
        "fast_period_overrun_fraction": float(np.mean(np.array(fast_times) > cfg.fast_stride / cfg.fps)),
        "slow_delay_overrun_fraction": float(np.mean(np.array(slow_times) > cfg.slow_delay / cfg.fps)),
        "timing_note": "Measured sequential replay on shared GPU; no camera/robot transport included",
    }
    write_json(output / "shadow_result.json", report)
    print(json.dumps(report), flush=True)


def audit(args, cfg):
    output, cache = Path(args.output), Path(args.cache)
    if (output / "exit_code").read_text().strip() != "0":
        raise ValueError("Stage runner did not exit successfully")
    data, _, _, manifest, _ = load_cache(cache, cfg)
    if len(data["train"]) + len(data["val"]) != 90:
        raise ValueError("This experiment requires the verified full 90-episode dataset")
    results = {}
    for mode in ["smoke", "gru8", "mlp"] if cfg.variant == "act" else ["smoke", "gru8"]:
        result = json.loads((output / mode / "result.json").read_text())
        if result["status"] != "verified" or not result["reload_ok"]:
            raise ValueError("Unverified training result")
        for name in ("best", "last"):
            saved = torch.load(output / mode / f"{name}.pt", map_location="cpu", weights_only=False)
            if saved["cache_id"] != manifest["id"]:
                raise ValueError("Checkpoint provenance mismatch")
            if not all(torch.isfinite(t).all() for t in saved["model"].values()):
                raise ValueError("Nonfinite checkpoint")
            model = build_model(saved["spec"], saved["model"]["bounds"])
            model.load_state_dict(saved["model"], strict=True)
        results[mode] = result
    checkpoint = Path(manifest["checkpoint"])
    weights = checkpoint / ("model.safetensors" if cfg.variant == "act" else "training_state.pt")
    if sha(weights) != manifest["weights_sha256"]:
        raise ValueError("Frozen source checkpoint changed")
    adapter = BasePolicyAdapter(checkpoint, cfg, args.device)
    root = Path(args.root)
    info, records = load_metadata(root)
    q, _, _ = episode_table(root, info, records[0])
    cached = torch.load(cache / "episode_000.pt", weights_only=False)
    probes = [0, min(60, len(cached["time"]) - 1), min(200, len(cached["time"]) - 1)]
    frames = [round(float(cached["time"][i]) * cfg.fps) for i in probes]
    starts = [max(p for p in range(0, len(q), cfg.slow_stride) if p + cfg.slow_delay <= f) for f in frames]
    selected = sorted(set(frames + starts))
    images = [video_frames(root, info, records[0], k, selected) for k in adapter.keys]
    errors = []
    for row, frame, start in zip(probes, frames, starts, strict=True):
        rgbs = [
            x[selected.index(start) : selected.index(start) + 1].to(args.device).float() / 255 for x in images
        ]
        adapter.predict(rgbs, q[start : start + 1].to(args.device))
        rgbs = [
            x[selected.index(frame) : selected.index(frame) + 1].to(args.device).float() / 255 for x in images
        ]
        direct = adapter.features(rgbs, q[frame : frame + 1].to(args.device)).cpu()[0]
        if adapter.latest_plan_features is not None:
            direct = torch.cat((direct, adapter.latest_plan_features[0]))
        torch.testing.assert_close(direct, cached["features"][row], atol=3e-4, rtol=3e-4)
        errors.append(float((direct - cached["features"][row]).abs().max()))
    shadow_result = json.loads((output / "gru8/shadow_result.json").read_text())
    if shadow_result["robot_io"] or shadow_result["commands"] < 1:
        raise ValueError("Missing offline-only shadow verification")
    if shadow_result.get("commands_with_nonzero_residual", 0) < 1:
        raise ValueError("Shadow did not actually apply any nonzero corrections")
    tests = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_slow_fast_residual.py",
            "tests/test_unet_sem_v5.py",
            "tests/test_mask_act_inference_compat.py",
            "tests/policies/test_act_image_feature_residual.py",
        ],
        capture_output=True,
        text=True,
    )
    (output / "regression.log").write_text(tests.stdout + tests.stderr)
    if tests.returncode:
        raise RuntimeError("Regression tests failed; see regression.log")
    sources = {
        str(p): sha(p)
        for p in [Path(__file__), Path("mycode/slow_fast_residual.py"), Path("mycode/slow_fast_semantic.py")]
    }
    report = {
        "status": "verified",
        "stage": cfg.variant,
        "checkpoint_unchanged": True,
        "tests": tests.stdout,
        "results": results,
        "shadow": shadow_result,
        "source_sha256": sources,
        "cache_id": manifest["id"],
        "rechecked_cache_online_max_differences": errors,
        "claims": "Offline residual-head validation only; no robot actions",
    }
    write_json(output / "audit.json", report)
    print(json.dumps({"audit": "verified", "output": str(output), "tests": tests.stdout}), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["cache", "train", "shadow", "audit"])
    p.add_argument("--variant", choices=["act", "unet", "qtoken"], default="act")
    p.add_argument("--root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mode", choices=["gru", "mlp"], default="gru")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--encode-batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--slow-delay", type=int, default=3)
    p.add_argument("--fast-delay", type=int, default=1)
    p.add_argument("--slow-stride", type=int, default=30, help="Control frames between ACT plans")
    p.add_argument("--fast-stride", type=int, default=2, help="Control frames between residual updates")
    p.add_argument("--history", type=int, default=8)
    p.add_argument("--correction-steps", type=int, default=None)
    p.add_argument("--history-gap", type=float, default=None)
    p.add_argument("--residual-max-age", type=float, default=None)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cv2.setNumThreads(1)
    random.seed(1000)
    np.random.seed(1000)
    torch.manual_seed(1000)
    cfg = ResidualConfig(
        variant=args.variant,
        slow_delay=args.slow_delay,
        fast_delay=args.fast_delay,
        slow_stride=args.slow_stride,
        fast_stride=args.fast_stride,
        correction_steps=args.correction_steps if args.correction_steps is not None else args.fast_stride,
        plan_steps=max(6, args.correction_steps if args.correction_steps is not None else args.fast_stride),
        history_gap=args.history_gap,
        residual_max_age=args.residual_max_age,
        history=args.history,
        hidden=args.hidden,
    )
    print(
        json.dumps({"pid": os.getpid(), "tmux": os.environ.get("TMUX_PANE"), "config": asdict(cfg)}),
        flush=True,
    )
    if args.command == "cache":
        prepare_cache(args, cfg)
    elif args.command == "train":
        train(args, cfg)
    elif args.command == "shadow":
        shadow(args, cfg)
    else:
        audit(args, cfg)

"""Thin LeRobot adapter for the pinned official VIOLA implementation.

No semantic annotations are used. Run with .cache/viola/venv/bin/python.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import cv2
import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / ".cache/viola/upstream"
WEIGHTS = ROOT / ".cache/viola/Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size.pth"
REVISION = "1e8b5ae90e73a1c2a33a369323117cd8d7b0ef36"
WEIGHT_SHA256 = "17fd4f671780ce1fa9389b8f627cb3e86bf82e143871c0c076c990312377a75a"
BOX_KEY = "centernet_bbox_15"
OBS_KEYS = ["agentview_rgb", "eye_in_hand_rgb", "joint_states", BOX_KEY]
VIEWS = {"agentview_rgb": "observation.images.front", "eye_in_hand_rgb": "observation.images.side"}


def sha256(path):
    with open(path, "rb") as stream:
        return (
            hashlib.file_digest(stream, "sha256").hexdigest()
            if hasattr(hashlib, "file_digest")
            else _hash_stream(stream)
        )


def _hash_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(2**20), b""):
        digest.update(block)
    return digest.hexdigest()


def imports():
    revision = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise RuntimeError(f"Unexpected VIOLA revision: {revision}")
    if subprocess.check_output(["git", "-C", str(UPSTREAM), "diff", "--name-only"], text=True).strip():
        raise RuntimeError("Official VIOLA checkout has modifications; audit before proceeding")
    sys.path[:0] = [
        str(UPSTREAM),
        str(UPSTREAM / "third_party/Detic"),
        str(UPSTREAM / "third_party/Detic/third_party/CenterNet2/projects/CenterNet2"),
    ]


def predictor(device="cuda"):
    imports()
    if sha256(WEIGHTS) != WEIGHT_SHA256:
        raise RuntimeError("Official Detic checkpoint checksum mismatch")
    # Only the checksum-verified official checkpoint needs legacy pickle loading.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from centernet.config import add_centernet_config
    from detic.config import add_detic_config

    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(
        str(UPSTREAM / "third_party/Detic/configs/Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size.yaml")
    )
    cfg.MODEL.WEIGHTS = str(WEIGHTS)
    cfg.MODEL.DEVICE = device
    cfg.MODEL.META_ARCHITECTURE = "ProposalNetwork"
    cfg.MODEL.CENTERNET.POST_NMS_TOPK_TEST = 128
    cfg.MODEL.CENTERNET.NMS_TH_TEST = 0.05
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    return DefaultPredictor(cfg)


def resize_rgb(bgr):
    return cv2.cvtColor(cv2.resize(bgr, (128, 128), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def proposals(detector, rgb):
    # DefaultPredictor accepts BGR, independent of its internal INPUT.FORMAT.
    # robomimic globally enables cuDNN benchmarking on import; do not let that
    # switch the proposal convolutions between offline caching and inference.
    with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=False, allow_tf32=True):
        boxes = detector(np.ascontiguousarray(rgb[..., ::-1]))["proposals"].proposal_boxes
    boxes = boxes[(boxes.area() > 16) & (boxes.area() < 128 * 128 / 4)]
    if len(boxes) < 15:
        raise RuntimeError(f"Only {len(boxes)} valid proposals; refusing to fabricate/pad object boxes")
    result = boxes.tensor[:15].detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(result).all() or not (result[:, 2:] > result[:, :2]).all():
        raise RuntimeError("Invalid proposal coordinates")
    return result, len(boxes)


def normalizer(states, actions):
    return dict(
        state_mean=states.mean(0).tolist(),
        state_std=np.maximum(states.std(0), 1e-6).tolist(),
        action_center=((actions.max(0) + actions.min(0)) / 2).tolist(),
        action_scale=np.maximum((actions.max(0) - actions.min(0)) / 2, 1e-6).tolist(),
    )


def normalize(values, stats, kind):
    center, scale = ("state_mean", "state_std") if kind == "state" else ("action_center", "action_scale")
    return ((values - np.asarray(stats[center])) / np.asarray(stats[scale])).astype(np.float32)


def denormalize_action(values, stats):
    return values * np.asarray(stats["action_scale"]) + np.asarray(stats["action_center"])


def select_episodes(rows, per_class, valid_per_class):
    if per_class < 0 or valid_per_class < 0:
        raise ValueError("Episode counts cannot be negative")
    selected, valid = [], []
    for _, group in rows.groupby("task"):
        group = group.sort_values("episode_index")
        if per_class:
            if per_class > len(group):
                raise ValueError("Requested more episodes than available")
            group = group.iloc[np.linspace(0, len(group) - 1, per_class, dtype=int)]
        ids = group.episode_index.astype(int).tolist()
        if valid_per_class >= len(ids):
            raise ValueError("Each task needs at least one training episode")
        selected.extend(ids)
        if valid_per_class:
            valid.extend(ids[-valid_per_class:])
    return sorted(selected), sorted(valid)


def episode_data(root, info, row):
    path = root / info["data_path"].format(
        chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"])
    )
    frame = pd.read_parquet(path)
    frame = frame[frame.episode_index == row.episode_index].sort_values("frame_index")
    n = int(row.length)
    if len(frame) != n or not np.array_equal(frame.frame_index, np.arange(n)):
        raise ValueError(f"Episode {row.episode_index}: invalid frame indexing")
    if not np.allclose(frame.timestamp, np.arange(n) / info["fps"], atol=1e-4):
        raise ValueError("Data timestamps do not match frame rate")
    states, actions = np.stack(frame["observation.state"]), np.stack(frame.action)
    if states.shape != (n, 10) or actions.shape != (n, 10) or not np.isfinite([states, actions]).all():
        raise ValueError("Expected finite 10-joint state and original action targets")
    return states, actions, path


def video_path(root, info, row, key):
    return root / info["video_path"].format(
        video_key=key,
        chunk_index=int(row[f"videos/{key}/chunk_index"]),
        file_index=int(row[f"videos/{key}/file_index"]),
    )


def contact_sheet(rgb, boxes, frame_ids, path):
    tiles = []
    for i in frame_ids:
        tile = cv2.cvtColor(rgb[i], cv2.COLOR_RGB2BGR)
        tile = cv2.resize(tile, (256, 256))
        for j, box in enumerate(boxes[i]):
            x1, y1, x2, y2 = np.rint(box * 2).astype(int)
            cv2.rectangle(tile, (x1, y1), (x2, y2), (0, 220, 255), 1)
            cv2.putText(tile, str(j), (x1, max(12, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        tile = cv2.copyMakeBorder(tile, 22, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(
            tile, f"frame {i}; top-15 proposals", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
        )
        tiles.append(tile)
    if not cv2.imwrite(str(path), np.concatenate(tiles, axis=1)):
        raise IOError(path)


def prepare(args):
    root, output = args.root.resolve(), args.output.resolve()
    info = json.loads((root / "meta/info.json").read_text())
    rows = pd.concat([pd.read_parquet(p) for p in sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))])
    if rows.episode_index.duplicated().any() or len(rows) != info["total_episodes"]:
        raise ValueError("Episode metadata mismatch")
    rows["task"] = rows.tasks.map(lambda values: "|".join(values))
    selected, valid = select_episodes(rows, args.per_class, args.valid_per_class)
    rows = rows[rows.episode_index.isin(selected)].sort_values("episode_index")
    train_ids = sorted(set(selected) - set(valid))
    records, signatures = {}, []
    for _, row in rows.iterrows():
        state, action, path = episode_data(root, info, row)
        records[int(row.episode_index)] = (state, action)
        signatures.append([str(path.relative_to(root)), sha256(path)])
        for key in VIEWS.values():
            path = video_path(root, info, row, key)
            stat = path.stat()
            signatures.append(
                [
                    str(path.relative_to(root)),
                    stat.st_size,
                    stat.st_mtime_ns,
                    float(row[f"videos/{key}/from_timestamp"]),
                    float(row[f"videos/{key}/to_timestamp"]),
                ]
            )
    stats = normalizer(
        np.concatenate([records[i][0] for i in train_ids]), np.concatenate([records[i][1] for i in train_ids])
    )
    spec = dict(
        version=1,
        root=str(root),
        revision=REVISION,
        weight_sha256=WEIGHT_SHA256,
        selected=selected,
        train=train_ids,
        valid=valid,
        signatures=signatures,
        fps=info["fps"],
        image_size=128,
        top_k=15,
        nms=0.05,
        area=(16, 4096),
        stats=stats,
        action_names=info["features"]["action"]["names"],
        state_names=info["features"]["observation.state"]["names"],
    )
    serialized = json.dumps(spec, sort_keys=True)
    if output.exists():
        with h5py.File(output, "r") as f:
            if f.attrs["spec"] != serialized or not f.attrs.get("complete", False):
                raise ValueError("Existing cache differs; choose another output path")
        print(f"Verified existing cache: {output}", flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    previews = output.with_suffix(".previews")
    previews.mkdir(exist_ok=True)
    partial = output.with_suffix(".partial.hdf5")
    detector = predictor(args.device)
    with h5py.File(partial, "a") as f:
        if "spec" in f.attrs and f.attrs["spec"] != serialized:
            raise ValueError("Partial cache configuration changed; choose another path")
        f.attrs["spec"] = serialized
        data = f.require_group("data")
        data.attrs["env_args"] = "{}"
        for _, row in rows.iterrows():
            ep, n = int(row.episode_index), int(row.length)
            name = f"demo_{ep}"
            if name in data and data[name].attrs.get("complete", False):
                print(f"Reuse episode {ep}", flush=True)
                continue
            if name in data:
                del data[name]
            caps = {}
            for obs, key in VIEWS.items():
                cap = cv2.VideoCapture(str(video_path(root, info, row, key)))
                if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - info["fps"]) > 0.01:
                    raise ValueError(f"Video open/fps failure: {ep}, {key}")
                start = float(row[f"videos/{key}/from_timestamp"])
                stop = float(row[f"videos/{key}/to_timestamp"])
                if abs((stop - start) * info["fps"] - n) > 1.01:
                    raise ValueError("Video metadata duration mismatch")
                cap.set(cv2.CAP_PROP_POS_FRAMES, round(start * info["fps"]))
                caps[obs] = cap
            images = {key: [] for key in VIEWS}
            boxes, counts, timings = [], [], []
            try:
                for i in range(n):
                    for obs, cap in caps.items():
                        ok, bgr = cap.read()
                        if not ok:
                            raise ValueError(f"Video ended early: episode {ep}, frame {i}, {obs}")
                        images[obs].append(resize_rgb(bgr))
                    begin = time.perf_counter()
                    box, count = proposals(detector, images["agentview_rgb"][-1])
                    timings.append(time.perf_counter() - begin)
                    boxes.append(box)
                    counts.append(count)
                    if (i + 1) % 200 == 0:
                        print(
                            f"episode={ep} frames={i + 1}/{n} detector_ms={np.mean(timings) * 1000:.1f}",
                            flush=True,
                        )
            finally:
                for cap in caps.values():
                    cap.release()
            group = data.create_group(name)
            group.attrs["num_samples"] = n
            group.attrs["task"] = row.task
            obs = group.create_group("obs")
            for key, value in images.items():
                obs.create_dataset(key, data=np.asarray(value), chunks=(1, 128, 128, 3), compression="lzf")
            obs.create_dataset(BOX_KEY, data=np.asarray(boxes))
            state, action = records[ep]
            obs.create_dataset("joint_states", data=normalize(state, stats, "state"))
            group.create_dataset("actions", data=normalize(action, stats, "action"))
            group.create_dataset("physical_actions", data=action)
            group.create_dataset("physical_states", data=state)
            group.create_dataset("proposal_counts", data=counts)
            group.create_dataset("proposal_seconds", data=timings)
            contact_sheet(
                images["agentview_rgb"],
                boxes,
                np.linspace(0, n - 1, 5, dtype=int),
                previews / f"episode-{ep:03d}.jpg",
            )
            group.attrs["complete"] = True
            f.flush()
            print(f"Completed episode {ep} ({row.task}); valid boxes min={min(counts)}", flush=True)
        mask = f.require_group("mask")
        for key, ids in (("train", train_ids), ("valid", valid)):
            if key in mask:
                del mask[key]
            mask.create_dataset(key, data=np.asarray([f"demo_{i}".encode() for i in ids], dtype="S32"))
        f["data"].attrs["total"] = sum(int(g.attrs["num_samples"]) for g in data.values())
        f.attrs["complete"] = True
    partial.replace(output)
    output.with_suffix(".json").write_text(json.dumps(spec, indent=2) + "\n")
    print(f"READY {output}; train={train_ids}; valid={valid}", flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(device, config=None):
    imports()
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from viola_bc.policy import CenterNetSpatialTemporalPolicy

    if config is None:
        with initialize_config_dir(config_dir=str(UPSTREAM / "viola_bc/configs"), version_base="1.1"):
            cfg = compose(config_name="config", overrides=["experiment=stack_viola"])
        config = cfg.algo.model
        config.grouping.network_kwargs.joint_states_dim = 10
        config.grouping.network_kwargs.use_gripper = False
    else:
        config = OmegaConf.create(config)
    OmegaConf.set_struct(config, False)
    shapes = dict(agentview_rgb=(3, 128, 128), eye_in_hand_rgb=(3, 128, 128), joint_states=(10,))
    shapes[BOX_KEY] = (15, 4)
    # Upstream chooses this device at import time; keep its masks with the model.
    import viola_bc.modules as modules

    modules.DEVICE = torch.device(device)
    model = CenterNetSpatialTemporalPolicy(config, {"all_shapes": shapes, "ac_dim": 10}).to(device)
    return model, OmegaConf.to_container(config, resolve=True)


def sequence_dataset(path, split):
    imports()
    # Pinned robomimic uses the removed np.bool alias for its padding mask.
    if "bool" not in np.__dict__:
        np.bool = np.bool_
    from robomimic.utils.dataset import SequenceDataset
    import robomimic.utils.obs_utils as obs_utils

    obs_utils.initialize_obs_utils_with_obs_specs(
        {"obs": {"rgb": list(VIEWS), "low_dim": ["joint_states", BOX_KEY]}}
    )
    return SequenceDataset(
        hdf5_path=str(path),
        obs_keys=OBS_KEYS,
        dataset_keys=["actions"],
        load_next_obs=False,
        frame_stack=1,
        seq_length=10,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode="low_dim",
        hdf5_use_swmr=False,
        hdf5_normalize_obs=False,
        filter_by_attribute=split,
    )


def to_device(batch, device):
    if isinstance(batch, dict):
        return {key: to_device(value, device) for key, value in batch.items()}
    return batch.to(device=device, dtype=torch.float32, non_blocking=True)


@torch.no_grad()
def evaluate(model, loader, stats, device, batches):
    model.eval()
    sums = dict(nll=0.0, normalized_mae=0.0, native_unit_mae=0.0)
    count = 0
    scale = torch.tensor(stats["action_scale"], device=device)
    for idx, raw in enumerate(loader):
        if idx >= batches:
            break
        data = to_device(raw, device)
        dist = model(data)
        error = (dist.mean - data["actions"]).abs()
        size = len(error)
        values = (-dist.log_prob(data["actions"]).mean(), error.mean(), (error * scale).mean())
        for key, value in zip(sums, values):
            if not torch.isfinite(value):
                raise FloatingPointError(f"Non-finite validation {key}")
            sums[key] += value.item() * size
        count += size
    return {key: value / count for key, value in sums.items()}


def train(args):
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite training run: {args.output}")
    seed_all(args.seed)
    torch.set_num_threads(4)
    with h5py.File(args.dataset, "r") as f:
        if not f.attrs.get("complete", False):
            raise ValueError("Incomplete proposal cache")
        spec = json.loads(f.attrs["spec"])
    model, config = build_model(args.device)
    dataset = sequence_dataset(args.dataset, "train")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    validation = None
    if spec["valid"]:
        valid = sequence_dataset(args.dataset, "valid")
        # Spread validation windows across all held-out trajectories and fix them for every evaluation.
        indices = np.linspace(
            0, len(valid) - 1, min(len(valid), args.eval_batches * args.batch_size), dtype=int
        )
        validation = torch.utils.data.DataLoader(
            torch.utils.data.Subset(valid, indices), batch_size=args.batch_size, num_workers=0
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, eta_min=1e-6)
    steps = args.steps if args.steps else args.epochs * len(loader)
    if steps < 1:
        raise ValueError("Training needs at least one step")
    args.output.mkdir(parents=True)
    run = dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "func"},
        upstream_revision=REVISION,
        config=config,
        dataset_spec=spec,
        steps=steps,
        parameters=sum(p.numel() for p in model.parameters()),
        torch=torch.__version__,
    )
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    before = (
        evaluate(model, validation, spec["stats"], args.device, args.eval_batches) if validation else None
    )
    print(f"Initial validation: {before}; training steps={steps}", flush=True)
    iterator, recent, start = iter(loader), [], time.perf_counter()
    with (args.output / "metrics.jsonl").open("w", buffering=1) as log:
        log.write(json.dumps(dict(step=0, validation=before)) + "\n")
        for step in range(1, steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            model.train()
            data = to_device(batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            dist = model(data)
            loss = -dist.log_prob(data["actions"]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite NLL at {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step((step - 1) / len(loader))
            recent.append(loss.item())
            if step % args.log_every == 0 or step == steps:
                record = dict(
                    step=step,
                    train_nll=float(np.mean(recent)),
                    grad_norm=norm.item(),
                    elapsed_seconds=time.perf_counter() - start,
                    lr=optimizer.param_groups[0]["lr"],
                )
                if validation and (step % args.eval_every == 0 or step == steps):
                    # Validation must not consume training augmentation/dropout RNG.
                    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                        record["validation"] = evaluate(
                            model, validation, spec["stats"], args.device, args.eval_batches
                        )
                log.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
                recent.clear()
            if step % args.save_every == 0 or step == steps:
                payload = dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    config=config,
                    dataset_spec=spec,
                    step=step,
                    revision=REVISION,
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(),
                )
                temp = args.output / "checkpoint.tmp"
                torch.save(payload, temp)
                temp.replace(args.output / f"checkpoint-{step:06d}.pt")
    model.eval()
    probe = to_device(next(iter(loader)), args.device)
    with torch.no_grad():
        expected = model(probe).mean.cpu()
    restored, _ = build_model(args.device, config)
    saved = torch.load(
        args.output / f"checkpoint-{steps:06d}.pt", map_location=args.device, weights_only=False
    )
    restored.load_state_dict(saved["model"], strict=True)
    restored.eval()
    with torch.no_grad():
        actual = restored(probe).mean.cpu()
    max_error = (expected - actual).abs().max().item()
    if max_error > 1e-6:
        raise AssertionError(f"Checkpoint reload mismatch: {max_error}")
    summary = dict(
        completed_steps=steps,
        initial_validation=before,
        final_record=record,
        checkpoint_reload_max_error=max_error,
        elapsed_seconds=time.perf_counter() - start,
        checkpoint=str((args.output / f"checkpoint-{steps:06d}.pt").resolve()),
        status="training_chain_verified_not_robot_success",
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


class VIOLAInference:
    """RGB + current state -> native dataset action; no robot connection.

    Reset at episode boundaries. Temporal features are cached by upstream VIOLA.
    Output is a sampled GMM action, as in the official policy, not its mean.
    """

    def __init__(self, checkpoint, device="cuda"):
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if payload["revision"] != REVISION:
            raise ValueError("Unsupported upstream revision")
        self.model, _ = build_model(device, payload["config"])
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.eval()
        self.stats = payload["dataset_spec"]["stats"]
        self.detector = predictor(device)
        self.reset()

    def reset(self):
        self.model.reset()

    @torch.no_grad()
    def predict(self, front_bgr, side_bgr, state):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (10,) or not np.isfinite(state).all():
            raise ValueError("Expected a finite 10-joint state in dataset-native units")
        front, side = resize_rgb(front_bgr), resize_rgb(side_bgr)
        boxes, _ = proposals(self.detector, front)
        self.last_boxes = boxes
        obs = {
            "agentview_rgb": torch.from_numpy(front.transpose(2, 0, 1).copy()).float() / 255.0,
            "eye_in_hand_rgb": torch.from_numpy(side.transpose(2, 0, 1).copy()).float() / 255.0,
            "joint_states": torch.from_numpy(normalize(np.asarray(state), self.stats, "state")),
            BOX_KEY: torch.from_numpy(boxes),
        }
        action = denormalize_action(self.model.get_action({"obs": obs}), self.stats)
        if action.shape != (10,) or not np.isfinite(action).all():
            raise ValueError("Invalid policy output")
        return action


def verify(args):
    """Replay recorded observations only; never connect to robot hardware."""
    torch.set_num_threads(4)
    seed_all(1000)
    policy = VIOLAInference(args.checkpoint, args.device)
    with h5py.File(args.dataset, "r") as f:
        spec = json.loads(f.attrs["spec"])
        checkpoint_spec = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["dataset_spec"]
        if checkpoint_spec != spec:
            raise ValueError("Replay dataset does not match checkpoint")
        ep = (spec["valid"] or spec["train"])[0]
        group = f[f"data/demo_{ep}"]
        start = int(group.attrs["num_samples"]) // 3
        states = group["physical_states"][start : start + args.frames]
        cached_front = group["obs/agentview_rgb"][start : start + args.frames]
        cached_side = group["obs/eye_in_hand_rgb"][start : start + args.frames]
        cached_boxes = group[f"obs/{BOX_KEY}"][start : start + args.frames]
    root = Path(spec["root"])
    info = json.loads((root / "meta/info.json").read_text())
    rows = pd.concat([pd.read_parquet(p) for p in sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))])
    row = rows[rows.episode_index == ep].iloc[0]
    caps = []
    for key in VIEWS.values():
        cap = cv2.VideoCapture(str(video_path(root, info, row, key)))
        cap.set(
            cv2.CAP_PROP_POS_FRAMES, round(float(row[f"videos/{key}/from_timestamp"]) * info["fps"]) + start
        )
        caps.append(cap)
    latencies, actions, max_box_error = [], [], 0.0
    try:
        for i, state in enumerate(states):
            frames = [cap.read() for cap in caps]
            if not all(ok for ok, _ in frames):
                raise IOError("Replay video ended early")
            front, side = [frame for _, frame in frames]
            np.testing.assert_array_equal(resize_rgb(front), cached_front[i])
            np.testing.assert_array_equal(resize_rgb(side), cached_side[i])
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            begin = time.perf_counter()
            action = policy.predict(front, side, state)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - begin)
            actions.append(action)
            max_box_error = max(max_box_error, float(np.max(np.abs(policy.last_boxes - cached_boxes[i]))))
    finally:
        for cap in caps:
            cap.release()
    if max_box_error > 1e-3:
        raise AssertionError(f"Online/offline proposals differ: {max_box_error}")
    if len(latencies) <= 10:
        raise ValueError("Need more than 10 replay frames to benchmark warmed history")
    report = dict(
        episode=ep,
        start_frame=start,
        frames=len(actions),
        box_comparison_frames=len(actions),
        rgb_offline_online_exact=True,
        bbox_offline_online_max_error=max_box_error,
        action_shape=list(np.shape(actions)),
        actions_finite=bool(np.isfinite(actions).all()),
        history_queue_length=len(policy.model.queue),
        end_to_end_ms_p50=float(np.percentile(latencies[10:], 50) * 1000),
        end_to_end_ms_p95=float(np.percentile(latencies[10:], 95) * 1000),
        includes="resize + Detic proposals + cached-history VIOLA + action denormalization; excludes camera/robot I/O",
        gpu=torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--root", type=Path, default=ROOT / "data/finaldata_seg")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--per-class", type=int, default=4, help="0 uses all episodes")
    prep.add_argument("--valid-per-class", type=int, default=1)
    prep.add_argument("--device", default="cuda")
    prep.set_defaults(func=prepare)
    fit = commands.add_parser("train")
    fit.add_argument("--dataset", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--steps", type=int, default=0, help="Overrides epochs when nonzero")
    fit.add_argument("--epochs", type=int, default=51)
    fit.add_argument("--batch-size", type=int, default=16)
    fit.add_argument("--workers", type=int, default=4)
    fit.add_argument("--seed", type=int, default=1000)
    fit.add_argument("--log-every", type=int, default=50)
    fit.add_argument("--eval-every", type=int, default=250)
    fit.add_argument("--eval-batches", type=int, default=12)
    fit.add_argument("--save-every", type=int, default=5000)
    fit.add_argument("--device", default="cuda")
    fit.set_defaults(func=train)
    check = commands.add_parser("verify")
    check.add_argument("--checkpoint", type=Path, required=True)
    check.add_argument("--dataset", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--frames", type=int, default=60)
    check.add_argument("--device", default="cuda")
    check.set_defaults(func=verify)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

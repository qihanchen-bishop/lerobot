#!/usr/bin/env python
"""Summarize and rank the organized Bettersetup-family training outputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


CATEGORY_LABELS = {
    "act": "ACT baselines",
    "camera_embedding": "Camera embedding",
    "diffusion": "Diffusion Policy",
    "mask_ablation": "Legacy mask ablations",
    "semantic": "Semantic input",
    "action_semantic": "Action-supervised semantics",
    "stage": "Stage-conditioned policies",
    "ssact": "SSACT",
    "view_fusion": "View fusion",
    "action_representation": "Action representation",
    "delta": "Follower Delta",
    "smoke": "Smoke tests",
}
FORMAL_CATEGORIES = tuple(name for name in CATEGORY_LABELS if name != "smoke")


@dataclass(frozen=True)
class RunSummary:
    category: str
    name: str
    dataset: str
    step: int
    metric: str
    late_mean: float
    final: float
    complete: bool

    @property
    def path(self) -> str:
        return f"{self.category}/{self.name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=Path("outputs/train"))
    parser.add_argument("--window-steps", type=int, default=10_000)
    parser.add_argument("--expected-steps", type=int, default=100_000)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def dataset_name(run_dir: Path) -> str:
    mask_config = load_json(run_dir / "mask_act_run_config.json")
    identifier = str(mask_config.get("root") or mask_config.get("repo_id") or "")
    if not identifier:
        configs = sorted(run_dir.glob("checkpoints/*/pretrained_model/train_config.json"))
        if configs:
            dataset = load_json(configs[-1]).get("dataset", {})
            identifier = str(dataset.get("repo_id") or dataset.get("root") or "")
    combined = f"{run_dir.name} {identifier}".lower()
    if "newdata_3object" in combined or run_dir.name == "DP-1F-Full":
        return "newdata_3object"
    if "bettersetup_v5" in combined:
        return "bettersetup_v5"
    if "bettersetup_v4" in combined:
        return "bettersetup_v4"
    if "bettersetup" in combined:
        return "bettersetup"
    return "unknown"


def metric_name(run_name: str) -> str:
    if run_name.startswith("DP-"):
        return "diffusion_loss"
    if run_name.startswith("AM-ACT"):
        return "reconstructed_action_l1"
    if "FAnchorDelta" in run_name:
        return "anchor_delta_l1"
    if "FDelta" in run_name:
        return "cumulative_delta_l1"
    return "l1_loss"


def metric_key(metric: str) -> str:
    if metric == "diffusion_loss":
        return "loss"
    if metric in {"anchor_delta_l1", "cumulative_delta_l1"}:
        return "l1_loss"
    return metric


def checkpoint_complete(run_dir: Path, expected_steps: int) -> bool:
    return (
        run_dir / "checkpoints" / f"{expected_steps:06d}" / "pretrained_model"
    ).is_dir() or (
        run_dir / f"checkpoint_step_{expected_steps:06d}" / "training_state.pt"
    ).is_file()


def summarize_run(
    category: str,
    run_dir: Path,
    window_steps: int,
    expected_steps: int,
) -> RunSummary | None:
    metrics_path = run_dir / "metrics" / "train_metrics.jsonl"
    if not metrics_path.is_file():
        return None
    records = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record.get("step"), (int, float)):
            records.append(record)
    if not records:
        return None

    metric = metric_name(run_dir.name)
    key = metric_key(metric)
    final_record = max(records, key=lambda record: int(record["step"]))
    final_step = int(final_record["step"])
    window_start = final_step - window_steps
    values = [
        float(record[key])
        for record in records
        if int(record["step"]) > window_start and key in record
    ]
    if not values or key not in final_record:
        return None
    return RunSummary(
        category=category,
        name=run_dir.name,
        dataset=dataset_name(run_dir),
        step=final_step,
        metric=metric,
        late_mean=mean(values),
        final=float(final_record[key]),
        complete=final_step >= expected_steps and checkpoint_complete(run_dir, expected_steps),
    )


def discover_runs(train_root: Path, window_steps: int, expected_steps: int) -> list[RunSummary]:
    runs = []
    for category in CATEGORY_LABELS:
        category_dir = train_root / category
        if not category_dir.is_dir():
            continue
        for run_dir in sorted(category_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name in {"dataset_views", "queue_logs", "launcher_logs"}:
                continue
            summary = summarize_run(category, run_dir, window_steps, expected_steps)
            if summary is not None:
                runs.append(summary)
    return runs


def table(rows: list[RunSummary], include_category: bool = True) -> list[str]:
    if include_category:
        lines = [
            "| Rank | Experiment | Method | Dataset | Metric | Last-window mean | Final | Status |",
            "| ---: | --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    else:
        lines = [
            "| Rank | Experiment | Dataset | Metric | Last-window mean | Final | Status |",
            "| ---: | --- | --- | --- | ---: | ---: | --- |",
        ]
    for rank, run in enumerate(sorted(rows, key=lambda item: item.late_mean), start=1):
        status = "complete" if run.complete else f"step {run.step}"
        if include_category:
            values = [
                str(rank), f"`{run.name}`", CATEGORY_LABELS[run.category], run.dataset,
                f"`{run.metric}`", f"{run.late_mean:.6f}", f"{run.final:.6f}", status,
            ]
        else:
            values = [
                str(rank), f"`{run.name}`", run.dataset, f"`{run.metric}`",
                f"{run.late_mean:.6f}", f"{run.final:.6f}", status,
            ]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def render_master(runs: list[RunSummary], window_steps: int) -> str:
    formal = [run for run in runs if run.category in FORMAL_CATEGORIES]
    standard = [
        run for run in formal
        if run.dataset.startswith("bettersetup")
        and run.category not in {"delta", "diffusion"}
    ]
    legacy = [
        run for run in formal
        if run.dataset == "newdata_3object" and run.category != "diffusion"
    ]
    anchor_delta = [run for run in formal if run.metric == "anchor_delta_l1"]
    cumulative_delta = [run for run in formal if run.metric == "cumulative_delta_l1"]
    diffusion = [run for run in formal if run.category == "diffusion"]

    lines = [
        "# Bettersetup Training Results",
        "",
        "Generated by `mycode/summarize_bettersetup_training.py`.",
        "",
        "## Organization",
        "",
    ]
    for category in CATEGORY_LABELS:
        count = sum(run.category == category for run in runs)
        lines.append(f"- `{category}/`: {CATEGORY_LABELS[category]} ({count} runs with metrics)")
    lines += [
        "",
        "## Ranking Rules",
        "",
        f"- Ranking uses the mean metric over each run's final {window_steps:,} training steps, not one noisy final batch.",
        "- Standard ACT-family runs use normalized absolute-action `l1_loss`.",
        "- AM-ACT uses decoded `reconstructed_action_l1`, which is the closest comparable absolute-action metric.",
        "- Bettersetup, v4, and v5 have identical state/action rows and action statistics, so their standard absolute-action L1 values can be compared.",
        "- Follower Delta targets have different statistics and are ranked only inside the Delta section.",
        "- Diffusion noise-prediction loss is not an action L1 and is ranked only inside the Diffusion section.",
        "- Training error does not establish real-robot policy quality.",
        "",
        "## Bettersetup Absolute-Action Ranking",
        "",
        *table(standard),
        "",
        "## Fixed-Anchor Follower Delta Ranking",
        "",
        *table(anchor_delta),
        "",
        "## Cumulative Follower Delta Ranking",
        "",
        *table(cumulative_delta),
        "",
        "## Diffusion Ranking",
        "",
        *table(diffusion),
        "",
        "## Legacy Newdata Ranking",
        "",
        *table(legacy),
        "",
    ]
    return "\n".join(lines)


def render_category(category: str, runs: list[RunSummary], window_steps: int) -> str:
    lines = [
        f"# {CATEGORY_LABELS[category]}",
        "",
        f"Sorted by the mean reported metric over the final {window_steps:,} training steps.",
        "Rankings are separated when the dataset family or target metric differs.",
        "",
    ]
    groups: dict[tuple[str, str], list[RunSummary]] = {}
    for run in runs:
        dataset_group = "bettersetup family" if run.dataset.startswith("bettersetup") else run.dataset
        groups.setdefault((dataset_group, run.metric), []).append(run)
    for (dataset_group, metric), group_runs in sorted(groups.items()):
        if len(groups) > 1:
            lines.extend([f"## {dataset_group}: `{metric}`", ""])
        lines.extend([*table(group_runs, include_category=False), ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    runs = discover_runs(args.train_root, args.window_steps, args.expected_steps)
    master = args.train_root / "BETTERSETUP_RESULTS.md"
    master.write_text(render_master(runs, args.window_steps), encoding="utf-8")
    for category in CATEGORY_LABELS:
        category_dir = args.train_root / category
        if not category_dir.is_dir():
            continue
        category_runs = [run for run in runs if run.category == category]
        (category_dir / "README.md").write_text(
            render_category(category, category_runs, args.window_steps), encoding="utf-8"
        )
    print(f"Wrote {master} with {len(runs)} ranked runs.")


if __name__ == "__main__":
    main()

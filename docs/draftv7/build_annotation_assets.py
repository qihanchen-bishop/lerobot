"""Freeze audited finaldata annotation evidence and rebuild English paper assets.

Run --refresh to read the SAM2 reports; ordinary rebuilds use the frozen snapshot.
No source videos, annotations, or models are modified.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

P = Path(__file__).resolve().parent
DATA = P / "results/annotation_finaldata"
FIG = P / "figures/annotation_finaldata"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def refresh(root):
    sources = {}

    def read(path):
        sources[str(path)] = digest(path)
        return json.loads(path.read_text())

    result = {"views": {}, "scope": "Recorded evaluations, not new inference or manual relabeling."}
    for view, sub in [("front", ""), ("side", "side/")]:
        run = root / f"segdata/temp/finaldata/iterunet/{sub}iter_r1"
        cfg = read(run / "config.json")
        state = read(run / "iteration.json")
        overall = read(run / "overalltest/report.json")
        initial = read(root / f"segdata/temp/finaldata/manual_seed/unet_evaluations/{sub}eval_r1/report.json")
        assert initial["models"][0]["model_sha256"] == cfg["initial_model_sha256"]
        assert initial["quality_policy"] == overall["quality_policy"] == cfg["quality_policy"]
        assert initial.get("filter_settings") == overall.get("filter_settings")
        assert state["state"] == "done" or all(r["state"] == "done" for r in state["rounds"])
        for report in overall["source_reports"]:
            path = Path(report["evaluation_dir"]) / "report.json"
            assert digest(path) == report["report_sha256"]
            sources[str(path)] = report["report_sha256"]
        data = {k: cfg[k] for k in ["batches", "exclusions", "training", "gate", "quality_policy"]}
        data["filter_settings"] = overall["filter_settings"]
        data["rounds"] = []
        seen, accepted, frames = set(), 0, 0
        for row in state["rounds"]:
            episodes = []
            for rec in row["records"]:
                ep = rec["item"]["episode_index"]
                assert ep not in seen
                seen.add(ep)
                selection = rec["selection"]
                n = sum(s["accepted"] for s in selection)
                assert n == rec["accepted_frames"]
                assert len(selection) == rec["total_frames"] == rec["item"]["frame_count"]
                assert [s["frame"] for s in selection] == list(range(len(selection)))
                episodes.append({"episode": ep, "frames": len(selection), "accepted": n,
                                 "teacher_sha256": rec["teacher_model_sha256"]})
                accepted += n
                frames += len(selection)
            assert set(row["episodes"]) == {e["episode"] for e in episodes}
            assert accepted == row["accepted_frames"] == sum(row["counts"].values())
            assert frames == row["total_frames"] and len(seen) == row["cumulative_videos"]
            data["rounds"].append({k: row[k] for k in ["round", "cumulative_videos", "accepted_frames", "total_frames", "counts"]} | {"new_episodes": episodes})
        data["models"] = []
        for generation, model in enumerate([initial["models"][0]] + overall["models"]):
            assert model.get("generation", generation) == generation
            item = {"generation": generation, "sha256": model["model_sha256"],
                    "model_dir": model["model_dir"], "aggregate": model["aggregate"], "records": []}
            ancestry = model["provenance"]
            assert ancestry["verified"] and not ancestry["unknown_ancestry"]
            assert not set(cfg["exclusions"]["benchmark"]) & {r["episode_index"] for r in ancestry["all_influencing_episodes"]}
            item["provenance"] = ancestry
            for rec in model["records"]:
                item["records"].append({k: rec[k] for k in ["panel", "labels", "reference_role", "semantic", "temporal"]} |
                                       {"episode": rec["item"]["episode_index"], "frames": rec["item"]["frame_count"],
                                        "video_sha256": rec["item"].get("video_sha256")})
            data["models"].append(item)
        exported = P.parents[1] / f"data/finaldata_seg/models/unet_{view}"
        data["export_manifest"] = read(exported / "bundle_manifest.json")
        model_path = exported / "model.ts"
        sources[str(model_path)] = digest(model_path)
        assert sources[str(model_path)] == data["models"][-1]["sha256"]
        data["exported_model_sha256"] = sources[str(model_path)]
        result["views"][view] = data
    for name in ["unet_iteration.py", "video_segmentation_quality.py", "evaluate_workbench_unet.py"]:
        path = root / "tools" / name
        sources[str(path)] = digest(path)
    result["source_sha256"] = sources
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "snapshot.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


def miou(matrix):
    n = len(matrix)
    scores = []
    for c in range(1, n):
        union = sum(matrix[c]) + sum(row[c] for row in matrix) - matrix[c][c]
        if union:
            scores.append(matrix[c][c] / union)
    return sum(scores) / len(scores)


def validate(snapshot):
    rows = []
    for view, data in snapshot["views"].items():
        seeds = set(data["exclusions"]["manual_seed"])
        holdout = set(data["exclusions"]["benchmark"])
        batches = data["batches"]
        pool = [ep for batch in batches for ep in batch]
        assert not seeds & holdout and not (seeds | holdout) & set(pool)
        assert len(pool) == len(set(pool)) == 102
        assert seeds | holdout | set(pool) == set(range(108))
        cumulative_n = cumulative_f = 0
        for r, batch in zip(data["rounds"], batches):
            assert set(batch) == {e["episode"] for e in r["new_episodes"]}
            cumulative_n += sum(e["accepted"] for e in r["new_episodes"])
            cumulative_f += sum(e["frames"] for e in r["new_episodes"])
            assert cumulative_n == r["accepted_frames"] == sum(r["counts"].values())
            assert cumulative_f == r["total_frames"]
        reference_videos = None
        for m in data["models"]:
            records = m["records"]
            benchmark = [r for r in records if r["panel"] == "benchmark"]
            assert {r["episode"] for r in benchmark} == holdout
            videos = {r["episode"]: r["video_sha256"] for r in benchmark}
            assert all(videos.values())
            if reference_videos is None:
                reference_videos = videos
            assert videos == reference_videos
            assert all(r["semantic"] is None and r["reference_role"] == "independent" for r in benchmark)
            for panel, aggregate in m["aggregate"].items():
                subset = [r for r in records if r["panel"] == panel]
                for branch, a in aggregate.items():
                    n = sum(r["frames"] for r in subset)
                    anomalies = sum(r["temporal"][branch]["anomaly_frames"] for r in subset)
                    assert n == a["frames"] and anomalies == a["anomaly_frames"]
                    assert math.isclose(anomalies / n, a["anomaly_rate"])
                    if a["semantic"]:
                        matrices = [r["semantic"][branch]["confusion_matrix"] for r in subset]
                        cm = [[sum(matrix[i][j] for matrix in matrices) for j in range(7)] for i in range(7)]
                        assert cm == a["semantic"]["confusion_matrix"]
                        assert math.isclose(miou(cm), a["semantic"]["foreground_miou"])
                        assert sum(map(sum, cm)) == n * 640 * 360
                        assert all(r["reference_role"] != "independent" for r in subset)
            a = m["aggregate"]["benchmark"]
            seed = m["aggregate"].get("manual_seed", {}).get("raw", {}).get("semantic")
            r = data["rounds"][m["generation"] - 1] if m["generation"] else None
            rows.append({"view": view, "generation": m["generation"],
                         "cumulative_pseudo_videos": r["cumulative_videos"] if r else 0,
                         "accepted_frames": r["accepted_frames"] if r else 0,
                         "candidate_frames": r["total_frames"] if r else 0,
                         "seed_raw_miou": seed["foreground_miou"] if seed else None,
                         "holdout_frames": a["raw"]["frames"],
                         "raw_anomaly_frames": a["raw"]["anomaly_frames"],
                         "filtered_anomaly_frames": a["filtered"]["anomaly_frames"]})
        assert data["exported_model_sha256"] == data["models"][-1]["sha256"]
    assert snapshot["views"]["front"]["batches"] == snapshot["views"]["side"]["batches"]
    return rows


def build(snapshot, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 9, "axes.titlesize": 10,
                         "axes.titleweight": "bold", "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                         "legend.fontsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": "#26343D", "axes.labelcolor": "#26343D", "axes.edgecolor": "#8C969E",
                         "pdf.fonttype": 42, "savefig.dpi": 300})
    FIG.mkdir(parents=True, exist_ok=True)
    outputs = []
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), layout="constrained")
    for view, color in [("front", "#0072B2"), ("side", "#D99026")]:
        rs = [r for r in rows if r["view"] == view]
        axes[0].plot([1, 2, 3], [r["accepted_frames"] / 1000 for r in rs[1:]], "o-", color=color, label=view.title(), ms=4)
        for ax, key in zip(axes[1:], ["raw_anomaly_frames", "filtered_anomaly_frames"]):
            ax.plot(range(4), [100 * r[key] / r["holdout_frames"] for r in rs], "o-", color=color, ms=4)
    for i, ax in enumerate(axes):
        ax.grid(axis="y", color="#E5E9EC", lw=.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("Iteration")
        ticks = [1, 2, 3] if i == 0 else list(range(4))
        ax.set_xticks(ticks, [f"R{x}" for x in ticks])
        ax.set_ylim(bottom=0)
    axes[0].set_title("(a) Pseudo-label expansion", loc="left")
    axes[0].set_ylabel("Accepted frames (thousands)")
    axes[0].set_ylim(0, 55)
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].set_title("(b) Raw predictions", loc="left")
    axes[2].set_title("(c) Structural filtering", loc="left")
    for ax in axes[1:]:
        ax.set_ylabel("Flagged frames (%)")
        ax.set_ylim(0, 14)
        ax.set_yticks([0, 4, 8, 12])
    for ext in ["pdf", "png"]:
        path = FIG / f"annotation_iteration_en.{ext}"
        kw = {"metadata": {"CreationDate": None, "ModDate": None}} if ext == "pdf" else {}
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kw)
        outputs.append(path)
    plt.close(fig)
    path = DATA / "metrics.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    outputs.append(path)
    table = []
    for r in rows:
        seed = "--" if r["seed_raw_miou"] is None else f"{100 * r['seed_raw_miou']:.2f}"
        raw = 100 * r["raw_anomaly_frames"] / r["holdout_frames"]
        filtered = 100 * r["filtered_anomaly_frames"] / r["holdout_frames"]
        table.append(f"{r['view'].title()} & R{r['generation']} & {seed} & {raw:.2f} & {filtered:.2f} " + r"\\")
    path = DATA / "iteration_rows_en.tex"
    path.write_text("\n".join(table) + "\n")
    outputs.append(path)
    manifest = {"source_sha256": {str((DATA / "snapshot.json").relative_to(P)): digest(DATA / "snapshot.json")},
                "output_sha256": {str(path.relative_to(P)): digest(path) for path in outputs}}
    (DATA / "assets_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--sam2-root", type=Path, default=Path("/home/qihan/data/sam2"))
    args = parser.parse_args()
    if args.refresh:
        refresh(args.sam2_root)
    snapshot = json.loads((DATA / "snapshot.json").read_text())
    rows = validate(snapshot)
    build(snapshot, rows)
    print(json.dumps(rows, indent=2))

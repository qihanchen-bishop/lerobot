"""Rebuild English paper figures and tables from the frozen trial snapshot."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

P = Path(__file__).resolve().parent
OUT = P / "figures/results_20261023"
OUT.mkdir(parents=True, exist_ok=True)
METHODS = ["ACT", "DP", "RGBQtoken", "segNoQtoken", "viewQtoken"]
NAMES = dict(zip(METHODS, ["ACT", "DP", "ACT + Q", "ACT + S", "Ours (S + Q)"]))
COLORS = dict(zip(METHODS, ["#7A8691", "#D99026", "#0072B2", "#B76D9C", "#009E73"]))
OBJECTS = ["cube", "paperball", "screw", "crayon"]
OBJECT_NAMES = ["Cube", "Paper ball", "Screw", "Crayon*"]
RATE_MAP = LinearSegmentedColormap.from_list("success", ["#F2F5F7", "#8ABCC3", "#194F62"])
# Bundled Times-style TrueType avoids CFF embedding warnings in Matplotlib PDFs.
FONT = "STIXGeneral"
LABEL_SIZE = 9
SMALL_SIZE = 8.5
DENSE_SIZE = 8
plt.rcParams.update({
    "font.family": FONT, "font.size": LABEL_SIZE, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlepad": 8,
    "axes.labelsize": LABEL_SIZE, "axes.labelpad": 4,
    "xtick.labelsize": SMALL_SIZE, "ytick.labelsize": SMALL_SIZE,
    "legend.fontsize": SMALL_SIZE, "legend.labelspacing": .4,
    "legend.handlelength": 1.7, "legend.handletextpad": .5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8C969E", "axes.labelcolor": "#26343D",
    "text.color": "#26343D", "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300,
})
with (P / "results/trials.csv").open(encoding="utf-8-sig", newline="") as f:
    ROWS = list(csv.DictReader(f))
S = json.loads((P / "results/summary.json").read_text())
A = {(r["method"], r["object"]): r for r in S["summary"]}
TIMING = {r["method"]: r for r in S["timing"] if r["object"] == "ALL"}
FAIL = {r["method"]: r for r in S["failure_stages"] if r["object"] == "ALL"}
trial_cells = Counter((r["method"], r["object"], r["x"], r["y"]) for r in ROWS)
assert len(ROWS) == 480 and len(trial_cells) == 240 and set(trial_cells.values()) == {2}


def subset(method, obj=None, x=None, y=None):
    return [r for r in ROWS if r["method"] == method
            and (obj is None or r["object"] == obj)
            and (x is None or int(r["x"]) == x)
            and (y is None or int(r["y"]) == y)]


def count(method, obj=None, x=None, y=None):
    return sum(int(r["success"]) for r in subset(method, obj, x, y))


for m in METHODS:
    for o in OBJECTS + ["ALL"]:
        rs = subset(m, None if o == "ALL" else o)
        assert len(rs) == A[m, o]["n"]
        assert sum(int(r["success"]) for r in rs) == A[m, o]["successes"]
        times = [float(r["control_duration_s"]) for r in rs]
        success_times = [float(r["control_duration_s"]) for r in rs if int(r["success"])]
        assert np.isclose(np.mean(times), A[m, o]["mean_all_control_s"])
        assert np.isclose(np.mean(success_times), A[m, o]["mean_success_control_s"])
    assert sum(int(r["success"]) and float(r["control_duration_s"]) <= 20
               for r in subset(m)) == TIMING[m]["success_by_20s"]

OUTPUTS = []


def save(fig, name):
    for ext in ["pdf", "png"]:
        path = OUT / f"{name}_en.{ext}"
        kwargs = {"metadata": {"CreationDate": None, "ModDate": None}} if ext == "pdf" else {}
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        OUTPUTS.append(path)
    plt.close(fig)


def rate_grid(ax, values, denominator, fontsize=9):
    rates = np.asarray(values) / denominator * 100
    im = ax.imshow(rates, vmin=0, vmax=100, cmap=RATE_MAP)
    for y, x in np.ndindex(rates.shape):
        ax.text(x, y, f"{values[y][x]}/{denominator}", ha="center", va="center",
                color="white" if rates[y, x] >= 70 else "#26343D", fontsize=fontsize)
    ax.set_xticks(range(4))
    ax.set_yticks(range(3))
    ax.tick_params(length=0)
    return im


fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.16, 2.8),
                            gridspec_kw={"width_ratios": [1, 1.7]}, layout="constrained")
v = np.array([A[m, "ALL"]["success_rate_pct"] for m in METHODS])
lo = np.array([A[m, "ALL"]["wilson95_low_pct"] for m in METHODS])
hi = np.array([A[m, "ALL"]["wilson95_high_pct"] for m in METHODS])
ax.barh(range(5), v, height=.62, color=[COLORS[m] for m in METHODS],
        xerr=[v-lo, hi-v], error_kw={"lw": .8, "capsize": 2, "ecolor": "#26343D"})
ax.set_yticks(range(5), [NAMES[m] for m in METHODS])
ax.invert_yaxis()
ax.set_xlim(0, 130)
ax.set_xticks([0, 50, 100])
ax.set_xlabel("Success rate (%)")
ax.set_title("(a) Overall: 96 trials per method", loc="left")
for i, m in enumerate(METHODS):
    ax.text(129, i, f"{count(m)}/96", ha="right", va="center", fontsize=SMALL_SIZE)
x = np.arange(4)
for i, m in enumerate(METHODS):
    bx.bar(x + (i-2)*.15, [100*count(m, o)/24 for o in OBJECTS], .145,
           color=COLORS[m], label=NAMES[m])
bx.set_xticks(x, OBJECT_NAMES)
bx.set_ylim(0, 106)
bx.set_yticks([0, 25, 50, 75, 100])
bx.set_ylabel("Success rate (%)")
bx.set_title("(b) Objects: 24 trials per method", loc="left")
bx.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, -.15),
          frameon=False, columnspacing=1)
save(fig, "overall_objects")

fig, ax = plt.subplots(figsize=(3.45, 2.35), layout="constrained")
order = [["ACT", "RGBQtoken"], ["segNoQtoken", "viewQtoken"]]
values = np.array([[count(m) for m in row] for row in order])
ax.imshow(values/96*100, vmin=0, vmax=100, cmap=RATE_MAP, aspect="auto")
for y, x in np.ndindex(values.shape):
    m = order[y][x]
    text_color = "white" if values[y, x]/96 >= .7 else "#26343D"
    ax.text(x, y-.12, NAMES[m], ha="center", va="center",
            fontsize=10, fontweight="bold", color=text_color)
    ax.text(x, y+.16, f"{values[y,x]}/96  ({values[y,x]/96*100:.2f}%)",
            ha="center", va="center", fontsize=9.5, color=text_color)
ax.set_xticks([0, 1], ["Without Q", "With Q"])
ax.set_yticks([0, 1], ["Without S", "With S"])
ax.tick_params(length=0)
ax.set_title("Semantic inputs and supervised queries", pad=10)
save(fig, "ablation")

fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.85), layout="constrained")
for m, ax in zip(METHODS, axes.flat):
    values = [[count(m, x=x, y=y) for x in range(4)] for y in range(3)]
    im = rate_grid(ax, values, 8)
    ax.set_title(NAMES[m], color=COLORS[m], fontweight="bold")
    ax.set_xlabel("Column x")
    ax.set_ylabel("Row y")
ax = axes.flat[-1]
ax.axis("off")
ax.text(.05, .85, "Each cell: 8 trials\n4 objects x 2 repeats\n\nIdentical scale across methods",
        transform=ax.transAxes, va="top", linespacing=1.7)
cb = fig.colorbar(im, cax=ax.inset_axes([.07, .14, .85, .09]), orientation="horizontal")
cb.set_ticks([0, 50, 100])
cb.set_label("Success rate (%)", labelpad=3)
save(fig, "position")

fig, axes = plt.subplots(5, 4, figsize=(7.16, 7.1), layout="constrained")
for i, m in enumerate(METHODS):
    for j, o in enumerate(OBJECTS):
        ax = axes[i, j]
        values = [[count(m, o, x, y) for x in range(4)] for y in range(3)]
        im = rate_grid(ax, values, 2, DENSE_SIZE)
        if i == 0:
            ax.set_title(OBJECT_NAMES[j], fontweight="bold")
        if j == 0:
            ax.set_ylabel(NAMES[m] + "\nRow y", color=COLORS[m], fontsize=LABEL_SIZE)
        if i == 4:
            ax.set_xlabel("Column x")
cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=.025, pad=.025, aspect=45)
cb.set_ticks([0, 50, 100], labels=["0/2", "1/2", "2/2"])
cb.set_label("Successful repeats per object-position condition")
save(fig, "object_position")

FAIL_KEYS = ["never_exposed", "exposed_never_separated",
             "separated_never_reached_region", "reached_region_final_failure"]
FAIL_NAMES = ["Never exposed", "Exposed, not separated",
              "Separated, not in region", "Reached region, final failure"]
FAIL_COLORS = ["#8393A0", "#DDA653", "#487C98", "#AA7189"]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.16, 3.05), layout="constrained")
for m, style in zip(METHODS, ["--", ":", "-.", (0, (4, 1)), "-"]):
    t = sorted(float(r["control_duration_s"]) for r in subset(m) if int(r["success"]))
    ax.step([0]+t+[40], [0]+list(np.arange(1, len(t)+1)/96*100)+[len(t)/96*100],
            where="post", color=COLORS[m], linestyle=style, lw=1.6, label=NAMES[m])
ax.set(xlim=(0, 40), ylim=(0, 103), xlabel="Elapsed control time (s)",
       ylabel="Completed / all trials (%)")
ax.set_title("(a) Cumulative task completion", loc="left")
ax.grid(axis="y", color="#E5E9EC", lw=.6)
ax.legend(ncol=2, frameon=False, loc="upper center",
          bbox_to_anchor=(.5, -.2), columnspacing=1)
left = np.zeros(5)
for key, name, color in zip(FAIL_KEYS, FAIL_NAMES, FAIL_COLORS):
    counts = np.array([FAIL[m][key] for m in METHODS])
    bx.barh(range(5), counts, left=left, color=color, height=.62, label=name)
    for i, n in enumerate(counts):
        if n >= 3:
            bx.text(left[i]+n/2, i, str(n), ha="center", va="center", color="white", fontsize=SMALL_SIZE)
    left += counts
assert all(left[i] == 96-count(m) for i, m in enumerate(METHODS))
for i, n in enumerate(left):
    bx.text(n+.8, i, str(int(n)), va="center", fontsize=SMALL_SIZE)
bx.set_yticks(range(5), [NAMES[m] for m in METHODS])
bx.invert_yaxis()
bx.set(xlim=(0, 51), xlabel="Failed trials (out of 96)")
bx.set_title("(b) Failure-stage profile", loc="left")
bx.legend(loc="upper center", bbox_to_anchor=(.5, -.2),
          frameon=False, ncol=1)
save(fig, "time_failure")


def table(name, lines):
    path = P / "results" / f"{name}_en.tex"
    path.write_text("% Generated from frozen results; do not edit by hand.\n" + "\n".join(lines) + "\n")
    OUTPUTS.append(path)


def entry(method, obj):
    a = A[method, obj]
    return f"{a['successes']}/{a['n']} ({a['success_rate_pct']:.1f}\\%)"


lines = []
for m in METHODS:
    cells = [NAMES[m]] + [entry(m, o) for o in OBJECTS + ["ALL"]]
    cells.append(f"{A[m,'ALL']['mean_success_control_s']:.2f}")
    if m == "viewQtoken":
        cells = [r"\textbf{" + c + "}" for c in cells]
    lines.append(" & ".join(cells) + r"\\")
table("main_table_rows", lines)
table("timing_rows", [
    f"{NAMES[m]} & [{A[m,'ALL']['wilson95_low_pct']:.1f}, {A[m,'ALL']['wilson95_high_pct']:.1f}]"
    f" & {A[m,'ALL']['mean_success_control_s']:.2f}"
    f" & {A[m,'ALL']['mean_all_control_s']:.2f} & {TIMING[m]['success_by_20s']}/96" + r"\\"
    for m in METHODS])
table("failure_rows", [
    " & ".join([NAMES[m]] + [str(FAIL[m][key]) for key in FAIL_KEYS] + [str(FAIL[m]["failures"])]) + r"\\"
    for m in METHODS])
lines = []
for left, right in [("ACT", "segNoQtoken"), ("ACT", "RGBQtoken"), ("ACT", "viewQtoken"),
                    ("DP", "viewQtoken"), ("RGBQtoken", "viewQtoken"), ("segNoQtoken", "viewQtoken")]:
    row = next(r for r in S["method_tests"] if r["object"] == "ALL"
               and r["left"] == left and r["right"] == right)
    p = row["p_holm"]
    if p < .001:
        exponent = int(np.floor(np.log10(p)))
        ptext = "$" + f"{p/10**exponent:.2f}" + r"\times10^{" + str(exponent) + "}$"
    else:
        ptext = f"{p:.4f}"
    lines.append(f"{NAMES[right]} vs. {NAMES[left]} & +{(count(right)-count(left))/96*100:.2f}"
                 f" & {ptext}" + r"\\")
table("contrast_rows", lines)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


sources = sorted((P/"results").glob("*.csv")) + [P/"results/summary.json"]
manifest = {
    "builder": Path(__file__).name, "trials": len(ROWS), "condition_cells": len(trial_cells),
    "method_labels": NAMES, "method_colors": COLORS,
    "typography": {"family": FONT, "title_pt": 10, "label_pt": LABEL_SIZE,
                   "tick_legend_pt": SMALL_SIZE, "dense_cell_pt": DENSE_SIZE},
    "source_sha256": {str(p.relative_to(P)): digest(p) for p in sources},
    "output_sha256": {str(p.relative_to(P)): digest(p) for p in OUTPUTS},
}
(P/"results/english_assets_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Verified {len(ROWS)} trials; generated {len(OUTPUTS)} English figure/table files.")

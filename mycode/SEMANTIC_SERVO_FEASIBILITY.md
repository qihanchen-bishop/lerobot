# Semantic Servo Draft: Feasibility and Implementation Boundary

This note maps `docs/draft/main.tex` to code. It deliberately separates a
trainable predictor from a calibrated controller. Running the code does not by
itself establish any CLF or safety guarantee.

## Feasibility assessment

| Draft component | Current status | What is still required |
| --- | --- | --- |
| One/multi-view soft semantic state | View-specific metrics implemented for a configured view list | Set aggregation, per-class confidence gates, and view dropout for one checkpoint to accept a changing view count |
| Offline SAM2 label quality | Implemented | Per-class threshold calibration on a manually reviewed subset |
| Action-conditioned semantic dynamics | Implemented in `SSACT-1` | Held-out residual calibration and intervention/recovery coverage |
| Five-phase history model | `SSACT-1` trains ordered offline soft labels and a semantic-history GRU | Review low-confidence/stratified trajectories and calibrate phase confidence |
| Structured residual servo | Not connected to the robot | Kinematic choice, residual action semantics, and recovery demonstrations |
| Robust CLF-QP | Box-constrained solver implemented but disabled in policy inference | Valid local dynamics model, independent error bounds, CLF domain/feasibility tests, robot limits |
| Event trigger | Reusable monitor implemented | Valid thresholds from an independent validation split |

The existing dataset contains RGB, four SAM2-derived semantic masks per view,
joint state, and expert action. It does not contain phase labels, camera
calibration, recovery trajectories, or certified model-error bounds. Therefore
semantic prediction is feasible now, while the theorem assumptions cannot yet
be asserted experimentally.

## Implemented semantic state

`SoftSemanticStateExtractor` maps the existing labels as follows:

- `occluder -> cloth`
- `object -> object`
- `region -> goal`
- `tool -> actuator`

For each view the dynamics state contains 14 normalized values: four area
ratios, normalized centroids of object/goal/actuator, object-goal and
object-cloth soft contact, and object-goal and actuator-object centroid
distance. A run with $N$ views therefore has $14N$ predicted values; the current
front/side configuration has 28. Segmentation
confidence is computed online as a diagnostic but is not assigned a fake
all-ones target from hard future labels. Values remain view-local;
the code does not incorrectly treat front and side image coordinates as one
metric coordinate system.

Object velocity is intentionally omitted from a single-frame extractor. It
must be computed from timestamped state history after camera synchronization.

The current semantic map is mutually exclusive, so hard object and goal pixels
cannot overlap. `SoftSemanticStateExtractor` therefore uses a small local
dilation before computing object-goal/object-cloth contact. This is a proxy,
not the physical overlap in the draft. A true occupancy ratio requires an
independent amodal goal/cloth mask or a calibrated table-plane goal polygon.

## SSACT-1 semantic dynamics contract

`SSACT-1` preserves the SEM-1 inputs and losses for the view list selected by
`--rgb-keys`:

- each selected view's RGB and soft semantic map enter the same ACT ResNet;
- segmentation uses weighted multiclass CE plus foreground Dice;
- ACT action gradients do not update the segmentation network;
- the dynamics head sees the predicted current semantic state, normalized robot
  state, and expert action chunk;
- hard SAM2 masks are converted offline into semantic-state targets at all
  frames; training indexes configured future offsets from this cache;
- future offsets crossing an episode boundary are excluded from Gaussian
  rollout NLL without decoding future mask videos.

Run `mycode/sam2_mask_quality.py`, then
`mycode/precompute_semantic_states.py`, before training. Pass the resulting
cache through `--semantic-states`; it embeds the classwise quality scores and
uncertainty flags. Current low-quality classes are masked from weighted CE and
Dice. A future prediction offset is removed from dynamics NLL when any required
view/class label at that offset is unreliable. Reports and the cache must cover
all dataset frames; smoke outputs generated with `--max-episodes` are rejected.

The report stores temporal IoU, area log change, normalized centroid jump, and
connected-component count. Individual anomaly flags remain visible for review,
while the aggregate uncertainty flag uses the calibrated score. Thresholds
must ultimately be calibrated per view and class: in this dataset the side-view
tool is more easily confused with the background cloth than the front-view
tool. The multi-view model should gate that feature specifically instead of
discarding every side-view feature. This also avoids treating every genuine
fast tool movement as a label error.

The default offsets `1 8 24 60` correspond to approximately 0.03, 0.27, 0.8,
and 2.0 seconds at 30 Hz. Offline caching removes future mask decoding from the
training data path. The action, segmentation, image resolution, backbone, and
60-step ACT chunk remain aligned with SEM-1 for the main comparison.

## Phase labels

Semantic labels can propose event boundaries but cannot uniquely determine all
five phases from one frame. In particular, both initial uncover and final
restore can have high cloth area. Generate the following soft candidates using
temporal metrics, confidence weights, consecutive-frame voting, and the known
forward phase order:

1. `t_vis`: object is reliably visible for consecutive frames.
2. `t_sep`: object-cloth overlap remains below a calibrated threshold.
3. `t_goal`: object-goal overlap remains above a calibrated threshold.
4. `t_restore`: cloth recovery, goal occupancy, and low action speed all remain valid.

Train `PhaseHistoryModel` on these soft labels and use history to disambiguate
visually similar states. Review only trajectories with low confidence, missing
events, reversed candidates, or strong model/rule disagreement. A small audited
subset is still needed to assign semantic meaning to the five latent phases and
to calibrate confidence; four manually corrected boundaries for every episode
are not required. Do not tune thresholds on the final test trajectories.

The first implemented version uses `semantic_phase_labels.py` to alternate
between global semantic phase centroids and ordered per-episode segmentation.
It includes a duration prior and rejects phases that collapse to the minimum
duration in most episodes. `SSACT-1` uses semantic history only; robot-state and
actual executed-action history remain a documented extension rather than being
approximated with planned ACT actions.

## Required validation before CLF deployment

1. Split episodes by trajectory, not frame, into train/calibration/test sets.
2. Calibrate segmentation confidence and semantic rollout intervals on the
   calibration split; report coverage against requested confidence.
3. Use genuinely view-specific image-space objectives when uncalibrated. Add a
   camera-to-table or 3D mapping only for cross-view physical fusion, metric
   distances, or physical safety constraints.
4. Verify local controllability and QP feasibility for every claimed phase
   domain, including actuator delay and absolute-target-to-increment conversion.
5. Collect failed/recovery demonstrations. Success-only behavior cloning cannot
   identify useful corrections away from the demonstration manifold.
6. Run the CLF projector in shadow mode first and log violations, slack, action
   corrections, and latency before allowing it to modify commands.

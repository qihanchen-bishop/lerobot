# SSACT-1 Training Design

`SSACT-1` is the first trainable version of the semantic-servo draft. It is a
controlled extension of `PSEM-1`, not a deployed visual-servo controller.

## Offline phase supervision

`semantic_phase_labels.py` builds 12 geometric semantic features per configured
view from the complete mask-quality reports. It applies static view/class gates
before fitting a five-state ordered segment model over all episodes. The model
alternates between global phase centroids and per-episode dynamic programming.
A normalized-time prior and minimum duration prevent phase collapse.

The output contains soft labels around each of four learned boundaries. Labels
remain pseudo-labels: low-confidence episodes and a stratified sample of the
rest still require review before the phase names are treated as ground truth.

For the current front/side dataset, `side_tool=0.25` encodes the known ambiguity
between the side-view tool and background cloth. It gates only side-tool phase
features, its semantic color supplied to ACT, and side-tool components of the
semantic dynamics state. It does not suppress side-view object or region
features, and it does not weaken supervised segmentation loss.

## Trainable modules

- the same RGB-to-soft-semantic U-Net as `PSEM-1`;
- the same ACT inputs: one RGB and one soft semantic image per view;
- the same expert-action-conditioned semantic dynamics at offsets 1, 8, 24, 60;
- a GRU over 16 semantic samples spaced four frames apart (about two seconds at
  30 Hz), trained with confidence-weighted soft phase cross entropy;
- a five-value phase probability vector supplied to ACT as environment state.

ACT sees pseudo-label phase probabilities for the first 10k updates. Training
then transitions linearly to the phase model prediction over 20k updates. Phase
and action losses do not backpropagate into segmentation in this first version.

## Deliberate boundary

The CLF/QP action correction is not enabled. It first needs held-out semantic
dynamics residual calibration, QP feasibility checks, and robot-side shadow
evaluation. The phase GRU currently uses semantic history only; robot-state and
executed-action histories can be added after the live inference API records
actual executed actions rather than treating predicted actions as execution.

The current implementation accepts any configured one-or-more-view list, but a
checkpoint has a fixed view list. Runtime-changing view counts require a set
aggregator and are not claimed by `SSACT-1`.

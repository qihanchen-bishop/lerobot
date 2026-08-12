# SSACT-1 Training Design

`SSACT-1` is the first trainable version of the semantic-servo draft. It combines
SEM-1 visual inputs with semantic dynamics and phase history; it is not yet a
deployed visual-servo controller.

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

- the RGB-to-soft-semantic U-Net and ACT inputs from SEM-1;
- one RGB and one soft semantic image per view;
- expert-action-conditioned semantic dynamics at offsets 1, 8, 24, 60, trained
  against offline cached semantic targets rather than future video decoding;
  its `delta_head` predicts the future semantic-state increment, and the final
  prediction is `semantic_state + semantic_delta`; it does not predict an
  action residual;
- a GRU over 16 semantic samples spaced four frames apart (about two seconds at
  30 Hz), trained with confidence-weighted soft phase cross entropy;
- a five-value phase probability vector supplied to ACT as environment state.

ACT sees pseudo-label phase probabilities for the first 10k updates. Training
then transitions linearly to the phase model prediction over 20k updates. Phase
and action losses do not backpropagate into segmentation in this first version.

## Runtime controller and deliberate boundary

The evaluation GUI can run an experimental semantic controller in `shadow` or
`active` mode. It differentiates a phase-specific semantic CLF through the
learned semantic dynamics and uses a bounded CLF-QP to produce a separate
normalized action correction. It also selects 1--4 execution steps from phase,
phase confidence, dynamics uncertainty, semantic innovation, and nominal CLF
progress. Every replan writes these values to `ssact_runtime.jsonl` and shows
them in the GUI.

This runtime execution-length rule is not a learned hazard model. The current
checkpoint contains no hazard-head parameters. The CLF/QP is also not certified:
it still needs held-out semantic-dynamics residual calibration, QP feasibility
checks, and robot-side shadow evaluation before a stability guarantee can be
claimed. The phase GRU currently uses semantic history only; robot-state and
executed-action histories can be added after the live inference API records
actual executed actions rather than treating predicted actions as execution.

The current implementation accepts any configured one-or-more-view list, but a
checkpoint has a fixed view list. Runtime-changing view counts require a set
aggregator and are not claimed by `SSACT-1`.

# SSACT-3-StageFiLM Experiment Design

`SSACT-3` is the first event-supervised stage-conditioning experiment built on
`SEM-1-V2`. The unique main run name is
`SSACT-3-StageFiLM-bettersetup-front-side`; it does not reuse `SSACT-1` or
`SSACT-2`, whose objectives and implementations are different.

## Question under test

The experiment tests whether event-derived stage belief and phase-specific
semantic relations improve ACT without allowing action gradients to corrupt the
segmentation network. It is not yet the learned execution-length, residual
action, CLF-QP, or event-triggered chunk-truncation experiment. Those controller
features remain separate follow-up modules after stage recognition is shown to
work.

## Offline supervision

`precompute_semantic_states.py` decodes the fixed mask videos once. For every
view and frame it stores class area, soft-centroid equivalents, local contact,
normalized distance, and the existing SAM2 quality diagnostics.

`generate_stage_supervision.py` then derives four continuous event scores:

- object visible;
- object separated from the occluder;
- object inside the target region;
- occluder restored while the object remains inside the region.

A hysteretic state machine converts only stable, sufficiently reliable events
into `expose -> separate -> transport -> restore -> done`. It can roll back one
stage when visibility, separation, or placement is lost. Labels around observed
boundaries are softened. They are event-derived pseudo-labels, not fixed-time
segments and not manually verified ground truth.

Front is the primary view. A secondary view may replace the front visibility or
separation score only when the corresponding front signal has low quality. Raw
image coordinates are never averaged across cameras, because the views are not
calibrated and differ substantially. SAM2 quality is used continuously as a loss
weight. No low-quality sample is deleted by a hard threshold.

Before training, generation must report non-degenerate frame counts for all five
stages and an acceptable fraction of incomplete episodes. Incomplete trajectories
must be inspected or threshold assumptions revised; `--allow-incomplete` is a
diagnostic option, not the main training setting.

## Trainable modules

The model has independent, ablatable heads over a shared semantic-history GRU:

- `Phase Head`: five-way current stage belief;
- `Event Head`: positive and failure signals such as object visible/lost and
  separation achieved/lost;
- `Progress Head`: continuous progress in the current phase;
- `Transition Head`: `stay / advance / rollback / uncertain`;
- `Relation Head`: the phase-relevant geometry used to audit what the temporal
  representation learned.

ACT receives the detached five-value phase distribution as environment state.
The semantic adapter receives detached phase belief and temporal context, builds
five probability channels plus object-occluder, object-region, and tool-object
local relation maps, and injects one residual per corresponding RGB view at
ResNet layer 4. Explicit phase priors initialize which channels matter. A small
learned attention delta and FiLM modulation adapt this prior. This does not add
visual tokens and does not concatenate the two camera images along width.

## Protection against unstable semantics

The first 20k updates train RGB ACT, quality-weighted segmentation, and all stage
heads while semantic residual fusion is zero. Fusion ramps to full strength over
the next 10k updates. The stage model initially receives the cached current
semantic state; its current input changes to the predicted soft segmentation over
the same 20k + 10k schedule. ACT is conditioned on target phase probabilities
for 30k updates, then changes linearly to detached predicted phase probabilities
over 20k updates.

The segmentation probability, stage context, and predicted phase paths are all
detached before the action loss. Therefore:

- segmentation is optimized only by quality-weighted multiclass CE plus
  foreground Dice;
- stage heads are optimized only by their named auxiliary losses;
- action loss optimizes ACT and the semantic adapter, not the U-Net or stage
  classifier;
- zero-initialized residual and FiLM projections make step zero behavior match
  RGB ACT rather than injecting random semantic features.

## Loss and default weights

The total training objective is

```text
L = L_action + L_seg
  + 0.20 L_phase
  + 0.10 L_event
  + 0.10 L_progress
  + 0.10 L_transition
  + 0.10 L_relation
  + 0.01 L_attention_regularization.
```

Each stage loss is normalized by the sum of its continuous quality weights.
Phase and transition classes receive capped inverse-frequency balancing. The
weights are conservative auxiliary weights; they keep action and segmentation as
the two primary objectives. Their ablation flags are independent, so setting one
weight to zero removes that supervision without changing data or architecture.

## Required evidence before controller claims

This experiment can establish that stage information is predictable and useful
for action learning only if all of the following hold:

1. reviewed stage events have adequate precision and recall, especially
   `separate -> transport`;
2. phase confusion and transition confusion matrices are reported on held-out
   episodes rather than only training accuracy;
3. `SSACT-3` improves robot success or removes the expose-stage deadlock relative
   to same-data, same-seed RGB ACT and `SEM-1-V2` baselines;
4. ablations distinguish gains from phase state, attention, FiLM, event losses,
   and semantic quality weighting;
5. deployment includes persistence/hysteresis and immediate replanning at a
   predicted transition.

`SSACT-3` alone does not guarantee convergence or recovery. Learned chunk length,
actual action truncation, residual action generation, semantic dynamics, and
CLF-QP/MPC correction require subsequent experiments with independent runtime and
safety validation.

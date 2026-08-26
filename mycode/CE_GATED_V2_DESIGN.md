# CE-Gated-v2 Experimental Design

## Question

Determine whether a gated camera identity embedding is actively optimized and used by ACT, rather than
inferring its effect from final success rate or the scalar gate alone.

## Controlled Change

CE-Gated-v2 keeps the completed two-view ACT baseline configuration unchanged:

- dataset: `bettersetup`, using the exact existing filtered ACT dataset view;
- image order: front, side;
- seed: 1000;
- ImageNet-pretrained ResNet18, shared between views;
- chunk size and executed action steps: 60;
- batch size: 8; training updates: 100,000.

The only policy change is a camera identity embedding with front=0 and side=1. Its raw vectors are
initialized from `N(0, 0.02)` and multiplied by one learned scalar gate initialized to `0.01`. Unlike the
old zero-gate run, both the gate and embedding vectors can receive gradients from the first update. The
initial effective embedding RMS is expected near `2e-4`: small relative to RGB features, but not blocked.

## Training Evidence

Every 200 updates, the training JSONL records:

- gate value and gate gradient magnitude;
- raw/effective embedding RMS and embedding-weight gradient RMS;
- effective front-side embedding difference RMS;
- projected RGB feature RMS for each view;
- effective embedding/RGB feature ratio for each view;
- deterministic normalized-action RMS/max difference when forcing gate to zero;
- deterministic normalized-action RMS/max difference when swapping front/side camera IDs.

Checkpoints are retained every 20,000 updates. This permits the same forward and real-robot ablations at
20k, 40k, 60k, 80k, and 100k instead of reconstructing training behavior from the final checkpoint.

## Interpretation

1. If scale, gradients, and action deltas remain negligible at every checkpoint, the direct gated camera
   path did not become functionally active.
2. If intermediate checkpoints have measurable action deltas but the final checkpoint does not, camera
   identity may have shaped shared weights transiently and then decayed.
3. If the final checkpoint has measurable gate-disable and ID-swap action deltas, camera identity is used
   during inference. Real-robot paired ablations must then determine whether that use is beneficial.
4. If camera identity is active but task success does not improve, the mechanism works technically but does
   not address the current policy bottleneck.

Training curves establish when and how the path becomes active. They do not by themselves prove that the
gated mechanism caused a success-rate improvement: a strict causal claim still requires a paired sham run
with the same additional parameters and training environment but a permanently disabled camera path.

## Real-Robot Evaluation

For the final checkpoint and any intermediate checkpoint with a substantial action delta, use the same grid,
trial order, and reset procedure for three paired modes:

1. learned gate and correct camera IDs;
2. gate forced to zero;
3. learned gate with front/side IDs swapped.

Report success, failure category, duration, normalized action delta, and denormalized joint-angle delta. The
camera embedding is considered beneficial only if correct IDs outperform gate-off and swapped-ID controls,
not merely the historical ACT checkpoint.

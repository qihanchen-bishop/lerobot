# Anchor-and-Motion ACT v1

## Objective

AM-ACT-v1 factorizes an ACT action chunk into one absolute joint-space entry anchor and a motion
trajectory expressed relative to that anchor. It is a representation experiment; semantic inputs,
runtime phase alignment, and residual correction are intentionally excluded.

For a normalized expert action chunk `A` with shape `[B, S, D]`, the targets are:

```text
anchor_target = A[:, 0:1]
motion_target = A[:, 1:] - A[:, 0:1]
```

The policy decodes its predictions back to the standard absolute ACT interface:

```text
absolute[:, 0:1] = anchor
absolute[:, 1:] = anchor + motion_offsets
```

All transformations happen after the existing per-joint action normalization. Since that
normalization is affine, decoding exactly recovers normalized absolute coordinates and the standard
ACT postprocessor can unnormalize the chunk without new dataset statistics.

## Architecture

- The ACT visual backbone, transformer, decoder queries, VAE, and external action API are unchanged.
- Decoder token 0 is passed through an independent `anchor_head`.
- Decoder tokens 1 to `S-1` are passed through the existing action head, which acts as the motion head.
- A projected predicted-anchor feature is added to all motion tokens through a learnable scalar gate.
- The gate starts at `0.1` by default and its effective feature ratio is logged.
- The anchor head starts as an exact copy of the motion head without consuming additional RNG state.
- The anchor-to-motion projection starts from the transpose of the motion head with zero bias.
- The VAE encoder continues to consume absolute expert actions in v1, isolating the decoder action
  representation from changes to the ACT latent objective.

## Supervision

The action objective is:

```text
L_action = 0.25 * L_anchor + 0.25 * L_motion + 0.50 * L_reconstruction
L_total  = L_action + 10.0 * L_KL
```

- `L_anchor`: L1 loss on the absolute first action.
- `L_motion`: L1 loss on offsets relative to the first expert action.
- `L_reconstruction`: L1 loss on the decoded absolute chunk.
- All temporal losses respect `action_is_pad`.

Training also logs endpoint and velocity L1 metrics. They are diagnostics and do not affect v1
optimization.

## First Experiment

`AM-ACT-v1-FS-bettersetup-v5` uses front and side RGB, the 10-dimensional bimanual joint state,
ImageNet-pretrained ResNet18, a 60-step chunk, seed 1000, and no semantic/camera embeddings. Its
primary comparison is the existing two-view RGB ACT baseline.

## Explicit Non-goals

- No nearest-waypoint or latency-based runtime alignment.
- No learned residual correction.
- No semantic or stage input.
- No VAE anchor-offset encoding.
- No velocity, acceleration, or smoothness training loss.

Runtime alignment should later be evaluated independently on both ordinary ACT and AM-ACT using a
2x2 policy-representation versus execution-alignment experiment.

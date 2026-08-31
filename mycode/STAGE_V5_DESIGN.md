# UNET-SEM-V5 and STAGE-V5

## Goal

These experiments separate three questions that were previously entangled:

1. Does a stable pretrained semantic representation improve ACT?
2. Does an explicit task-stage belief improve ACT even when semantic maps are not visual inputs?
3. Do front and side RGB provide complementary information once stage and semantic inputs are controlled?

All runs use the same `bettersetup_v5` demonstrations, action chunk, seed, ACT backbone, optimizer,
and training length. The frozen segmentation networks never receive action gradients and contribute no
segmentation loss.

## View-Specific Semantic Inputs

`UNET-SEM-V5-FS` uses one frozen checkpoint per ordered RGB view. `UNET-SEM-V5` remains an alias for
backward compatibility:

| View | Checkpoint | Probability classes |
| --- | --- | --- |
| front | `models/unet_front_v4_r1/best.pt` | background, occluder, object, region, tool, leftarm, rightarm |
| side | `models/unet_side/best.pt` | background, occluder, object, region, tool |

Each checkpoint is checked against the dataset mask suffixes for its view. Its soft probabilities are
converted to semantic RGB with the canonical class palette. Front and side therefore keep different
class counts without inventing side-arm labels. Semantic RGB and camera RGB are separate ACT image
features, but all image features use the same ACT ResNet18 parameters. No camera or modality embedding
is enabled, keeping the comparison aligned with the existing basic ACT runs.

## Stage Definition

The stage target is a five-way probability, not a scalar regression target. The first four values are
action stages and the fifth is an absorbing completion state:

| Index | Name | Advance condition |
| ---: | --- | --- |
| 0 | expose | front object is reliably visible |
| 1 | separate | object visibility is sufficient and object-occluder contact is low |
| 2 | transport | object is reliably inside the target-region proxy |
| 3 | restore | front occluder area is stably at least 50% |
| 4 | done | completion state; no further advance |

The labels come only from front dataset masks. Continuous SAM2 quality scores weight the event, phase,
transition, progress, and relation losses. Advance requires eight consecutive frames. Rollback from
separate, transport, or restore requires twelve consecutive frames below a lower threshold, providing
hysteresis. Soft labels are used around boundaries.

Because the current mutually exclusive masks remove region pixels under the object, "fully inside" is
represented by the existing object-region local-contact and centroid-distance proxy. It is not a formal
geometric containment guarantee. A future version should reconstruct the region boundary or add an
explicit containment label before making that stronger claim.

## Policy Conditioning

A temporal stage model reads 16 front semantic states at a stride of four frames plus current robot
state. It predicts Phase, Event, Progress, Transition, and Relation heads. Only the five phase
probabilities are passed to ACT as an environment-state token. During training, phase conditioning uses
the quality-weighted target for 10k steps, then transitions to the detached Phase Head prediction over
20k steps. Detaching prevents action loss from changing the stage classifier to obtain an easier action
loss. At inference the stage history is computed from the frozen front U-Net; GT masks are unavailable
and are never read.

The auxiliary stage heads improve identifiability and provide diagnostics, but this experiment does not
yet truncate queued action chunks or enforce a runtime finite-state machine. Those mechanisms should be
evaluated only after the predicted stage and transition signals are accurate.

## Ablation Matrix

| Experiment | ACT RGB | ACT semantic RGB | Stage source |
| --- | --- | --- | --- |
| `UNET-SEM-V5-FS` | front + side | front + side | none |
| `STAGE-V5-F-RGB` | front | none | frozen front U-Net |
| `STAGE-V5-FS-RGB` | front + side | none | frozen front U-Net |
| `STAGE-V5-F-UNETSEM` | front | front | frozen front U-Net |
| `STAGE-V5-FS-UNETSEM` | front + side | front + side | frozen front U-Net |

The primary comparisons are:

- existing front ACT vs `STAGE-V5-F-RGB`: stage contribution with identical visual input;
- existing two-view ACT vs `STAGE-V5-FS-RGB`: stage contribution with identical visual input;
- `STAGE-V5-F-RGB` vs `STAGE-V5-F-UNETSEM`: front semantic visual contribution;
- `STAGE-V5-FS-RGB` vs `STAGE-V5-FS-UNETSEM`: two-view semantic visual contribution;
- existing two-view ACT vs `UNET-SEM-V5-FS`: pretrained semantic input without stage conditioning.

Report action loss separately from auxiliary stage loss. Real-robot evaluation must reuse the same grid,
object order, trial count, replan settings, and success criteria. Also report phase confusion, transition
delay, rollback count, expose-to-transport entry rate, action duration, and failure type.

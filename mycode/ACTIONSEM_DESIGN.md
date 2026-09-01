# ActionSEM-F / ActionSEM-FS

## Question

ActionSEM tests whether action supervision can make an already accurate semantic segmenter more useful
for control without destroying its semantic meaning. It is compared against the corresponding frozen
U-Net semantic-input policy, not only against basic ACT.

## Inputs

| Experiment | RGB inputs | Semantic inputs | Pretrained segmenters |
| --- | --- | --- | --- |
| `ACTIONSEM-F` | front | front soft semantic RGB | front 7-class TinyUNet |
| `ACTIONSEM-FS` | front + side | front + side soft semantic RGB | front 7-class + side 5-class TinyUNet |

ACT image order is semantic views followed by RGB views. Semantic probabilities are converted to the
existing class-color expectation, then semantic RGB and camera RGB share the same ACT ResNet18. Camera
and modality embeddings are disabled in this first experiment.

The two-view model does not force identical segmentation heads. Front predicts background plus
occluder, object, region, tool, leftarm, and rightarm. Side predicts background plus occluder, object,
region, and tool. The two segmentation losses are computed separately and averaged.

## Trainable Segmentation Layers

Only the convolution weights in `fuse0` and the final `head` are trainable. The encoder, `fuse2`,
`fuse1`, and every BatchNorm parameter and running statistic stay frozen. Fine-tuned U-Net parameters
are part of the policy checkpoint and are restored for inference.

## Loss And Gradient Guard

Each view uses continuous-quality-weighted multiclass cross entropy plus foreground Dice:

```text
L_seg_view = L_weighted_CE + L_foreground_Dice
L_seg = mean_view(L_seg_view)
```

Training uses three periods:

1. Steps 0-20k: U-Nets are fixed and semantic maps are detached from action loss.
2. Steps 20k-40k: `fuse0 + head` receive segmentation supervision and action gradient ramps up.
3. Steps 40k-100k: the action-gradient norm is capped at 5% of the supervised segmentation-gradient
   norm for each view independently.

When action and segmentation gradients conflict, the component of the action gradient opposing the
segmentation gradient is removed before applying the norm cap. This protects semantic accuracy while
allowing task-relevant changes in non-conflicting directions.

## Required Comparisons

- Basic ACT F/FS: no semantic input.
- Frozen U-Net semantic F/FS: semantic input without segmentation fine-tuning.
- ActionSEM F/FS: semantic input with supervised late-layer fine-tuning and guarded action gradients.

To isolate action supervision from ordinary continued segmentation training, a later strict ablation
should use the same late-layer segmentation fine-tuning with action gradients disabled. Report policy
success, air-push frequency, completion failures, per-view/class Dice, action loss, gradient cosine,
conflict rate, raw/applied gradient ratio, and segmentation output drift.

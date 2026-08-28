# Front-view U-Net v4 r1

This is the `seg_v3` deployment copy of the front-camera semantic-segmentation
model trained from `bettersetup_v4`. It predicts one mutually exclusive class
ID per pixel.

## Classes

| ID | Label | Preview color (RGB) |
|---:|---|---|
| 0 | background | 0, 0, 0 |
| 1 | occluder | 64, 160, 255 |
| 2 | object | 255, 105, 97 |
| 3 | region | 119, 221, 119 |
| 4 | tool | 255, 209, 102 |
| 5 | leftarm | 234, 146, 199 |
| 6 | rightarm | 173, 214, 101 |

The input is RGB at 480 x 270, normalized with ImageNet mean and standard
deviation. The output is a 7-channel logit tensor; `argmax(channel)` gives the
class-ID mask.

## Validation result

The selected checkpoint is epoch 40. The deterministic validation split is
episode-disjoint and contains 2,819 sampled frames from 14 episodes.

| Metric | Value |
|---|---:|
| Mean IoU | 0.96488 |
| Foreground mean IoU | 0.96463 |
| Pixel accuracy | 0.98653 |

| Class | IoU | Precision | Recall |
|---|---:|---:|---:|
| background | 0.96637 | 0.98470 | 0.98111 |
| occluder | 0.98981 | 0.99524 | 0.99452 |
| object | 0.93741 | 0.95950 | 0.97604 |
| region | 0.99307 | 0.99576 | 0.99728 |
| tool | 0.96239 | 0.96948 | 0.99246 |
| leftarm | 0.96849 | 0.98053 | 0.98748 |
| rightarm | 0.93658 | 0.96368 | 0.97085 |

These metrics measure agreement with the reviewed SAM2 v4 masks, not with a
separate fully hand-labelled test set. See `evaluation.json` for the full
confusion matrix and `previews/` for cross-episode RGB/GT/PRED comparisons.

Targeted QA was also run on episodes 006, 009, 060, 061, 063 and 064. The
remaining difficult case is the tiny object/screw boundary during contact or
near-total occlusion; aggregate object IoU on those selected episodes is about
0.80 to 0.88, while object recall is about 0.916 to 0.947. See
`targeted_qa/targeted_quality.json` and the rendered worst-frame previews.

## Files

- `best.pt`: PyTorch training checkpoint, including labels and training args.
- `unet_front_v4_torchscript.pt`: portable inference model.
- `model_config.json`: preprocessing, class IDs and palette.
- `evaluation.json`: independent validation metrics and export-equivalence test.
- `history.json`: all 44 training epochs through early stopping.
- `benchmark.json`: batch-1 model-only latency on the training workstation.

## Inference

From this directory:

```bash
/home/qihan/miniconda3/envs/sam2/bin/python infer_light_semseg.py \
  --model-dir /home/qihan/data/lerobot/mycode/tool/seg_v3 \
  --input /path/to/front_image_or_video.mp4 \
  --output-dir /path/to/output \
  --device cuda:0
```

For an image, this writes a lossless class-ID PNG, a color mask and an overlay.
For a video, it writes a lossless FFV1 class-ID MKV and an MP4 overlay. The
class-ID output is the machine-readable semantic result; the colored output is
only for review.

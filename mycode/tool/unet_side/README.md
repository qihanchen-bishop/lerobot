# unet_side

Small side-view semantic segmentation model trained on `bettersetup_v5` revision `side_screw_ep065_r6`.

- Architecture: TinyUNet with depthwise-separable convolutions, width 32
- Parameters: 104,037
- Input: RGB uint8, resized to 480x270, ImageNet normalization
- Output classes: background, occluder, object, region, tool
- Validation split: deterministic episode-disjoint, 2819 frames
- Validation mIoU: 0.9693
- Foreground mIoU: 0.9636
- Pixel accuracy: 0.9944
- TorchScript parity: 1.0000
- Batch-1 latency on NVIDIA RTX PRO 6000 Blackwell Workstation Edition: 0.876 ms

Per-class validation metrics:

- background: IoU 0.9919, precision 0.9979, recall 0.9939
- occluder: IoU 0.9836, precision 0.9877, recall 0.9959
- object: IoU 0.9289, precision 0.9524, recall 0.9742
- region: IoU 0.9860, precision 0.9901, recall 0.9958
- tool: IoU 0.9561, precision 0.9687, recall 0.9866

Use `unet_side_torchscript.pt` for deployment or `best.pt` with the TinyUNet Python definition.
Class ids, colors, resizing, and normalization are recorded in `model_config.json`.

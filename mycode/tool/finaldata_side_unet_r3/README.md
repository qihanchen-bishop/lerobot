# U-Net side

Model: 第3代迭代模型

Run from this directory (no original training directory required):

```bash
python infer.py --model-dir . --input /path/to/video.mp4 --output-dir /path/to/prediction --device cpu
```

`model_config.json` includes labels, colors, input size and normalization. `infer.py` generates raw predictions; the reviewed dataset applies `filter_policy.json` afterward.
The checkpoint is included only if its hash was recorded when prediction was submitted.

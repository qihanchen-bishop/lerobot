# finaldata Training Commands

## 数据与配置来源

- 原始数据：`/home/qihan/data/lerobot/data/finaldata`。
- 分割项目副本：`/home/qihan/data/sam2/segdata/origin/finaldata`，为独立文件副本，不是软链接或硬链接。
- 数据元信息：108 episodes、61,740 frames、30 Hz；front/side 均为 640 x 360；state/action 均为10维。
- 对照来源：`usefulcommand.md` 中的 `ACT-bettersetup` 双视角基线与 `DP-1F-Full-bettersetup` 命令。
- 仅更换数据路径、repo ID、实验名及输出路径；DP 显式写出 `--seed 1000`，与旧命令的脚本默认值相同。
- 两个实验都以 `observation.state` 为本体输入，以数据集原有 `action` 为监督；不启用 follower delta、anchor、语义、QToken 或 camera embedding。
- 本文件只提供命令，不自动启动训练，也不上传数据或模型。先运行 ACT/DP 不需要等待分割完成。

## ACT-finaldata-front-side

双视角原始 RGB、完整视野、ImageNet 预训练 ResNet18；chunk size 60，n_action_steps 60。

```bash
cd /home/qihan/data/lerobot

/home/qihan/miniconda3/envs/lerobot/bin/python mycode/train_lerobot_policy.py \
    --policy-type act \
    --root /home/qihan/data/lerobot/data/finaldata \
    --repo-id finaldata \
    --image-keys observation.images.front observation.images.side \
    --state-keys observation.state \
    --output-dir outputs/train/act/ACT-finaldata-front-side \
    --job-name ACT-finaldata-front-side \
    --seed 1000 \
    --chunk-size 60 \
    --n-action-steps 60 \
    --pretrained-backbone-weights ResNet18_Weights.IMAGENET1K_V1 \
    --batch-size 8 \
    --steps 100000 \
    --device cuda \
    --num-workers 16 \
    --video-backend pyav \
    --rebuild-view
```

## DP-1F-Full-finaldata-front-side

双视角原始 RGB、单帧完整视野、ImageNet 预训练 ResNet18 + FrozenBatchNorm。
保持旧配置：horizon 64、n_action_steps 30、drop_n_last_frames 29，屏蔽补齐动作损失；训练/推理扩散步数 100/50。
调度器仍采用原脚本的 DDPM 默认值，不在此改成 DDIM 16步。

```bash
cd /home/qihan/data/lerobot

/home/qihan/miniconda3/envs/lerobot/bin/python mycode/train_lerobot_policy.py \
    --policy-type diffusion \
    --root /home/qihan/data/lerobot/data/finaldata \
    --repo-id finaldata \
    --image-keys observation.images.front observation.images.side \
    --state-keys observation.state \
    --output-dir outputs/train/diffusion/DP-1F-Full-finaldata-front-side \
    --job-name DP-1F-Full-finaldata-front-side \
    --seed 1000 \
    --diffusion-n-obs-steps 1 \
    --diffusion-horizon 64 \
    --diffusion-n-action-steps 30 \
    --diffusion-drop-n-last-frames 29 \
    --diffusion-mask-padding-loss \
    --diffusion-full-frame \
    --diffusion-backbone-norm frozen_batch \
    --pretrained-backbone-weights ResNet18_Weights.IMAGENET1K_V1 \
    --diffusion-lr-backbone 1e-5 \
    --diffusion-num-train-timesteps 100 \
    --diffusion-num-inference-steps 50 \
    --batch-size 8 \
    --steps 100000 \
    --device cuda \
    --num-workers 16 \
    --video-backend pyav \
    --rebuild-view
```

## 运行说明

- 两组均训练100,000步、batch size 8、seed 1000、16 workers，使用 PyAV。
- 保持脚本默认日志与保存设置：每200步记录、每20,000步保存，不启用在线环境评估。
- `--rebuild-view` 重建该实验的特征筛选视图，不修改原始数据。两个实验使用各自的输出与视图目录。
- 未添加 `--overwrite-output`；若输出目录已经存在，应确认续训或改名，不要直接覆盖。
- 副本一致性已验证；本次没有执行完整视频解码、图像统计修复或正式训练。

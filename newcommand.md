# finaldata Training Commands

## 数据与配置来源

- 原始数据：`/home/qihan/data/lerobot/data/finaldata`。
- 分割项目副本：`/home/qihan/data/sam2/segdata/origin/finaldata`，为独立文件副本，不是软链接或硬链接。
- 数据元信息：108 episodes、61,740 frames、30 Hz；front/side 均为 640 x 360；state/action 均为10维。
- 对照来源：`usefulcommand.md` 中的 `ACT-bettersetup` 双视角基线与 `DP-1F-Full-bettersetup` 命令。
- 仅更换数据路径、repo ID、实验名及输出路径；DP 显式写出 `--seed 1000`，与旧命令的脚本默认值相同。
- 上述 ACT/DP 基线都以 `observation.state` 为本体输入，以数据集原有 `action` 为监督；不启用 follower delta、anchor、语义、QToken 或 camera embedding。
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

## final-ViewQtoken

数据：`finaldata_seg`；双视角 RGB + 已导出的 filtered 硬语义彩色图，共享 ACT 视觉编码器。
训练不运行 U-Net、不微调分割；实机从 RGB 调用对应冻结 U-Net，argmax 后执行相同面积/连通分量过滤。
不增加相机/模态 embedding；动作监督仍为原数据集 `action`，不是 follower delta。

三个任务共享 Query，每个连接独立 front/side 输出头：

- visibility：两个视角分别预测 object 像素面积占比。
- tool-object：front 为工具 PCA 主轴左端点到 object 质心的向量；side 为 tool 质心到 object 质心的向量。
- object-region：可见 object 像素到固定区域内部的最大距离，按图像对角线归一化；完全入区为零。

六个头共8个标量。辅助监督使用 filtered `quality_score` 直接加权；相关类别任一 `uncertain=true` 即跳过该视角该指标。
目标不可见时面积仍监督零，距离/向量不监督。质量只控制辅助损失，不屏蔽动作样本。
损失先平均坐标、视角、任务；分母使用几何有效项数量，不用质量权重和。

### 一次性准备监督

```bash
cd /home/qihan/data/lerobot

/home/qihan/miniconda3/envs/lerobot/bin/python mycode/prepare_view_qtoken.py \
    --root /home/qihan/data/lerobot/data/finaldata_seg \
    --output-dir /home/qihan/data/lerobot/outputs/cache/final-ViewQtoken-v1 \
    --workers 8
```

重复调用会检查并复用完整缓存，不覆盖旧版本；源标签/质量变化需换缓存目录重建。
训练前检查缓存目录中的 `region_front.png`、`region_side.png` 红色区域轮廓。
轮廓仅从当前训练集高质量 region 标签估计：大面积候选的多数投票后取凸包。
该固定轮廓依赖相机与目标区域固定，不表示三维包含关系。

### 训练

```bash
cd /home/qihan/data/lerobot

/home/qihan/miniconda3/envs/lerobot/bin/python mycode/train_mask_act_policy.py \
    --experiment final-ViewQtoken \
    --root /home/qihan/data/lerobot/data/finaldata_seg \
    --repo-id finaldata_seg \
    --rgb-keys observation.images.front observation.images.side \
    --state-keys observation.state \
    --mask-target-keys \
      observation.images.front_occluder \
      observation.images.front_object \
      observation.images.front_region \
      observation.images.front_tool \
      observation.images.front_leftarm \
      observation.images.front_rightarm \
      observation.images.side_occluder \
      observation.images.side_object \
      observation.images.side_region \
      observation.images.side_tool \
      observation.images.side_leftarm \
      observation.images.side_rightarm \
    --pretrained-segmentation-checkpoints \
      /home/qihan/data/lerobot/data/finaldata_seg/models/unet_front/model.ts \
      /home/qihan/data/lerobot/data/finaldata_seg/models/unet_side/model.ts \
    --view-query-cache /home/qihan/data/lerobot/outputs/cache/final-ViewQtoken-v1/targets.npz \
    --log-freq 200 \
    --save-freq 20000 \
    --metric-loss-weight 1.0 \
    --action-loss-weight 1.0 \
    --output-dir outputs/train/semantic/final-ViewQtoken \
    --steps 100000 \
    --seed 1000 \
    --batch-size 8 \
    --chunk-size 60 \
    --n-action-steps 60 \
    --pretrained-backbone-weights ResNet18_Weights.IMAGENET1K_V1 \
    --device cuda \
    --num-workers 16 \
    --video-backend pyav \
    --rebuild-view
```

质量已包含在校验过的监督缓存中，无需另传旧 `--mask-quality-*` 参数。
每200步记录、每20000步保存，沿用旧 QToken 训练预算。
模型导出包会复制到训练输出的 `segmentation_models/front` 与 `side`，实机使用现有
`mask_act` 加载入口，并保留整份运行目录及其 `mask_act_run_config.json`。
现有推理加载器仍需要可用的数据集 metadata 来恢复 state/action 归一化统计。

### 本次验证

- 已完成108条轨迹、61,740帧的双视角监督缓存；直接运行上面的训练命令即可。
- 全量标签扫描通过，双视角123,480个帧检查单元无类别重叠；质量数组和 LeRobot 加载通过。
- 40项相关测试通过；真实数据两步训练、checkpoint 重载及 RGB-only 硬分割推理通过。
- 检查报告：`outputs/cache/final-ViewQtoken-v1/dataset_check.json`。
- 两步验证目录：`outputs/temp/final-ViewQtoken-smoke-20261018`，仅验证代码链路，不代表收敛结果。
- 未启动正式训练。固定 region 轮廓只适用于当前相机和区域布置，变更布置后需重建监督缓存。

## final 双视角消融

与 final-ViewQtoken 使用同一 finaldata_seg、双视角、原始 action 监督和训练预算。
不增加 camera/modality embedding；以下两组均保留 RGB 输入，不是纯分割策略。

| 实验 | RGB | 硬分割输入 | 受监督 Query |
| --- | --- | --- | --- |
| final-ViewQtoken | front + side | front + side | 3个共享任务 token、6个输出头 |
| final-SEM-FS-NoQ | front + side | front + side | 无 |
| final-RGB-FS-Qtoken | front + side | 无 | 与完整方法相同 |

### RGB + 分割，无 Query

不创建 Query token 或辅助输出头，不读取质量/几何监督缓存，只有 ACT 动作损失。
训练读取已有硬分割，推理仍调用对应冻结 U-Net 并硬化、过滤，与完整方法一致。

```bash
cd /home/qihan/data/lerobot

/home/qihan/miniconda3/envs/lerobot/bin/python mycode/train_mask_act_policy.py \
    --experiment final-SEM-FS-NoQ \
    --root /home/qihan/data/lerobot/data/finaldata_seg \
    --repo-id finaldata_seg \
    --rgb-keys observation.images.front observation.images.side \
    --state-keys observation.state \
    --mask-target-keys \
      observation.images.front_occluder \
      observation.images.front_object \
      observation.images.front_region \
      observation.images.front_tool \
      observation.images.front_leftarm \
      observation.images.front_rightarm \
      observation.images.side_occluder \
      observation.images.side_object \
      observation.images.side_region \
      observation.images.side_tool \
      observation.images.side_leftarm \
      observation.images.side_rightarm \
    --pretrained-segmentation-checkpoints \
      /home/qihan/data/lerobot/data/finaldata_seg/models/unet_front/model.ts \
      /home/qihan/data/lerobot/data/finaldata_seg/models/unet_side/model.ts \
    --log-freq 200 \
    --save-freq 20000 \
    --metric-loss-weight 0 \
    --action-loss-weight 1.0 \
    --output-dir outputs/train/semantic/final-SEM-FS-NoQ \
    --steps 100000 \
    --seed 1000 \
    --batch-size 8 \
    --chunk-size 60 \
    --n-action-steps 60 \
    --pretrained-backbone-weights ResNet18_Weights.IMAGENET1K_V1 \
    --device cuda \
    --num-workers 16 \
    --video-backend pyav \
    --rebuild-view
```

### 仅 RGB + 受监督 Query

这组不是“无监督 learnable token”：仍以双视角标签计算的相同几何目标和质量权重监督 Query。
分割标签仅用于离线生成辅助监督，不作为模型输入；训练无需解码 mask 视频，推理无需分割模型或标签。
复用前面的 `outputs/cache/final-ViewQtoken-v1/targets.npz`。
命令中的 mask key 列表记录监督类别定义，不代表策略读取这些图像。

```bash
cd /home/qihan/data/lerobot

/home/qihan/miniconda3/envs/lerobot/bin/python mycode/train_mask_act_policy.py \
    --experiment final-RGB-FS-Qtoken \
    --root /home/qihan/data/lerobot/data/finaldata_seg \
    --repo-id finaldata_seg \
    --rgb-keys observation.images.front observation.images.side \
    --state-keys observation.state \
    --mask-target-keys \
      observation.images.front_occluder \
      observation.images.front_object \
      observation.images.front_region \
      observation.images.front_tool \
      observation.images.front_leftarm \
      observation.images.front_rightarm \
      observation.images.side_occluder \
      observation.images.side_object \
      observation.images.side_region \
      observation.images.side_tool \
      observation.images.side_leftarm \
      observation.images.side_rightarm \
    --view-query-cache /home/qihan/data/lerobot/outputs/cache/final-ViewQtoken-v1/targets.npz \
    --log-freq 200 \
    --save-freq 20000 \
    --metric-loss-weight 1.0 \
    --action-loss-weight 1.0 \
    --output-dir outputs/train/semantic/final-RGB-FS-Qtoken \
    --steps 100000 \
    --seed 1000 \
    --batch-size 8 \
    --chunk-size 60 \
    --n-action-steps 60 \
    --pretrained-backbone-weights ResNet18_Weights.IMAGENET1K_V1 \
    --device cuda \
    --num-workers 16 \
    --video-backend pyav \
    --rebuild-view
```
两组均已通过真实数据两步训练和 checkpoint 重载推理检查，42项相关测试通过；未启动正式训练。

## final-ViewQtoken 双视角残差：MLP / GRU × RGBOnly / SemQ

四组都冻结同一个 `outputs/train/semantic/final-ViewQtoken/checkpoint_step_100000`，
使用 `data/finaldata_seg`，不重新训练 ACT、视觉编码器、U-Net 或 Query。

| 实验名 | 残差网络 | 残差分支的视觉与任务输入 |
| --- | --- | --- |
| final-ViewQtoken-Residual-MLP-RGBOnly-FS | MLP | 双视角 RGB 特征 |
| final-ViewQtoken-Residual-MLP-SemQ-FS | MLP | 双视角 RGB + 当前预测语义描述 + 当前计划的 SQtoken |
| final-ViewQtoken-Residual-GRU-RGBOnly-FS | GRU | 双视角 RGB 特征 |
| final-ViewQtoken-Residual-GRU-SemQ-FS | GRU | 双视角 RGB + 当前预测语义描述 + 当前计划的 SQtoken |

**消融范围**：RGBOnly 仅去掉残差分支的显式语义和 Query 输入；基础 ACT 的 RGB、语义、SQtoken
以及生成的基础动作均不变。四组均读取关节状态、速度、上一条指令、基础动作后缀和计划时间信息。
SemQ 将语义与 SQtoken 作为一组一起加入，不分别消融。

### 输入、监督与归一化

- 基础策略复用现有推理加载器和 checkpoint 的 state/action 统计、RGB 归一化；
  不重新切换为 ImageNet 统计或 0–1 关节缩放。U-Net 继续使用各自导出包中的预处理与硬分割过滤规则。
- 缓存与推理都从真实 RGB 调用冻结 U-Net；不读取标注 mask 作为残差输入。
  语义特征为双视角类别面积、质心、有效性、粗空间池化及对象关系，沿用已有残差描述器。
  硬分割中的置信字段仅反映类别存在性，不作为概率校准后的质量分数。
- SQtoken 是 ACT 编码器中三个共享任务 token 的隐藏特征，不是八维监督标签。
  每次 ACT 规划时提取，随对应计划缓存；快分支不重新运行整个 Transformer。
- 观测仍是 follower `observation.state`，监督仍是数据集原始 `action`（记录的 leader 指令），
  与本次基础 ACT 保持一致；不是之前 Follower-Delta / Anchor-Delta 实验。
- 沿用 SF-v2：残差上下文和误差按训练 episode 的 action 标准差缩放，
  输出还原为数据集原生关节单位并加到基础动作上。四组使用同一 seed=1000、按任务分层的 80/20 episode 划分，
  因此缩放因子相同；残差幅度上限取训练残差绝对值 P95，并限制在 [0.05, 5] 原生单位。
  验证集仅对残差头留出，基础 ACT 已见过这些示范，不能称作完整策略的未见轨迹验证。

### 共同配置

- ACT chunk=60、30帧重规划；残差每6帧更新，fast delay=3，预测9帧；30 FPS。
- GRU 保留8次观测记录，间隔6帧，跨度1.4秒；MLP 只读当前记录。
- hidden=128，batch=128，AdamW lr=1e-4、weight decay=1e-4，最多50 epochs、早停 patience=8。
- 使用原有归一化 Smooth L1 动作误差（beta=0.1）和幅值正则（权重1e-3）；
  不额外增加时间平滑损失。初始残差为零，不改变基础动作。
- 两种输入各生成一份离线缓存，MLP/GRU复用；同时运行时缓存写入自动加锁。
  正式训练后自动执行 checkpoint 重载、在线/离线窗口一致性检查及无机器人动作的视频回放测速。
  `fast delay=3` 是100ms时间预算，不是已经达到的性能保证，需检查各组 `shadow_result.json`。

### 训练命令

四组串行运行（不指定 GPU 编号，沿用当前环境）：

```bash
cd /home/qihan/data/lerobot
bash run_final_viewqtoken_residuals.sh all
```

单独运行任意一组，也可在不同 tmux 中执行：

```bash
bash run_final_viewqtoken_residuals.sh mlp-rgb
bash run_final_viewqtoken_residuals.sh mlp-semquery
bash run_final_viewqtoken_residuals.sh gru-rgb
bash run_final_viewqtoken_residuals.sh gru-semquery
```

恢复已有训练用 `RESUME=1 bash run_final_viewqtoken_residuals.sh <组名>`；
只打印命令用 `DRY_RUN=1 bash run_final_viewqtoken_residuals.sh all`。
输出：`outputs/train/residual/<实验名>/{best.pt,last.pt,metrics.jsonl,result.json,shadow_result.json}`；
日志：`outputs/logs/<实验名>.log`。实机需搭配原始完整的 `final-ViewQtoken` 运行目录，
选择对应 MLP/GRU 残差 `best.pt`，不可搭配另一个 ACT checkpoint。

### 代码验证（2026-10-19）

53项相关测试通过，包括原有残差回归测试、四组输入消融、真实 checkpoint 的基础动作一致性，
以及两条真实轨迹上的缓存生成、每组2个 epoch 的临时训练、重载与无机器人动作回放。
两种输入缓存的基础动作、目标、上下文、缩放、限幅和公共 RGB 特征逐项一致。
这些检查不代表正式收敛或实机成功率；本次仅生成代码和命令，未启动四组正式训练。

## 两路训练队列及归一化对齐基线（2026-10-19）

本次使用 `finaldata_seg`，新增以下两个纯 RGB 双视角基线，不覆盖之前的 ACT/DP：

| 实验 | RGB 归一化 | state/action 归一化 | 其他配置 |
| --- | --- | --- | --- |
| ACT-finaldata-seg-FS-DatasetNorm | 数据集 mean/std | MEAN_STD，与 final-ViewQtoken 一致 | chunk=60，100k steps |
| DP-1F-Full-finaldata-seg-FS-DatasetNorm | 数据集 mean/std | 保留 DP 原有 MIN_MAX | 单帧全图，horizon=64，100k steps |

新增 `train_lerobot_policy.py --image-normalization dataset`，关闭默认 ImageNet 统计覆盖；
不改变 ImageNet **预训练权重**，也不改变旧命令的默认行为。
两基线均不使用语义、Query、camera embedding，监督仍为原始 `action`。
DP 保持原有 DDPM 100/50 步与 `[-1,1]` 去噪裁剪，因此不强制改成 ACT 的动作标准化；
不能直接比较 DP 去噪 loss 与 ACT 的 L1 loss。

两个脚本各自串行，脚本之间并行：

```bash
# task1: MLP-RGBOnly-FS -> MLP-SemQ-FS -> ACT-finaldata-seg-FS-DatasetNorm
cd /home/qihan/data/lerobot
bash run_final_mlp_act.sh

# task2: GRU-RGBOnly-FS -> GRU-SemQ-FS -> DP-1F-Full-finaldata-seg-FS-DatasetNorm
cd /home/qihan/data/lerobot
bash run_final_gru_dp.sh
```

完整参数在脚本内；`DRY_RUN=1 bash <脚本名>` 可打印实际命令。
不指定 GPU 编号，沿用当前 CUDA 环境。遇到错误立即停止该队列，不继续跑后续实验。
重复启动有锁保护，已有基线输出目录不会被覆盖；残差需要恢复时可设置 `RESUME=1`。
ACT/DP 的日志分别位于 `outputs/logs/<实验名>.log`，模型输出在
`outputs/train/act/ACT-finaldata-seg-FS-DatasetNorm` 和
`outputs/train/diffusion/DP-1F-Full-finaldata-seg-FS-DatasetNorm`。

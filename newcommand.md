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

## final-FrontQtoken-FS

恢复旧版 front-only Qtoken 监督，视觉输入仍为 front/side RGB + front/side 硬语义图。
三个查询的输出维数为 `[1, 1, 2]`，共 4 个标量：

- object 像素占整幅 front 图像的比例；可靠的不可见帧监督为 0。
- object 与 region 质心的欧氏距离 / 图像对角线长度。恢复旧定义，进入区域不强制置零。
- front tool 的 PCA 线段左端点到 object 质心的二维向量，按图像宽高归一化。

沿用 final 数据的质量机制：相关类别异常时不监督；有效描述量按类别质量分数加权，
涉及两个类别时取最小质量分数。每个查询内部平均坐标，再平均三个查询；
不除以质量权重之和，低质量会降低监督强度。不是旧版权重归一化的复刻。
与 final-ViewQtoken 同时存在监督视角和区域距离定义两处差异，不应称为纯视角消融。

### 一次性缓存准备

将原始 12 路二值 mask 无损保存为双视角 uint8 类别图，并调用旧几何函数计算前视监督。
约占 26.5 GiB；原始数据不修改，不使用有损语义视频。
已有完整缓存时只校验并复用。缓存过期或不完整时应使用新的目录，避免覆盖已有工作。

```bash
cd /home/qihan/data/lerobot
/home/qihan/miniconda3/envs/lerobot/bin/python mycode/prepare_front_qtoken.py \
    --root data/finaldata_seg \
    --output-dir outputs/cache/final-FrontQtoken-FS-v1 \
    --workers 8
```

### 训练

```bash
cd /home/qihan/data/lerobot
/home/qihan/miniconda3/envs/lerobot/bin/python mycode/train_mask_act_policy.py \
    --experiment final-FrontQtoken-FS \
    --root /home/qihan/data/lerobot/data/finaldata_seg \
    --repo-id finaldata_seg \
    --rgb-keys observation.images.front observation.images.side \
    --state-keys observation.state \
    --mask-target-keys \
      observation.images.front_occluder observation.images.front_object \
      observation.images.front_region observation.images.front_tool \
      observation.images.front_leftarm observation.images.front_rightarm \
      observation.images.side_occluder observation.images.side_object \
      observation.images.side_region observation.images.side_tool \
      observation.images.side_leftarm observation.images.side_rightarm \
    --pretrained-segmentation-checkpoints \
      /home/qihan/data/lerobot/data/finaldata_seg/models/unet_front/model.ts \
      /home/qihan/data/lerobot/data/finaldata_seg/models/unet_side/model.ts \
    --view-query-cache /home/qihan/data/lerobot/outputs/cache/final-FrontQtoken-FS-v1/targets.npz \
    --semantic-label-cache /home/qihan/data/lerobot/outputs/cache/final-FrontQtoken-FS-v1 \
    --metric-loss-weight 1.0 --action-loss-weight 1.0 \
    --output-dir outputs/train/semantic/final-FrontQtoken-FS \
    --steps 100000 --seed 1000 --batch-size 8 \
    --chunk-size 60 --n-action-steps 60 \
    --pretrained-backbone-weights ResNet18_Weights.IMAGENET1K_V1 \
    --device cuda --num-workers 16 --video-backend pyav \
    --log-freq 200 --save-freq 20000 --rebuild-view
```

不改变数据集 RGB/action/state mean/std、学习率、精度模式、分辨率、动作监督或训练预算。
去掉 `--semantic-label-cache` 即回到原始二值 mask 解码路径；查询监督缓存仍须保留。
该类别图缓存也可用于 final-ViewQtoken / final-SEM-FS-NoQ，但各实验仍使用自身的查询监督配置。
训练不运行 U-Net；推理仍使用运行目录内复制的两套 U-Net，并保持 argmax + 导出过滤规则。
保存和部署时保留整个运行目录，包括 `mask_act_run_config.json` 和 `segmentation_models`。

### 本地验证（2026-10-20）

- 缓存已生成，108 episodes、61,740 帧；完整解码检查双视角 mask，无尺寸或重叠错误。
- 抽样 8 帧、双视角类别图与 PyAV 原始读取逐像素一致，旧几何函数的监督一致。
- 固定权重与随机种子时，缓存/原始路径的 loss、Query 和视觉投影梯度通过一致性检查；
  checkpoint 仅输入 RGB/state 即可输出 `[2, 60, 10]` 动作。
- 相关测试 44 passed、2 skipped（两个与本实验无关的可选残差测试）。
- RGB/state/action 的 mean/std 与源数据完全一致。

两次串行短训练各 200 步，batch=8、workers=16、原分辨率、同一 seed，排除前 20 步：

| 输入读取方式 | 平均秒/步 |
|---|---:|
| 原始 12 路二值 mask 视频 | 0.3135 |
| 无损 uint8 类别图缓存 | 0.09355 |

该次短测约快 3.35 倍；不是完整训练耗时承诺，也不是策略性能对比。
记录位于 `outputs/temp/frontq-cache-benchmark-20261020/timing.json`，两组目录中保留日志和
200 步 checkpoint。正式的 100,000 步训练尚未启动，不要将短测 checkpoint 当作训练完成模型。

## final-FrontQtoken-FS-SemMinus1To1

语义图归一化消融，不是新增损失正则项。与 `final-FrontQtoken-FS` 的唯一算法差异：
将送入共享 ResNet 的三通道语义颜色图从 ImageNet 标准化改为 `2*S-1`，
即固定 `mean=std=[0.5,0.5,0.5]`，输入颜色 `[0,1]` 映射到 `[-1,1]`。
双视角 RGB 的数据集 mean/std、state/action 标准化、U-Net 自身输入预处理、
颜色表、三个前视监督目标、质量权重、网络结构和训练超参数全部不变。
归一化参数保存在模型 buffer 与运行配置中，加载推理时保持一致。
复用 `final-FrontQtoken-FS-v1` 缓存，不需要重新生成标签或缓存，从头训练，不续训已有模型。

```bash
cd /home/qihan/data/lerobot
/home/qihan/miniconda3/envs/lerobot/bin/python mycode/train_mask_act_policy.py \
    --experiment final-FrontQtoken-FS-SemMinus1To1 \
    --root /home/qihan/data/lerobot/data/finaldata_seg \
    --repo-id finaldata_seg \
    --rgb-keys observation.images.front observation.images.side \
    --state-keys observation.state \
    --mask-target-keys \
      observation.images.front_occluder observation.images.front_object \
      observation.images.front_region observation.images.front_tool \
      observation.images.front_leftarm observation.images.front_rightarm \
      observation.images.side_occluder observation.images.side_object \
      observation.images.side_region observation.images.side_tool \
      observation.images.side_leftarm observation.images.side_rightarm \
    --pretrained-segmentation-checkpoints \
      /home/qihan/data/lerobot/data/finaldata_seg/models/unet_front/model.ts \
      /home/qihan/data/lerobot/data/finaldata_seg/models/unet_side/model.ts \
    --view-query-cache /home/qihan/data/lerobot/outputs/cache/final-FrontQtoken-FS-v1/targets.npz \
    --semantic-label-cache /home/qihan/data/lerobot/outputs/cache/final-FrontQtoken-FS-v1 \
    --metric-loss-weight 1.0 --action-loss-weight 1.0 \
    --output-dir outputs/train/semantic/final-FrontQtoken-FS-SemMinus1To1 \
    --steps 100000 --seed 1000 --batch-size 8 \
    --chunk-size 60 --n-action-steps 60 \
    --pretrained-backbone-weights ResNet18_Weights.IMAGENET1K_V1 \
    --device cuda --num-workers 16 --video-backend pyav \
    --log-freq 200 --save-freq 20000 --rebuild-view
```

验证（2026-10-20）：47 项测试通过，2 项可选残差测试跳过。
相同随机种子下两版模型初始化仅 `semantic_visual_mean/std` 不同；
4 步短训练、缓存一致性、梯度检查、checkpoint 重载与无标签推理均通过。
短测目录：`outputs/temp/frontq-sem-minus1to1-smoke-20261020`。
正式命令与对照组仅实验名称及输出目录不同，启动位置为 tmux `task2`（GPU 1）。

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

## final-ViewQtoken-v2 / v3

以原始 `final-ViewQtoken` 为基线，不继承 FrontQtoken 的 centroid 入区指标或 SemMinus1To1 标准化。
两版均输入双视角 RGB + 硬语义彩色图，共享 ResNet18，不加 camera/modality embedding。
RGB/state/action 沿用数据集统计；语义图沿用 ImageNet mean/std。动作仍监督数据集 action，不是 follower delta。

| Query | v2 | v3 |
|---|---|---|
| visibility | front/side 的 object 面积占比 | 相同 |
| tool-object | front PCA 左端点到 object 质心向量 | 同一 Query 再增加 front PCA 线段中点到 object 质心向量 |
| object-region | front/side object 像素到固定 ROI 的最大外部距离，完全入区为零 | 相同 |

v2 共6标量，v3 共8标量；中点是两个 PCA 端点的平均，不是 tool 像素质心。
PCA 沿用归一化图像坐标。两个版本都没有 side tool-object 监督。
每个头的 SmoothL1(beta=0.01) 用原质量分数加权、原异常门控；
分母为几何有效坐标数量，不以质量权重和归一化。
先在各任务内部平均各头，再平均三个任务，v3 不因多一个工具头增大工具任务总权重。
目标可靠不可见时面积监督零，向量/区域距离不监督；质量只影响辅助损失。

### 一次性准备

复用已校验的原版目标及无损语义 class-ID 缓存，不重复解码12路 mask 视频、不改变标签。
v3 全帧校验新 PCA 左端点与原监督一致，随后添加中点向量。
缓存按实验定义和源数据指纹严格校验；已有完整缓存复用，不覆盖旧文件。

```bash
cd /data/qihan/lerobot
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qihan/miniconda3/envs/lerobot/bin/python mycode/prepare_view_qtoken_revisions.py \
  --root data/finaldata_seg \
  --base-cache outputs/cache/final-ViewQtoken-v1 \
  --semantic-label-cache outputs/cache/final-FrontQtoken-FS-v1 \
  --output-parent outputs/cache --workers 8
```

两次训练均从头训练100000步，seed=1000，batch=8，chunk=60，学习率沿用默认1e-5，保存间隔20000步。
两个 tmux 会话独立并行，输出目录互不覆盖。

### v2（task1）

```bash
cd /data/qihan/lerobot
/home/qihan/miniconda3/envs/lerobot/bin/python mycode/train_mask_act_policy.py \
    --experiment final-ViewQtoken-v2 \
    --root /data/qihan/lerobot/data/finaldata_seg \
    --repo-id finaldata_seg \
    --rgb-keys observation.images.front observation.images.side \
    --state-keys observation.state \
    --mask-target-keys observation.images.front_occluder observation.images.front_object observation.images.front_region observation.images.front_tool observation.images.front_leftarm observation.images.front_rightarm observation.images.side_occluder observation.images.side_object observation.images.side_region observation.images.side_tool observation.images.side_leftarm observation.images.side_rightarm \
    --pretrained-segmentation-checkpoints /data/qihan/lerobot/data/finaldata_seg/models/unet_front/model.ts /data/qihan/lerobot/data/finaldata_seg/models/unet_side/model.ts \
    --view-query-cache /data/qihan/lerobot/outputs/cache/final-ViewQtoken-v2/targets.npz \
    --semantic-label-cache /data/qihan/lerobot/outputs/cache/final-FrontQtoken-FS-v1 \
    --log-freq 200 \
    --save-freq 20000 \
    --metric-loss-weight 1.0 \
    --action-loss-weight 1.0 \
    --output-dir outputs/train/semantic/final-ViewQtoken-v2 \
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

### v3（task2）

```bash
cd /data/qihan/lerobot
/home/qihan/miniconda3/envs/lerobot/bin/python mycode/train_mask_act_policy.py \
    --experiment final-ViewQtoken-v3 \
    --root /data/qihan/lerobot/data/finaldata_seg \
    --repo-id finaldata_seg \
    --rgb-keys observation.images.front observation.images.side \
    --state-keys observation.state \
    --mask-target-keys observation.images.front_occluder observation.images.front_object observation.images.front_region observation.images.front_tool observation.images.front_leftarm observation.images.front_rightarm observation.images.side_occluder observation.images.side_object observation.images.side_region observation.images.side_tool observation.images.side_leftarm observation.images.side_rightarm \
    --pretrained-segmentation-checkpoints /data/qihan/lerobot/data/finaldata_seg/models/unet_front/model.ts /data/qihan/lerobot/data/finaldata_seg/models/unet_side/model.ts \
    --view-query-cache /data/qihan/lerobot/outputs/cache/final-ViewQtoken-v3/targets.npz \
    --semantic-label-cache /data/qihan/lerobot/outputs/cache/final-FrontQtoken-FS-v1 \
    --log-freq 200 \
    --save-freq 20000 \
    --metric-loss-weight 1.0 \
    --action-loss-weight 1.0 \
    --output-dir outputs/train/semantic/final-ViewQtoken-v3 \
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

### 验证与查看

- 全量数据检查：`outputs/temp/viewqtoken-revisions-preflight.json`，108条轨迹、61,740帧，双视角无标签重叠，加载通过。
- 两版短程训练与 checkpoint 重载：`outputs/temp/final-ViewQtoken-v2-smoke`、`outputs/temp/final-ViewQtoken-v3-smoke`。
- 测试覆盖三任务等权、前视中点几何、质量加权、缓存定义隔离、缓存与原 mask 的损失/梯度一致性及 RGB-only 推理。
- 正式训练实时指标：`outputs/train/semantic/final-ViewQtoken-v{2,3}/metrics/train_metrics.jsonl`。
- 查看会话：`tmux attach -t task1` / `tmux attach -t task2`。此次分别使用 GPU 0 / GPU 1，不影响上述命令自行选择 CUDA 环境。
- 运行期间控制台日志暂存于 `outputs/train/semantic/.final-ViewQtoken-v{2,3}.train.log.tmp`；正常结束后移动到各输出目录的 `logs/train.log`。

## VIOLA-finaldata-FS：官方候选框双视角基线

直接使用官方 VIOLA 网络和预训练 Detic ProposalNetwork，不用分割标签、U-Net 或 Qtoken。
front 提供候选区域和全局特征，side 接入原版第二相机分支；保留10帧历史、因果 Transformer 和 GMM 单步动作头。
输入是 follower `observation.state`，监督仍是原始数据 `action`，不是 follower delta。
具体适配和复现边界见 [VIOLA_BASELINE.md](mycode/VIOLA_BASELINE.md)。

使用独立环境 `.cache/viola/venv/bin/python`，当前已经配置。重建时执行 `bash mycode/setup_viola_baseline.sh`。
不强制指定 GPU；需要分配时在命令前添加 `CUDA_VISIBLE_DEVICES=0` 或 `1`。

### 短程链路验证

三个物体各选4条完整轨迹，每类留1条独立验证，共9条训练、3条验证。以下输出已用于本次试跑，训练目录禁止覆盖；重新试跑请换输出目录。

```bash
cd /data/qihan/lerobot
.cache/viola/venv/bin/python mycode/viola_baseline.py prepare \
    --root data/finaldata_seg \
    --output outputs/cache/VIOLA-finaldata-FS-pilot/dataset.hdf5 \
    --per-class 4 --valid-per-class 1

.cache/viola/venv/bin/python mycode/viola_baseline.py train \
    --dataset outputs/cache/VIOLA-finaldata-FS-pilot/dataset.hdf5 \
    --output outputs/train/baselines/VIOLA-finaldata-FS-pilot \
    --steps 1000 --batch-size 16 --workers 4 --seed 1000

.cache/viola/venv/bin/python mycode/viola_baseline.py verify \
    --checkpoint outputs/train/baselines/VIOLA-finaldata-FS-pilot/checkpoint-001000.pt \
    --dataset outputs/cache/VIOLA-finaldata-FS-pilot/dataset.hdf5 \
    --output outputs/train/baselines/VIOLA-finaldata-FS-pilot/inference-verification.json \
    --frames 300
```

### 全量基线训练

使用和其他 finaldata 方法相同的108条示范。这里保留全部轨迹用于训练，不用训练集冒充验证集，最终性能用统一实机测试判断。
下面默认51epochs、batch16、AdamW 1e-4，沿用官方公开配置；如要设定100000步预算，将 `--epochs 51` 替换成 `--steps 100000`，不要声称不同方法相同步数等于相同训练曝光量。

```bash
cd /data/qihan/lerobot
.cache/viola/venv/bin/python mycode/viola_baseline.py prepare \
    --root data/finaldata_seg \
    --output outputs/cache/VIOLA-finaldata-FS-all/dataset.hdf5 \
    --per-class 0 --valid-per-class 0

.cache/viola/venv/bin/python mycode/viola_baseline.py train \
    --dataset outputs/cache/VIOLA-finaldata-FS-all/dataset.hdf5 \
    --output outputs/train/baselines/VIOLA-finaldata-FS \
    --epochs 51 --batch-size 16 --workers 4 --seed 1000 \
    --log-every 50 --save-every 5000
```

- RGB：全图缩放128×128、除255；state：训练集逐关节mean/std；action：训练集逐关节min/max到[-1,1]，匹配原版GMM的tanh均值。与ACT的action归一化不同是算法适配，不能直接比较二者原始loss。
- 候选框训练前离线生成，推理在线生成；未使用人工框。两者走同一Detic及预处理函数。
- 输出为10维原始数据单位的绝对关节指令；官方GMM采样可能越过训练范围，实机部署仍须现有限位/速度保护。
- 日志：训练目录 `metrics.jsonl`、`summary.json`；候选框预览：缓存目录 `dataset.previews/`。试跑不是收敛模型或实机成功率证明。
- 本次试跑已完成：9项测试通过，1000步后独立轨迹窗口上的归一化MAE从0.5016降至0.1322；300帧在线/离线候选框一致，端到端延迟中位数18.05ms、P95为29.67ms（不含相机/通信）。

全量流程一键运行（先预计算，再训练；任何步骤失败即停止，不覆盖已有训练目录）：

```bash
cd /data/qihan/lerobot
bash mycode/run_viola_finaldata_fs.sh
```

此次在 `tmux task1` 使用 `CUDA_VISIBLE_DEVICES=1` 启动上述流程；日志分别为
`outputs/logs/VIOLA-finaldata-FS/prepare.log` 和 `outputs/logs/VIOLA-finaldata-FS/train.log`。

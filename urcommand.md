# UR 数据训练命令

## UR-SemQ-v1：merged4 分割输入与五个任务查询（RGB / RGBD）

本实验使用 `data/realsense_rgbd_merged4_seg`，保留全部90条轨迹、181652帧及复位过程。
独立入口 `mycode/train_ur_semq_policy.py`，不修改 SO-ARM 的类别、查询布局、训练入口或旧 checkpoint。

| 实验 | 视觉输入 | 输出目录（outputs/train/下） |
|---|---|---|
| RGB + 语义 + Qtoken | front RGB + front 硬分割彩色图 | UR-SemQ-v1-RGB-realsense-merged4-fullprocess |
| RGBD + 语义 + Qtoken | front RGB + depth_preview + front 硬分割彩色图 | UR-SemQ-v1-RGBD-realsense-merged4-fullprocess |

### 模型与监督定义

- 共享可训练 ImageNet ResNet18；所有视觉输入使用 ImageNet mean/std；无相机/模态 embedding。
  RGBD 是同一相机的深度伪彩图。原始全分辨率668x742，不裁剪、不重采样。
- 动作、标准化与前面 UR ACT 一致：12维 follower 相邻帧 joint delta + 独立夹爪绝对0/1 BCE头；
  夹爪权重0.2、positive_weight=2.5，state/delta使用mean/std；预测/默认执行60步。
- 5个 encoder query token，独立输出维度 `[1,1,1,1,2]`：object面积、organ面积、object-region中心距离、occluder面积、object二维位置。
- 面积为可见像素数/(H*W)；二维位置为 `(cx/(W-1), cy/(H-1))`；距离为像素质心距离/图像对角线。
  距离不是入区判定，object落入region时不会强制为0。
- region每条episode单独用前20帧建立：质量>=0.8、非异常、面积>=20像素，至少10帧；
  质量加权多数投票，候选与参考mask的中位IoU>=0.8才接受。参考质量=候选平均质量*该中位IoU。
  距离监督从第21帧开始；参考不可靠则整条轨迹的距离头不监督，绝不搜索未来帧替代。
- 面积0仍监督；object不可见时不监督位置与距离。异常类别只屏蔽依赖该类别的查询，不屏蔽动作损失。
  其余查询乘对应类别质量分数，距离乘 `min(object当前质量, region参考质量)`。
  固定region只用于计算标签，不覆盖当前分割输入，不把标签直接输入策略，也不自动赋予策略历史记忆。
- 三个面积头以训练集可靠正样本的95%分位数（下限0.001）作固定尺度；距离/位置尺度为1。
  head预测缩放后的值，推理描述量乘回缓存中的scale。5头等权，位置两个坐标先平均。
  损失为 `原ACT损失 + 0.1 * 质量加权SmoothL1(beta=0.1)`，不以质量权重和归一化而抵消低质量降权。
- 单帧、batch8、100k steps、seed1000、LR1e-5、workers16、每20k保存一次；未开启混合精度。

### 加速与运行

```bash
cd /home/qihan/data/lerobot
# 首次离线合并7路二值mask为无损uint8类别缓存，并预计算描述量/质量权重/每episode区域。
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/qihan/miniconda3/envs/lerobot/bin/python -u mycode/prepare_ur_semq.py \
  --root data/realsense_rgbd_merged4_seg --output-dir outputs/cache/ur_semq_v1_merged4 --workers 12

# task1，RGB组
bash run_ur_semq_v1.sh rgb
# task2，RGBD组；可与上一组同时运行
bash run_ur_semq_v1.sh rgbd
```

两个模式共享只读无损类别缓存（约90GB十进制），但拥有独立派生训练视图、模型、优化器和日志。
训练不运行U-Net或逐帧计算几何，只解码RGB/深度并读取类别缓存；GPU查色表、uint8传输、persistent workers、
pin memory、prefetch、TF32/cuDNN benchmark。缓存锁防止两个脚本同时生成；已有输出不覆盖。

日志：`outputs/logs/<实验名称>.log`，完整分项均值在 `outputs/train/<实验名称>/metrics/train_metrics.jsonl`。
模型：`outputs/train/<实验名称>/checkpoints/last/pretrained_model`，包含U-Net TorchScript、过滤/颜色规则和`ur_semq.json`。
部署使用 `mycode/ur_semq_inference.py` 的 `URSemQInference`：原始RGB/深度[0,1]、follower弧度输入，
返回已解码的绝对关节动作块和6维描述量，不能再次累加delta。此适配器不控制真实机器人。
部署无需标注质量数组或query监督标签；U-Net使用随checkpoint保存的版本，不更新SO-ARM分割模型。

端到端预检入口：`mycode/check_ur_semq.py --output-dir outputs/dataset_checks/<新目录> --device cuda`。
两组会短训3步并验证checkpoint、冻结U-Net推理、动作解码和重载一致性；短训结果不是正式实验结果。

## realsense_rgbd_merged4：ACT / DP，RGB / RGBD，完整过程 follower delta

两队列分别串行执行 RGB -> RGBD，队列之间可并行。完整保留 90 条轨迹、181,652 帧，包含自动复位。
RGBD 指同一 RealSense 相机的 RGB + TURBO 深度伪彩图，不是两个相机视角，也不是原始深度值。

| 队列 | 实验名称（输出在 outputs/train 下同名目录） |
|---|---|
| task1 / ACT RGB | UR-FDeltaSeparateGrip-ACT-RGB-realsense-merged4-fullprocess |
| task1 / ACT RGBD | UR-FDeltaSeparateGrip-ACT-RGBD-realsense-merged4-fullprocess |
| task2 / DP RGB | UR-FDeltaGripAbs-DP-RGB-realsense-merged4-fullprocess |
| task2 / DP RGBD | UR-FDeltaGripAbs-DP-RGBD-realsense-merged4-fullprocess |

### 动作定义与对照条件

- 两类策略均输入当前 follower state；第 k 个关节目标是 `state[t+k+1,:12]-state[t+k,:12]`。
  不监督原始控制器 action，不使用固定锚点 delta。夹爪目标为下一帧 `state[...,12]` 的绝对 0/1 命令。
- ACT 沿用上一轮专用 UR 入口和独立夹爪 BCE 头。DP 的关节 delta 和夹爪绝对值一起参与扩散去噪，没有独立 BCE 头。
- DP 新增可选 follower 目标及推理解码；`--act-action-target` 是复用的历史参数名，此目标现也支持 diffusion。
  去噪结果先反归一化，关节从当前 state 累加，夹爪按 0.5 二值化。标准 policy + postprocessor 返回绝对关节指令。
  部署端不要再次累加。直接调用内部 `diffusion.generate_actions` 不包含此解码，应使用标准策略入口。
- 两类均单帧、原始 668x742 全图、ImageNet 预训练 ResNet18、图像 ImageNet mean/std、batch=8、100k steps、seed=1000、workers=16。
- ACT state/delta 使用 mean/std；DP state/delta 使用 MIN_MAX 到 [-1,1]，与去噪裁剪一致。两方法的 loss 数值不能直接比较。
- ACT 预测60步；DP horizon=64（符合 U-Net 下采样要求），两者默认执行60步。DP 不丢弃 episode 末尾采样点，补齐动作不计入去噪损失。
- DP 沿用 DDPM、100 个训练噪声等级、50 步推理；不是 DDIM。DP backbone LR=1e-5，其余LR=1e-4；ACT LR=1e-5。
- 每个方法内部 RGB/RGBD 仅视觉输入不同；跨 ACT/DP 仍有损失、归一化和结构差异，不是只替换骨干的严格消融。
- 实际采集平均约23.36 Hz，存储标称30 Hz；本轮不重采样。全流程关节统计小幅偏差作为已知问题保留，派生delta统计重新生成。

### 运行命令

```bash
# task1: ACT RGB -> ACT RGBD
cd /home/qihan/data/lerobot
bash run_ur_merged4_act.sh
```

```bash
# task2: DP RGB -> DP RGBD
cd /home/qihan/data/lerobot
bash run_ur_merged4_dp.sh
```

用 `DRY_RUN=1 bash run_ur_merged4_act.sh` 或 `DRY_RUN=1 bash run_ur_merged4_dp.sh` 查看完整独立命令。
脚本使用锁防止重复启动，拒绝覆盖任何已有实验输出；前一个实验失败时队列停止。没有自动续训或删除数据。
脚本本身不固定GPU；本次tmux启动分别设置 `CUDA_VISIBLE_DEVICES=0`（ACT）和 `1`（DP）。
实时日志：`outputs/logs/<实验名称>.log`；checkpoint：`outputs/train/<实验名称>/checkpoints/last/pretrained_model`。
DP 部署环境必须包含本次新增的 follower 解码代码，并加载同一 checkpoint 的预处理、后处理和统计量。

## realsense_rgbd_new1_train：完整过程 follower delta，RGB / RGBD 对照

保留全部 60 条轨迹、121,044 帧，包括人工操作和自动复位，不裁剪、不重采样。
两组都用 `mycode/train_bi_ur3_policy.py`，不使用原数据的控制器 `action` 值作为监督：

- 输入：当前 `observation.state`（12 个实际关节角，单位 rad；右夹爪命令 0/1）和选定图像。
- 关节目标：第 k 个输出为 `state[t+k+1, :12] - state[t+k, :12]`，k=0,...,59。
  这是相邻帧逐步 delta，不是固定锚点 delta；部署时从当前关节状态累加还原绝对指令。
- 夹爪：独立输出头预测 `state[t+k+1, 12]` 的绝对 0/1，使用加权 BCE，loss weight=0.2、positive weight=2.5，沿用前面的 UR 配置。
  这里的夹爪 state 是采集的二值命令，不是实际夹爪位置测量值。
- 分辨率：原始 H=668、W=742；RGBD 是同一相机的 RGB + 深度伪彩图，不是前侧双视角。
- 标准化沿用该 UR 入口：视觉使用 ImageNet mean/std；state 使用数据集 mean/std；delta action 的统计在派生数据视图中重新计算。
  两组使用相同 state/action 数据和统计，不修改源数据。
- 公共配置：ImageNet 预训练 ResNet18、100k steps、batch=8、seed=1000、workers=16；预测 60 步，默认执行 60 步。
  不增加相机或模态 embedding。两组分别生成独立派生视图和 checkpoint。
- 数据仍按标称 30 Hz 加载，但实际采集平均约 22.94 Hz；本命令不修正物理时间。
  60 个样本不能直接视为原示范的真实 2 秒；部署执行频率与重规划间隔须另行设置并核对。

### RGB：只使用 front 彩色图

```bash
cd /home/qihan/data/lerobot
conda run --no-capture-output -n lerobot python -u mycode/train_bi_ur3_policy.py \
  --root /home/qihan/data/lerobot/data/realsense_rgbd_new1_train \
  --repo-id local/realsense_rgbd_new1_train \
  --image-keys observation.images.front \
  --state-keys observation.state \
  --act-action-target follower_joint_delta_gripper_absolute \
  --act-follower-state-key observation.state \
  --act-gripper-loss-weight 0.2 \
  --act-gripper-positive-weight 2.5 \
  --output-dir outputs/train/UR-FDeltaSeparateGrip-ACT-RGB-realsense-new1-fullprocess \
  --job-name UR-FDeltaSeparateGrip-ACT-RGB-realsense-new1-fullprocess \
  --chunk-size 60 \
  --n-action-steps 60 \
  --pretrained-backbone-weights ResNet18_Weights.IMAGENET1K_V1 \
  --batch-size 8 \
  --steps 100000 \
  --seed 1000 \
  --num-workers 16 \
  --device cuda \
  --video-backend pyav \
  --log-freq 200 \
  --save-freq 20000 \
  --eval-freq 0
```

### RGBD：front 彩色图 + 同视角深度伪彩图

```bash
cd /home/qihan/data/lerobot
conda run --no-capture-output -n lerobot python -u mycode/train_bi_ur3_policy.py \
  --root /home/qihan/data/lerobot/data/realsense_rgbd_new1_train \
  --repo-id local/realsense_rgbd_new1_train \
  --image-keys observation.images.front observation.images.depth_preview \
  --state-keys observation.state \
  --act-action-target follower_joint_delta_gripper_absolute \
  --act-follower-state-key observation.state \
  --act-gripper-loss-weight 0.2 \
  --act-gripper-positive-weight 2.5 \
  --output-dir outputs/train/UR-FDeltaSeparateGrip-ACT-RGBD-realsense-new1-fullprocess \
  --job-name UR-FDeltaSeparateGrip-ACT-RGBD-realsense-new1-fullprocess \
  --chunk-size 60 \
  --n-action-steps 60 \
  --pretrained-backbone-weights ResNet18_Weights.IMAGENET1K_V1 \
  --batch-size 8 \
  --steps 100000 \
  --seed 1000 \
  --num-workers 16 \
  --device cuda \
  --video-backend pyav \
  --log-freq 200 \
  --save-freq 20000 \
  --eval-freq 0
```

首次执行会生成派生训练视图；未设置 `--overwrite-output`，不会自动删除已有训练结果。
此处只提供命令，不自动启动训练或同步 Git。

## test1 基础 ACT：关节状态输入，关节目标输出

`observation.state` 和 `action` 均为 13 维：左臂 6 个关节、右臂 6 个关节、右夹爪 1 维。
`observation.left_tcp_pose` 与 `observation.right_tcp_pose` 仅作为数据记录保留，不输入以下策略。

### 单视角：front

```bash
conda run --no-capture-output -n lerobot python -u mycode/train_lerobot_policy.py \
  --policy-type act \
  --root /home/qihan/data/lerobot/data/test1 \
  --repo-id local/test1 \
  --image-keys observation.images.front \
  --state-keys observation.state \
  --output-dir outputs/train/ACT-test1-front-joint \
  --job-name ACT-test1-front-joint \
  --chunk-size 60 \
  --n-action-steps 60 \
  --pretrained-backbone-weights ResNet18_Weights.IMAGENET1K_V1 \
  --batch-size 8 \
  --steps 100000 \
  --num-workers 16 \
  --device cuda \
  --video-backend pyav \
  --rebuild-view
```

### 双视角：front + top

```bash
conda run --no-capture-output -n lerobot python -u mycode/train_lerobot_policy.py \
  --policy-type act \
  --root /home/qihan/data/lerobot/data/test1 \
  --repo-id local/test1 \
  --image-keys observation.images.front observation.images.top \
  --state-keys observation.state \
  --output-dir outputs/train/ACT-test1-front-top-joint \
  --job-name ACT-test1-front-top-joint \
  --chunk-size 60 \
  --n-action-steps 60 \
  --pretrained-backbone-weights ResNet18_Weights.IMAGENET1K_V1 \
  --batch-size 8 \
  --steps 100000 \
  --num-workers 16 \
  --device cuda \
  --video-backend pyav \
  --rebuild-view
```

两组实验除视觉输入数量外配置一致。60 步在 30 Hz 数据上对应 2 秒动作块。

## test1 混合动作目标：关节 delta + 绝对夹爪状态

四个实验的第 13 维夹爪都预测下一帧绝对二值状态，并使用加权 BCE。`FDelta` 的前 12 维
预测相邻 follower 状态差并在推理时逐步累加；`FAnchorDelta` 的前 12 维预测相对当前规划
状态的偏移，不累计之前的预测误差。

| 实验 | 视角 | 关节目标 |
| --- | --- | --- |
| `UR-FDeltaGripAbs-ACT-v1-F-test1` | front | 逐步 delta |
| `UR-FDeltaGripAbs-ACT-v1-FT-test1` | front + top | 逐步 delta |
| `UR-FAnchorDeltaGripAbs-ACT-v1-F-test1` | front | 固定锚点 delta |
| `UR-FAnchorDeltaGripAbs-ACT-v1-FT-test1` | front + top | 固定锚点 delta |

以下是双视角逐步 delta 的独立训练命令；其余三条命令由脚本使用相同公共配置生成：

```bash
conda run --no-capture-output -n lerobot python -u mycode/train_lerobot_policy.py \
  --policy-type act \
  --root /home/qihan/data/lerobot/data/test1 \
  --repo-id local/test1 \
  --image-keys observation.images.front observation.images.top \
  --state-keys observation.state \
  --act-action-target follower_joint_delta_gripper_absolute \
  --act-follower-state-key observation.state \
  --act-action-representation absolute \
  --act-gripper-loss-weight 0.2 \
  --act-gripper-positive-weight 2.6 \
  --output-dir outputs/train/UR-FDeltaGripAbs-ACT-v1-FT-test1 \
  --job-name UR-FDeltaGripAbs-ACT-v1-FT-test1 \
  --chunk-size 60 \
  --n-action-steps 60 \
  --pretrained-backbone-weights ResNet18_Weights.IMAGENET1K_V1 \
  --batch-size 8 \
  --steps 100000 \
  --seed 1000 \
  --num-workers 16 \
  --device cuda \
  --video-backend pyav \
  --rebuild-view
```

一键串行运行四个实验：

```bash
./run_ur_test1_fdelta_gripper_ft.sh
```

## test1_full 双视角 follower delta + 独立夹爪头

使用专用入口 `mycode/train_bi_ur3_policy.py`。两个实验都输入 front + top RGB 和当前
`observation.state`；前 12 维只用未来 follower state 生成关节监督，不使用 leader action。
第 13 维由独立分类头预测下一帧 follower 夹爪的绝对 `0/1` 状态，并使用
`BCEWithLogitsLoss(pos_weight=2.5)`。默认使用两个视角的原始分辨率；只有显式传入
`--image-size HEIGHT WIDTH` 时才会等比例缩放并补边。

### 双视角逐步 delta（推理时累计）

以下命令使用原始相机分辨率。`fullres-v2` 用于避开早先实际采用 `240x320`、但被误命名为
`fullres` 的已有输出目录。

```bash
conda run --no-capture-output -n lerobot python -u mycode/train_bi_ur3_policy.py \
  --root /home/qihan/data/lerobot/data/test1_full \
  --repo-id local/test1_full \
  --image-keys observation.images.front observation.images.top \
  --state-keys observation.state \
  --act-action-target follower_joint_delta_gripper_absolute \
  --act-follower-state-key observation.state \
  --act-gripper-loss-weight 0.2 \
  --act-gripper-positive-weight 2.5 \
  --output-dir outputs/train/UR-FDeltaSeparateGrip-ACT-FT-test1-fullres-v2 \
  --job-name UR-FDeltaSeparateGrip-ACT-FT-test1-fullres-v2 \
  --chunk-size 60 \
  --n-action-steps 60 \
  --pretrained-backbone-weights ResNet18_Weights.IMAGENET1K_V1 \
  --batch-size 8 \
  --steps 100000 \
  --seed 1000 \
  --num-workers 16 \
  --device cuda \
  --video-backend pyav \
  --rebuild-view
```

### 双视角固定锚点 delta（不累计前序预测误差）

```bash
conda run --no-capture-output -n lerobot python -u mycode/train_bi_ur3_policy.py \
  --root /home/qihan/data/lerobot/data/test1_full \
  --repo-id local/test1_full \
  --image-keys observation.images.front observation.images.top \
  --state-keys observation.state \
  --act-action-target follower_joint_anchor_delta_gripper_absolute \
  --act-follower-state-key observation.state \
  --act-gripper-loss-weight 0.2 \
  --act-gripper-positive-weight 2.5 \
  --output-dir outputs/train/UR-FAnchorDeltaSeparateGrip-ACT-FT-test1-full \
  --job-name UR-FAnchorDeltaSeparateGrip-ACT-FT-test1-full \
  --chunk-size 60 \
  --n-action-steps 60 \
  --pretrained-backbone-weights ResNet18_Weights.IMAGENET1K_V1 \
  --batch-size 8 \
  --steps 100000 \
  --seed 1000 \
  --num-workers 16 \
  --device cuda \
  --video-backend pyav \
  --rebuild-view
```

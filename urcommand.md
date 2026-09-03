# UR 数据训练命令

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
  --num-workers 8 \
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
  --num-workers 8 \
  --device cuda \
  --video-backend pyav \
  --rebuild-view
```

一键串行运行四个实验：

```bash
./run_ur_test1_fdelta_gripper_ft.sh
```

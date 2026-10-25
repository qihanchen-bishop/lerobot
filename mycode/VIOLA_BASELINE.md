# VIOLA 双视角基线

## 实现边界

此基线直接实例化官方 `CenterNetSpatialTemporalPolicy`，不是用 ACT 加候选框模拟 VIOLA。
数据读取 `finaldata_seg` 中的 front RGB、side RGB、`observation.state` 和原始 `action`；**不读取 mask、U-Net 输出、质量标签或 Qtoken 监督**。

官方来源：

- [VIOLA 论文](https://proceedings.mlr.press/v205/zhu23a/zhu23a.pdf)。
- [官方代码](https://github.com/UT-Austin-RPL/VIOLA)，固定 commit `1e8b5ae90e73a1c2a33a369323117cd8d7b0ef36`。
- 官方 Detic 子模块 `ee8b103f6de5faf4d21cfe17ab3f8bc3516189b4`，Detectron2 `9258799e4e72786edd67940872e0ed2c4387aac5`。
- 官方依赖中的 robomimic commit `490dd4d4f758c47357d314d47ffa5bb4aefa8d65`。
- Detic 权重 `Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size.pth`；SHA256 为 `17fd4f671780ce1fa9389b8f627cb3e86bf82e143871c0c076c990312377a75a`。

## 保留与适配

| 部分 | 本基线 |
|---|---|
| 候选区域 | 官方冻结 Detic ProposalNetwork，从 RGB 自动生成类别无关框，不提供目标类别提示 |
| 候选筛选 | 官方 NMS=0.05，128×128 坐标下面积大于16、小于4096，按模型返回顺序取前15个；15对应论文实机设置 |
| 主视觉分支 | front 的全局特征、候选区域 RoIAlign 特征和框位置编码 |
| 第二相机 | side 替代官方 wrist/eye-in-hand 上下文分支，不额外设计第二套候选 token |
| 时间建模 | 官方10帧历史、因果 Transformer、action token |
| 输出 | 官方5分量 GMM，当前时刻10维绝对关节指令，不是 delta，也不是60步 action chunk |
| Backbone | 官方配置 ResNet18，不加载 ImageNet 权重；候选框模型本身使用官方预训练权重 |
| 数据增强 | 官方 `stack_viola` 配置的平移、随机擦除、框位置噪声；不修改上游网络 |
| 优化 | AdamW，学习率1e-4，weight decay=1e-4，梯度裁剪10，按epoch周期余弦重启，默认51epochs、batch16 |

论文应将它称作 **VIOLA adapted to our bimanual setup**。官方仓库不包含其真实机器人实验配置；这里采用公开 `stack_viola` 网络/优化配置及论文的实机 top-15 设置，不能声称逐项复现了未公开的真实机器人设置。

平台适配仅包括 front/side 的输入映射、10维双臂 state/action、移除本数据不存在的夹爪状态分支，以及必要的数据尺度变换。两台固定相机与原来的工作空间相机+腕部相机不完全等价，需披露。数据维持原始30Hz，因此10帧历史的时间跨度也与其他采样频率不同。

## 归一化与时序

- RGB：完整视野缩放到128×128，转RGB后除以255；不使用 ImageNet mean/std。候选框和策略使用同一缩放图像。
- State：仅由训练轨迹拟合逐关节均值/标准差，训练和推理共用。
- Action：仅由训练轨迹拟合逐关节 min/max，映射到[-1,1]；常量维度保护除零。官方 GMM 均值含 `tanh`，因此不能直接将原始约[-100,100]指令或无界z-score目标送进去。
- 验证集不参与统计拟合，超出训练范围的验证目标不裁剪。推理反变换回数据原始单位，**不擅自将这些单位当作角度制**。
- 监督为 `state[t], RGB[t] -> action[t]`，沿用数据已有指令时间戳；不改成 follower future state，不平移4帧。
- 原版 SequenceDataset 构造不跨 episode 的10帧序列，末端重复填充；每个时刻的注意力只读取当前及过去 token。
- 在线推理仅编码新到图像，缓存最近10帧特征，每次输出一步 GMM 采样动作；每条轨迹开始调用 `reset()`。

## 缓存和运行

入口为 `mycode/viola_baseline.py` 的 `prepare`、`train`、`verify` 子命令。命令见根目录 `newcommand.md`。

`prepare` 保存 RGB、Detic 框、归一化 state/action 与原始动作到 HDF5。训练不重复运行候选框网络；实机推理必须在线运行同一个 Detic。缓存记录数据文件哈希、视频大小/修改时间、时间戳、上游版本及模型校验和。完整缓存不覆盖；中断缓存可按已完成的 episode 继续。候选框不足15个会明确报错，不会偷换为人工框或重复框。

预览 `dataset.previews/episode-*.jpg` 用于人工检查候选框。类别无关框可以落在背景、机械臂或局部物体上，不是语义真值；能生成15个框不等于一定覆盖任务目标。

训练输出包括 `run.json`、`metrics.jsonl`、checkpoint 和完成后的 `summary.json`。已有训练目录拒绝覆盖。训练 checkpoint 目前用于加载评估；此轻量入口没有提供中断训练续跑，正式长程训练需保留日志/checkpoint。

## 验证与解释

短程验证按三个物体各4条完整轨迹选取，共12条；每类最后一条独立留作验证，9条训练、3条验证。该划分只是检查链路，不是论文最终108条示范的训练结果，也不是严格控制位置的新位置泛化评估。

核对项：数据/视频帧对齐、训练集统计、GMM形状与反向梯度、因果注意力、逐帧历史与批量历史输出一致性、checkpoint重载，以及未见轨迹上的NLL和动作MAE。

NLL可为负，不能与ACT的L1或DP的去噪loss直接比较。报告的MAE使用GMM期望动作，仅用于观察拟合变化；官方在线策略仍为GMM采样，实机成功率必须另外测试。正式比较应使用同一训练示范、同一动作定义、同一评估网格和超时规则，并记录各方法训练预算；不以相同步数等同于相同样本曝光量。

`verify` 只回放录制数据，测量包含Detic的端到端延迟；不连接机械臂。相机采集、通信和安全限制不包含在这个延迟里。GMM采样并不天然限制在[-1,1]，真实机器人接入需要现有的关节限位/速度保护，不能直接把本适配器当作安全控制器。

## 环境隔离

### 本地策略测试界面

在原来的 `lerobot` 环境运行 `python mycode/gui_eval_lerobot_policy.py`，
重启界面后选择 **Config preset → VIOLA**。预设位于
`mycode/eval_configs/finaldata/VIOLA.json`，加载
`outputs/train/finaldata/VIOLA/checkpoint-196809.pt`，结果保存到
`eval/finaldata/viola/VIOLA`。目录形式要求只有一个 `checkpoint-*.pt`；
有多个权重时需要在预设中明确指定文件，避免自动选错。

采用双视角、3×4 网格、每格两次、40 秒时限和同一视觉判定阈值。
VIOLA 每个新观测预测一个动作，因此预测长度和重规划间隔均为 1，
同步执行、关闭动作块融合/自适应重规划/AMP、锁定夹爪。
不能沿用 ACT 的 30 步重规划间隔。

`mycode/viola_eval.py` 在 `.cache/viola/venv` 子进程中运行官方模型与 Detic；
主界面继续使用原环境。子进程没有硬件连接，异常或通信超时会向界面报错；
每轮结束释放子进程。RGB 颜色转换、10 维关节顺序和训练归一化在适配层核对；
动作反归一化只做一次，不替换官方 GMM 采样。界面参数统计包含策略及 Detic，
两者的明细、实际 checkpoint 和推理日志路径另存到每轮的 `viola` 元数据。

新机器首次配置可运行：

```bash
BASE_PYTHON="$(which python)" bash mycode/setup_viola_baseline.sh
```

本机离线接入验证位于 `result/viola_gui_verification_20261023/`，只回放已有数据，
没有操作机械臂。目标频率保持 30 Hz；本机包含 Detic 和进程通信的单步推理约
33–40 ms，实际闭环频率需以 `[CONTROL] actual_fps` 为准，不能使用训练服务器的延时替代。

`.cache/viola/venv` 通过 `--system-site-packages` 继承服务器现有 Torch2.7.1/CUDA12.8，同时局部覆盖旧代码所需的 numpy1.26、Pillow9.5、timm0.4.12、kornia0.6.1 等。没有降级现有 ACT/DP 环境。

旧Torch1.10不适合当前Blackwell显卡，因此保持算法实现不变而适配依赖。Detectron2编译其CPU扩展，所用标准GPU运算由Torch/torchvision执行；当前R50候选网络不需要额外的自定义CUDA算子。孤立环境可能因继承未使用的主环境包出现pip依赖提示，不能据此将它用于其他策略训练。

重建入口：`bash mycode/setup_viola_baseline.sh`。源码及权重下载到忽略目录，不提交第三方权重或修改官方源码。

兼容处理还包括为旧 robomimic 的 padding mask 恢复 `np.bool` 别名，以及避免其全局 cuDNN benchmark 开关改变候选框计算。两项均不改变网络结构、监督或候选筛选规则。

## 本次验证结果（2026-10-22）

- 12条完整轨迹、7,539帧；训练5,778帧，独立验证1,761帧。各帧面积过滤后的候选数量为29–57，中位数44，均足够取15框。
- 1000步训练完成，约38秒（不含候选框预计算、模型初始化和初次验证）。没有 NaN/Inf；梯度范数日志是裁剪前的值，实际裁剪阈值为10。
- 固定验证窗口共192个10帧序列，覆盖三条验证轨迹；不是全部验证帧的独立样本统计。

| 独立轨迹验证指标 | 初始化 | 1000步后 |
|---|---:|---:|
| GMM NLL | 9.3142 | -1.2089 |
| GMM期望动作的归一化MAE | 0.5016 | 0.1322 |
| GMM期望动作的原始数据单位MAE | 22.0304 | 4.9249 |

- 最后50步平均训练NLL为-11.5797，明显低于验证NLL；只能说明短程训练可运行并开始拟合，不能宣称泛化充分或已收敛。
- 验证动作约2.03%的标量超出训练轨迹min/max范围，没有裁剪这些目标；保留了真实验证难度。
- 9项自动测试通过，包括原版序列加载、episode边界、RGB尺度、归一化、反向梯度、因果性与缓存历史一致性。Checkpoint重载的期望动作最大差为0。
- 300帧真实视频回放，在线和离线RGB、15框坐标逐帧完全一致；输出为300×10个有限值。只回放观测，未连接机器人。
- RTX PRO6000 Blackwell上，去掉前10帧热身后，端到端延迟中位数18.05ms、P95为29.67ms，含Detic/策略/反归一化，不含采集通信。这不是30Hz实时保证，负载变化及尾部延迟仍需部署评估。

结果路径：

- `outputs/cache/VIOLA-finaldata-FS-pilot/prepare.log` 与 `dataset.previews/`。
- `outputs/train/baselines/VIOLA-finaldata-FS-pilot/train.log`、`metrics.jsonl`、`summary.json`。
- 同目录 `checkpoint-001000.pt`、`inference-verification.json`、`inference.log`、`tests.log`、`environment.txt`。

用户随后授权的全量训练已于2026-10-22 16:35（UTC+8）完成。运行入口为 `mycode/run_viola_finaldata_fs.sh`，会话为 `tmux task1`（GPU 1）。流程日志位于 `outputs/logs/VIOLA-finaldata-FS/`，训练输出位于 `outputs/train/baselines/VIOLA-finaldata-FS/`。

2026-10-23复核：108条示范、51epochs、196,809步全部完成，纯训练及结束验证约107.8分钟；40个checkpoint齐全。3,937条训练指标记录的NLL和梯度范数均为有限值。最初5,000步的日志窗口平均NLL为-13.68，190,000步之后为-32.05；最后一条NLL为-34.28，连续密度的负NLL是允许的，不代表错误。

最终权重 `checkpoint-196809.pt` 重载输出最大差为0；300帧录制数据回放的RGB和候选框与缓存逐帧完全一致，输出300×10个有限动作值。该次测量的端到端延迟中位数18.87ms、P95为25.84ms，不含采集/通信，详情见训练目录 `inference-verification.json`。这次全量训练没有留出验证轨迹，回放也不是闭环机器人评估，因此不能据此断言泛化效果或任务成功率。

较早的 `VIOLA-finaldata-FS-pilot-numpy-compat-preflight` 仅是发现旧NumPy别名兼容问题时留下的初始化记录，没有执行参数更新，不是另一个训练结果。

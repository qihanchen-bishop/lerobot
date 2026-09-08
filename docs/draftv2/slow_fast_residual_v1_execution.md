# Slow-fast residual v1：实现、执行与验收记录

本文件对应 `lecodex-slow-fast-residual-v1.md` 的离线实现任务，不执行真实机器人或公网部署。实际训练/验证结果以各目录 `result.json`、`metrics.jsonl`、`shadow_result.json` 与 runner 退出码为准；仅有进程或 checkpoint 不代表通过验收。

## 基线与实验合同

| 阶段 | 基线 | 原 checkpoint |
| --- | --- | --- |
| A | ACT-bettersetup-front-side，双 RGB，60 步 | `outputs/train/act/ACT-bettersetup-front-side/checkpoints/last/pretrained_model` |
| B | UNET-SEM-V5-FS，双 RGB + 双语义图 | `outputs/train/semantic/UNET-SEM-v5-front-side-bettersetup-v5/checkpoint_step_100000` |
| C | UNET-SEM-V5-FS-QTOKEN，双 RGB + 双语义图 + 3 个监督 query | `outputs/train/semantic/UNET-SEM-v5-front-side-QToken-bettersetup-v5/checkpoint_step_100000` |

B/C 选择已完成的最终 100k checkpoint，不称作验证集最优 checkpoint：旧训练没有提供可用于选择最优版本的独立验证指标。

使用 `data/bettersetup_v5` 的 90 条 episode，53,723 帧，30 Hz，front/side 两个 RGB 视角。与 `data/bettersetup` 的全部 state/action/timestamp 已逐 episode 核对一致；抽查 episode 0/30/60/89 的两视角 RGB 文件 SHA256 相同。标准 LeRobot/PyAV 加载 sample 0 成功，包括双视角 RGB 与所有 mask，数值有限。

state 是 follower 观测；监督仍为原 ACT 所学的 dataset action，即记录的 leader 指令，不混入 follower delta/anchor delta。10 个关节必须保持 action/state 的原顺序。SO 默认电机配置和数据范围对应校准后的 -100..100 指令尺度，不能把这些数值写成角度。代码统一称 `dataset_native`，反归一化后加残差；输出不连接机器人。

残差训练按 task 分层、episode 级 80/20 划分，seed=1000。初始位置没有可靠结构化索引，因此不伪称按位置分层。保存具体清单在各缓存 `split.json`。基线可能已见过全部 episode，所以这是“残差头未见 episode”验证，不是完整系统的新数据泛化。

## 共享实现

- `mycode/slow_fast_residual.py`：类型/配置、计划时钟、因果 context、固定窗口 GRU/MLP、episode 数据集、双速率 shadow 执行器、损失与诊断。
- `mycode/train_slow_fast_residual.py`：原策略/处理器适配、最新双视角特征、顺序视频解码与离线缓存、训练/恢复、checkpoint reload、在线窗口一致性和 shadow 回放。
- `tests/test_slow_fast_residual.py`：边界与数值测试。
- `run_slow_fast_residual_stage.sh`：独立 tmux 内的测试、缓存、smoke、恢复与正式训练队列；每阶段互斥锁，已有产物需显式 `RESUME=1`。

主策略及 backbone 冻结。快分支使用每次最新 front/side 的共享 backbone 权重，各自保持全视野 resize 到 180x320，做 2x3 空间池化，拼接为 6144 维。只训练空间特征投影、context 投影、单层 hidden=128 GRU 和残差头。MLP 对照只看当前记录。

快更新 15 Hz，W=8，首尾约 0.467 s；慢计划 1 Hz。输出两个 30 Hz 槽位的有界残差。记录当前/上一状态、速度、上一条已发指令、6 步计划及相对 state 的差、有效性、计划年龄、时间间隔和切换标志。采样不越过 episode；历史是当时可获得的计划条件，不被后来的计划重写。

默认模拟慢推理延迟 3 帧、快推理延迟 1 帧，均可配置。目标槽位是观察帧加快延迟，不从旧计划第零项重复执行。计划版本切换丢弃旧残差，重复/不同步/过旧图像不作为新反馈。失效时退回尚有效的原计划，计划也失效则返回无命令。

残差头零初始化，逐关节上限为 `min(训练子集残差绝对值P95, 5个原生单位)`，下限仅用于数值稳定。这是离线实验限制，不是经过硬件认证的安全值。执行器提供可选位置/速度/加速度约束；第一版没有配置可直接用于真机的安全参数。

损失：以训练子集 action std 缩放后的 Smooth L1（beta=0.1），加 1e-3 的残差平方项；不对残差再减 action mean。AdamW，lr=1e-4，weight decay=1e-4，梯度裁剪1，batch128，最多50 epochs、验证连续8次无改进早停。按修正后的归一化 MAE 保存 best，同时保存 last、optimizer 与5-epoch快照。

缓存包含源元数据/视频文件属性、模型 SHA256、配置和独立 episode shard。缓存仅保存冻结特征，训练不重跑视觉网络。缓存与在线统一禁用 TF32：初次默认精度下批量/单帧最大差约0.00458，未通过验收；该尝试保留在 `outputs/cache/slow_fast_v1_A_tf32_attempt`，未用于正式训练。禁用 TF32 后 A 最大差约2.6e-5。

## 运行方式

需要包含 torch、OpenCV、PyArrow、pytest 和本仓库依赖的 Python 环境。脚本不设置 GPU 编号，但运行前必须检查资源；不要同时无条件启动三阶段。A 全部通过后再运行 B/C。

```bash
tmux new-session -d -s sf-v1-A -c /data/qihan/lerobot \
  'PYTHON=/home/qihan/miniconda3/envs/lerobot/bin/python bash run_slow_fast_residual_stage.sh A'
```

同类启动方式将阶段改为 B/C，并使用不冲突的 tmux 名称。环境路径仅为本机命令示例，Python 源码不依赖该路径。中断后核对运行进程和日志，再用 `RESUME=1` 恢复，不能启动第二个争用相同输出目录的任务。

产物根目录：`outputs/train/residual/SF-v1-{A,B,C}-FS/`，包含 `runner.log`、`process.txt`、`exit_code`、`smoke/`、`gru8/`；A 额外有 `mlp/`。特征缓存位于 `outputs/cache/slow_fast_v1_{A,B,C}/`。

## 必须明确的限制

1. 训练后续画面仍来自专家执行，上一条指令也来自示范。快模型可能依赖这个强提示；离线误差改善不保证在自身闭环轨迹上有效。
2. Shadow 使用实际视频、state 和网络，但不生成物理反事实画面，因此不是闭环仿真。即便记录了修正命令，也未发送到机器人。
3. 吞吐与延迟分开报告。缓存 batch16 的每帧耗时不是单帧实时延迟；shadow 测量 batch1 编码+GRU，另外统计模拟延迟超限率。相机和通信成本没有包含，不能宣称已达到实机15 Hz。
4. 当前没有训练分割、没有 GT mask 作为策略输入、没有 RL、没有可学习执行长度、阶段切换或 CLF。
5. MLP 与 GRU 必须用同一基线、验证集、缩放和调度比较。不同基线之间不能只按残差后的误差大小宣称某个语义/Query机制更强。

## 阶段 A 验收

全部90条缓存完成，72/18条 episode 划分得到21,583/5,076个快观测样本。
GRU 18 epochs 早停，最佳为零起始 epoch9；MLP 20 epochs 早停，最佳为 epoch11。
同一验证样本的归一化 MAE：基础计划0.0589675，MLP0.0456694，GRU0.0444937。
原生单位 MAE：基础1.25255，MLP0.97375，GRU0.94668。不能将其与旧训练的 L1+KL 总 loss 直接比较。

GRU 残差 RMS=1.41585，饱和率0，逐关节分量目标可达率94.08%。在线/离线窗口最大差7.15e-7。
初次smoke和optimizer恢复均通过；12项基础/时序测试通过，最后runner退出码0。
注意 MLP 仍有速度和上一条指令等共同输入，只是不使用多帧记录窗口。

A 的实际训练会话为 `sf-v1-A-20260907-r2`；`sf-v1-A-verify`、`sf-v1-A-finalcheck` 仅重载、验证及回放，没有重训已早停的模型。
早期配置加载/TF32检查失败，以及一次编辑运行中shell脚本导致的收尾错误，均保留在追加日志中；最后恢复验证退出0。
后续不再修改正在执行的shell脚本。

初轮预热后shadow在共享GPU上测得快分支p95约40.3ms，慢分支p95约29.5ms；不含相机/通信，最终复测见文末。
15Hz周期内完成，但18个第一目标槽位已过期，执行器丢弃这些前缀，只保留原目标时间仍有效的第二槽位，绝不把第一个残差错移到第二槽位。
58次快更新均有至少一个有效槽位；这不代表实机控制频率已验收。

## B/C 特征接入

B 继续冻结原有 `MaskACTPolicy` 与其双视角 U-Net，慢策略仍按原流程输入双RGB和双语义颜色图。
快分支在最新双RGB特征之外，接入直接从原U-Net预测概率计算的2x3空间soft map、各类面积/质心/置信度/可见性和object-region、object-tool质心关系。
前视含背景7类，侧视含背景5类，自适应各视角类别表，合计新增138维，快特征总维度6282。
关系附带validity，不可见时的零占位不会被标记为有效距离；概率损坏时只清空辅助描述，RGB仍可使用。

C 的快RGB特征仍为最新双视角6144维，另附原QToken模型三个512维encoder-query隐藏状态及三个数值有效标志，合计7683维。
通过原有metric-head的forward pre-hook读取隐藏状态，不更改原ACT forward或重新学习query。这里的validity表示token存在且数值有效，不表示物体一定可见。
token来自慢计划的观察，必须按已生效plan_id读取并结合plan age使用；新计划未完成时不得提前输入其token。它们是慢语义上下文，不是15Hz更新的语义状态。
最新视觉仍由快分支独立编码。缺失/长度变化/NaN token有显式槽位掩码，不拿无效token冒充有效特征。

## 阶段 B 验收

会话 `sf-v1-B-20260907-r1`，27 epochs 早停，最佳零起始 epoch18；初次启动的旧脚本导入路径错误已修复，未因此重训或丢弃有效缓存。
归一化 MAE 从自身 UNET 基线0.0682523降至0.0429986；原生单位 MAE 从1.47551降至0.908285。
残差 RMS=1.73030，饱和率约0.104%，逐关节分量目标可达率92.52%。
在线/离线窗口最大差1.19e-6；缓存与新鲜输入在多个时刻重新核对通过。
53项回归测试通过，源 checkpoint SHA256未变，best/last可加载，runner退出0，完整结果在B目录的 `audit.json`。

**B尚未达到共享GPU下的15Hz目标。** 初轮预热后shadow快分支p95约77.8ms、慢分支约118ms；58次快更新只有14次尚有未过期槽位，44次全部过期被拒绝。最终复测见文末。
这是已测得的性能限制，不是可用的实机实时性证明。已有过期保护会退回原计划，但不能因此声称高频修正有效。
几何描述已做等价的按类别向量化并复核缓存，瓶颈仍包含双U-Net/图像编码及共享GPU调度；后续应优化推理或重新按测得延迟配置采样和目标槽位，不能把旧残差错位应用到更晚动作。

## 加载与时间前提

`gru8/best.pt` 保存残差模块，不复制基础ACT或U-Net；使用时仍需要对应基础checkpoint、原处理器和语义模型包。读取 `config.json`、`action_contract.json` 与缓存身份，不能任意搭配另一套基础权重。
命令行支持 `--slow-stride`、`--fast-stride`、`--slow-delay`、`--fast-delay`、`--history` 和 `--hidden`。改变它们应使用新的缓存/输出目录，不应把现有checkpoint无条件当作对应配置的模型。
数据只提供一条公共帧时间轴，无法从中验证两个相机真实曝光时间差；双视角不同步保护在显式时间戳输入下经过测试。部署时需要采集真实的相机时间戳，不能假设metadata一致就等于曝光同步。

正式策略输出只在离线shadow中返回数组，没有调用机械臂接口，也未接入实机GUI。部署、安全参数和真实闭环成功率验证均不在本次授权范围。

## 阶段 C 与最终汇总

C 会话 `sf-v1-C-20260908`，21 epochs 早停，最佳零起始 epoch12；smoke/恢复、best/last重载和53项回归测试通过，runner退出0。
归一化 MAE 从自身 QToken 基线0.0694966降至0.0437435；原生单位 MAE 从1.50451降至0.931731。
残差 RMS=1.70964，饱和率约0.0237%，逐关节分量目标可达率92.75%；在线/离线窗口最大差1.37e-6。
最新双RGB特征及对应已生效计划的原query隐藏状态，在多个时刻重放核对通过，最大差约6.3e-5。
初轮预热后shadow快分支p95约47.2ms、慢分支约102.8ms；58次快更新均保留了有效槽位，18个过期前缀被跳过。仍不包含真实相机/通信开销，最终复测见文末。

| 离线实验 | 自身基础策略归一化MAE | 修正后归一化MAE | 修正后原生单位MAE |
| --- | ---: | ---: | ---: |
| A + MLP | 0.0589675 | 0.0456694 | 0.973755 |
| A + GRU8 | 0.0589675 | 0.0444937 | 0.946683 |
| B + GRU8 | 0.0682523 | 0.0429986 | 0.908285 |
| C + GRU8 | 0.0694966 | 0.0437435 | 0.931731 |

三阶段均已完成本任务要求的实现、正式离线训练、恢复/重载和shadow保护验证。三个训练队列已退出，没有遗留残差训练进程。
各目录 `audit.json` 汇总回归测试、源checkpoint未变证明、多个时间点的缓存一致性、训练/重载结果与shadow时序指标。
上述差异不能证明B/C的辅助特征有因果贡献，更不能代替实机成功率；特别是B未达到当前共享GPU上的15Hz目标。

可在独立tmux中重复完整验收，不启动训练，例如：

```bash
tmux new-session -d -s sf-v1-C-audit-user -c /data/qihan/lerobot \
  'PYTHON=/home/qihan/miniconda3/envs/lerobot/bin/python bash run_slow_fast_residual_stage.sh C audit'
```

相同命令将C改成A或B即可。已训练模型位于各阶段 `gru8/best.pt` 与 `last.pt`；A的MLP位于 `mlp/best.pt`。
没有修改基础ACT、原语义训练器或旧实验权重；没有删除原有未跟踪文档/图片，没有同步GitHub，也没有操作飞书机器人服务。

## 最终调度修复与复验（2026-09-08）

最终检查发现新残差更新会覆盖旧更新中仍待执行的当前槽位。执行器现按目标槽位合并，保留不冲突的待执行残差，并分别记录每个槽位的原观察时间，避免新更新错误延长旧残差寿命；计划切换则全部清空。
新增两项针对覆盖和错误续期的回归测试。该修改不改变训练输入、目标或权重，不需要重训。

在独立会话 `sf-v1-A-runtime-final`、`sf-v1-B-runtime-final`、`sf-v1-C-runtime-final` 串行重跑shadow和完整audit，三组均为55项测试通过，best/last重载、源权重不变、缓存/在线一致性均通过。全部会话正常结束。每组117条返回指令均记录基础动作与实际施加残差，而不只统计收到更新的次数。

| 最终复测 | 快分支p50/p95 (ms) | 慢分支p95 (ms) | 有效更新/尝试 | 实际非零修正指令/返回指令 | 快分支超过66.7ms比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 44.8 / 51.4 | 37.7 | 57/58 | 62/117 | 1.72% |
| B | 62.1 / 71.2 | 101.7 | 40/58 | 37/117 | 31.03% |
| C | 37.1 / 50.2 | 112.7 | 58/58 | 76/117 | 0% |

这些是共享GPU上episode 0短回放的测量，受其他任务负载影响，不是稳定实时性的统计保证。三组快分支超过默认一帧33.3ms预算的比例分别为84.48%、100%、62.07%；部分前缀被正确丢弃，但说明部署前必须重新匹配延迟与预测目标槽位，不能直接照搬训练默认延迟。B的15Hz目标仍未达成。正式离线误差表保持不变，不将回放非零指令数量解释成成功率。

最终记录为各阶段 `verify.log`、`audit.json`、`regression.log` 和 `gru8/shadow_result.json`。可将上述命令中的 `audit` 改为 `verify`，重新执行shadow与完整验收，不启动训练。

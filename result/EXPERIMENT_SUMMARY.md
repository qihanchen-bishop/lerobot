# LeRobot 全部策略实验总览

生成日期：2026-08-25  
数据范围：`eval/object3color2`、`eval/eval20`、`eval/newsetup`

## 1. 总结

`eval/` 下目前有 **842 条有效评估记录**，全部同时包含 `evaluation_metadata.json` 和 `evaluation_result.json`，15 份 `eval_results.jsonl` 与对应 episode 文件数量完全一致。另外有 2 个已建立但从未产生评估数据的实验目录：`1B` 和 `2A`。

按完整的 3 个任务、3x4 网格协议计算：

- **7 组完整实验，共 792 条记录。**
- **8 组部分测试或中途停止实验，共 50 条记录。**
- **2 组未开始实验，0 条记录。**

当前最可靠的总体结论：

- ACT 是唯一在三代评估中都稳定完成完整协议的策略，也是目前最可靠的基线。
- 旧 setup 的大样本 ACT 为 `272/360 (75.6%)`；`eval20` 的 ACT 为 `58/72 (80.6%)`；newsetup ACT 降为 `47/72 (65.3%)`。
- `eval20` 中 ACT、SEM-1、DP、SEM-2 依次为 `80.6%`、`72.2%`、`68.1%`、`62.5%`。
- newsetup 中 ACT 为 `65.3%`，DP 为 `54.2%`。两者都低于旧 setup，说明 newsetup 的数据、视角或任务分布没有自然带来性能提升。
- SEM-1 保留 RGB 后优于只使用语义图的 SEM-2，差距为 9.7 个百分点。当前证据支持“语义辅助 RGB”，不支持“用语义替代 RGB”。
- newsetup 的 SEM-1、SEM-1-V2 受前视相机采集模式异常影响；SSACT-3 测试又关闭了自适应步长和视觉伺服。它们不能用于完整方法排名。
- 所有完整实验都存在明显位置敏感性，尤其是 `y=0`。newsetup DP 在该行仅为 `6/24 (25.0%)`。

## 2. 三代实验协议

| 实验代次 | 时间 | 训练数据/模型代次 | 评估协议 | 主要目的 |
|---|---|---|---|---|
| `object3color2` | 2026-08-04 至 08-07 | `newdata_3object` | 每任务 12 格，每格 10 次；完整协议 360 次 | 建立大样本 ACT 基线，早期筛选 DP 和 Mask-ACT |
| `eval20` | 2026-08-10 至 08-12 | 仍以 `newdata_3object` 系列模型为主 | 每任务 12 格，每格 2 次；完整协议 72 次 | 降低测试负担，统一比较 ACT、DP、SEM-1、SEM-2 |
| `newsetup` | 2026-08-24 至 08-25 | `bettersetup` 重采数据并重训 | 每任务 12 格，每格 2 次；完整协议 72 次 | 比较新布置下 ACT、DP 和新语义策略 |

`object3color2` 与 `eval20` 的完整协议重复次数不同，因此后者适合快速横向比较，前者更适合估计 ACT 的稳定分布。`newsetup` 更换了训练数据和实际布置，不能把它与旧 setup 的差值只归因于策略架构。

## 3. 完成状态总表

“完成比例”以各代完整三任务协议为分母：`object3color2` 为 360 次，`eval20` 和 `newsetup` 为 72 次。

| 实验 | 记录数/完整协议 | 测试内容 | 成功 | 状态 | 建议 |
|---|---:|---|---:|---|---|
| `object3color2/act` | 360/360 | 3 任务，12 格，每格 10 次 | 272/360 (75.6%) | **完成** | 保留为最可靠旧基线 |
| `object3color2/diffusion` | 10/360 | cube，仅 `(0,0)` 10 次 | 1/10 (10.0%) | 停止 | 早期失败版本，归档，不续跑 |
| `object3color2/mask_act/1A` | 5/360 | cube，仅 `(0,0)` 5 次 | 1/5 (20.0%) | 停止 | 早期筛选失败，归档 |
| `object3color2/mask_act/1B` | 0/360 | 无 | 无 | **未开始** | 当前没有继续价值 |
| `object3color2/mask_act/2A` | 0/360 | 无 | 无 | **未开始** | 当前没有继续价值 |
| `object3color2/mask_act/2B` | 3/360 | cube，仅 `(0,0)` 3 次 | 0/3 (0.0%) | 停止 | 早期筛选失败，归档 |
| `eval20/act` | 72/72 | 3 任务，12 格，每格 2 次 | 58/72 (80.6%) | **完成** | 旧 setup 主基线 |
| `eval20/diffusion` | 72/72 | 3 任务，12 格，每格 2 次 | 49/72 (68.1%) | **完成** | 旧 setup DP 基线 |
| `eval20/mask_act/SEM-1` | 72/72 | 3 任务，12 格，每格 2 次 | 52/72 (72.2%) | **完成** | 最有效的旧语义策略 |
| `eval20/mask_act/SEM-2` | 72/72 | 3 任务，12 格，每格 2 次 | 45/72 (62.5%) | **完成** | 证明纯语义输入不足 |
| `eval20/mask_act/ASEM-1` | 4/72 | cube，`(0,0)`、`(1,0)` 各 2 次 | 0/4 (0.0%) | 停止 | 不建议按当前版本续跑 |
| `eval20/mask_act/SSACT-1` | 1/72 | cube，`(0,0)` 1 次 | 0/1 (0.0%) | 停止 | 仅用于诊断运行机制 |
| `newsetup/act` | 72/72 | 3 任务，12 格，每格 2 次 | 47/72 (65.3%) | **完成** | newsetup 主基线 |
| `newsetup/diffusion` | 72/72 | 3 任务，12 格，每格 2 次 | 39/72 (54.2%) | **完成** | newsetup DP 基线 |
| `newsetup/mask_act/SEM-1` | 12/72 | cube，12 格，每格 1 次 | 3/12 (25.0%) | 部分完成但条件异常 | 修复输入后从头复测 |
| `newsetup/mask_act/SEM-1-V2` | 12/72 | cube，12 格，每格 1 次 | 0/12 (0.0%) | 部分完成但条件异常 | 修复输入后从头复测 |
| `newsetup/mask_act/SSACT-3` | 3/72 | cube，3 个位置各 1 次 | 0/3 (0.0%) | 部分完成且功能未启用 | 启用完整控制后重测 |

## 4. 完整实验结果

### 4.1 总体排名

| 实验 | 成功率 | Wilson 95% 区间 | 可比范围 |
|---|---:|---:|---|
| `eval20/act` | **58/72 (80.6%)** | 70.0%-88.0% | 与 eval20 其他完整实验直接比较 |
| `object3color2/act` | **272/360 (75.6%)** | 70.9%-79.7% | 大样本旧 ACT 基线 |
| `eval20/SEM-1` | **52/72 (72.2%)** | 61.0%-81.2% | 与 eval20 ACT/DP/SEM-2 直接比较 |
| `eval20/diffusion` | **49/72 (68.1%)** | 56.6%-77.7% | 与 eval20 直接比较 |
| `newsetup/act` | **47/72 (65.3%)** | 53.8%-75.2% | 与 newsetup DP 直接比较 |
| `eval20/SEM-2` | **45/72 (62.5%)** | 51.0%-72.8% | 与 eval20 直接比较 |
| `newsetup/diffusion` | **39/72 (54.2%)** | 42.7%-65.2% | 与 newsetup ACT 直接比较 |

不能按这张表直接做跨代统一排名，因为训练数据、物理 setup、评估重复数和自动判定版本不同。可信的横向比较单元是 `eval20` 内部和 `newsetup` 内部。

### 4.2 分任务结果

| 实验 | cube | paperball | screw | 总体 |
|---|---:|---:|---:|---:|
| `object3color2/act` | 100/120 (83.3%) | 84/120 (70.0%) | 88/120 (73.3%) | 272/360 (75.6%) |
| `eval20/act` | 18/24 (75.0%) | 18/24 (75.0%) | 22/24 (91.7%) | 58/72 (80.6%) |
| `eval20/diffusion` | 16/24 (66.7%) | 14/24 (58.3%) | 19/24 (79.2%) | 49/72 (68.1%) |
| `eval20/SEM-1` | 19/24 (79.2%) | 15/24 (62.5%) | 18/24 (75.0%) | 52/72 (72.2%) |
| `eval20/SEM-2` | 16/24 (66.7%) | 15/24 (62.5%) | 14/24 (58.3%) | 45/72 (62.5%) |
| `newsetup/act` | 17/24 (70.8%) | 16/24 (66.7%) | 14/24 (58.3%) | 47/72 (65.3%) |
| `newsetup/diffusion` | 12/24 (50.0%) | 16/24 (66.7%) | 11/24 (45.8%) | 39/72 (54.2%) |

主要观察：

- `eval20` 中 SEM-1 的 cube 比 ACT 高 4.2 pp，但 paperball 低 12.5 pp、screw 低 16.7 pp，语义收益明显依赖任务。
- SEM-1 比 SEM-2 总体高 9.7 pp，screw 高 16.7 pp。去掉原始 RGB 后，模型更容易受分割误差和遮挡影响。
- newsetup 中 ACT 比 DP 总体高 11.1 pp，其中 cube 高 20.8 pp、screw 高 12.5 pp；paperball 完全相同。
- newsetup ACT 的 screw 从旧 `eval20` 的 91.7% 降到 58.3%，这是当前最需要复核的数据/setup 变化。

### 4.3 阶段到达率

| 实验 | 曾暴露物体 | 与遮挡物分离 | 最终成功 |
|---|---:|---:|---:|
| `object3color2/act` | 353/360 (98.1%) | 323/360 (89.7%) | 272/360 (75.6%) |
| `eval20/act` | 68/72 (94.4%) | 66/72 (91.7%) | 58/72 (80.6%) |
| `eval20/diffusion` | 62/72 (86.1%) | 61/72 (84.7%) | 49/72 (68.1%) |
| `eval20/SEM-1` | 68/72 (94.4%) | 60/72 (83.3%) | 52/72 (72.2%) |
| `eval20/SEM-2` | 65/72 (90.3%) | 61/72 (84.7%) | 45/72 (62.5%) |
| `newsetup/act` | 67/72 (93.1%) | 64/72 (88.9%) | 47/72 (65.3%) |
| `newsetup/diffusion` | 54/72 (75.0%) | 52/72 (72.2%) | 39/72 (54.2%) |

旧 ACT 和 newsetup ACT 都有较高的暴露、分离能力，损失主要发生在送入目标区、恢复布料和最终状态保持。newsetup DP 从暴露阶段就已经明显落后。

### 4.4 位置敏感性

三个任务合并后的网格行成功率：

| 实验 | y=0 | y=1 | y=2 |
|---|---:|---:|---:|
| `object3color2/act` | 76/120 (63.3%) | 98/120 (81.7%) | 98/120 (81.7%) |
| `eval20/act` | 13/24 (54.2%) | 23/24 (95.8%) | 22/24 (91.7%) |
| `eval20/diffusion` | 13/24 (54.2%) | 17/24 (70.8%) | 19/24 (79.2%) |
| `eval20/SEM-1` | 15/24 (62.5%) | 17/24 (70.8%) | 20/24 (83.3%) |
| `eval20/SEM-2` | 5/24 (20.8%) | 20/24 (83.3%) | 20/24 (83.3%) |
| `newsetup/act` | 13/24 (54.2%) | 14/24 (58.3%) | 20/24 (83.3%) |
| `newsetup/diffusion` | 6/24 (25.0%) | 14/24 (58.3%) | 19/24 (79.2%) |

`y=0` 是跨模型持续存在的薄弱区域。SEM-2 和 newsetup DP 对该区域尤其敏感，说明缺少原始 RGB 或闭环更新较慢时，较差视角、遮挡和接触几何会被进一步放大。

## 5. 未完成实验分析

### 5.1 object3color2 早期筛选

- 初版 DP 仅在 cube `(0,0)` 测试 10 次，成功 1 次。之后 `eval20` 使用同一个路径字符串 `DPF_3object/100000/pretrained_model` 得到 68.1%，但期间模型曾重新训练并可能覆盖原目录。
- 由于历史 metadata 没有保存 checkpoint 文件哈希，初版 DP 与 eval20 DP 是否为同一组权重无法从文件中证明，不能把提升全部解释成推理参数优化。
- 1A 在 cube `(0,0)` 为 1/5；2B 为 0/3。二者本地 checkpoint 目录目前已经不存在，无法严格复现实验。
- 1B、2A 只有空目录，没有任何 episode、metadata 或结果，应标记为“未开始”，不是 0% 成功率。

### 5.2 ASEM-1

ASEM-1 让动作损失以预热、限幅和冲突投影方式反向监督分割网络。现有 4 次 cube 测试全部失败，且都在最终时物体不可见。

样本只覆盖 `(0,0)`、`(1,0)` 两个位置，不能估计完整成功率。但连续 4 次都没有完成暴露阶段，已足以说明当前 checkpoint 不值得直接补齐 72 次；应先检查分割是否被动作梯度破坏。

### 5.3 SSACT-1

SSACT-1 只测试 1 次，失败原因为“已分离但未到目标区”。该次测试确实启用了语义动力学、五阶段预测、自适应 1-4 步执行和 active 视觉伺服，但运行日志暴露出严重问题：

- 438 次重规划中，阶段为 `uncover` 40 次、`expose` 398 次，只切换 1 次，未进入 transport/restore/done。
- 408/438 次仍选择执行 4 步，真正缩短为 1-2 步的只有 30 次。
- QP/伺服只在 20/438 次实际修正动作。
- `calibration_status=uncalibrated`、`clf_certified=false`，所以不能声称具有经过标定或理论认证的收敛保证。
- 名义 episode 为 25 秒，但该次实际运行 63.59 秒，说明阶段闭锁还会破坏预期测试时长。

这条数据只适合证明运行链路曾启动，不适合评价算法效果。

### 5.4 newsetup SEM-1 与 SEM-1-V2

SEM-1 原始结果为 3/12，SEM-1-V2 为 0/12。两组都只完成 cube 每格 1 次，尚缺每格第 2 次以及 paperball、screw 共 48 次。

更重要的是，这 24 条数据采集时前视 RealSense 曾工作在 `640x480`，随后被强制缩放到策略所需的 `640x360`，画面几何和训练输入不一致。因此不应继续在这些记录后补第 2 次，而应在原生 `640x360@30` 下从 cube 第 1 次重新开始。

SEM-1 仍有 11/12 次暴露、10/12 次分离，但最终只有 3 次成功，主要损失在布料恢复。SEM-1-V2 只有 7/12 次暴露、6/12 次分离，说明概率图特征调制在当前遮挡和输入异常下更早产生了负作用。

### 5.5 newsetup SSACT-3

SSACT-3 只测试 3 次并全部失败。训练模型包含 expose/separate/transport/restore/done 阶段、事件/进度/转移/关系头和 StageFiLM，但实际评估配置是：

```text
stage_only=true
execution_steps=4（固定）
adaptive_horizon_applied=false
servo_mode=off
ssact_clf_certified=false
ssact_learned_hazard=false
```

阶段输出也出现高置信度锁定：三次分别有 152/176、183/183、166/183 个重规划周期停在 `separate`。所以现有结果只能评价“固定 4 步的阶段条件化策略”，不能评价自适应执行长度或 CLF/QP 视觉伺服。

## 6. 策略定义差异

| 策略 | 输入和动作模型 | 语义梯度/控制特点 |
|---|---|---|
| ACT | front RGB + side RGB + 关节状态，Transformer 输出 action chunk | 无语义输入，当前最稳定基线 |
| DP | front RGB + side RGB + 关节状态，扩散模型输出动作序列 | DDIM 去噪，异步重规划，推理更慢 |
| 1A | RGB 经 U-Net 得到预测 mask，再由 ACT 控制 | 分割损失训练 U-Net，动作损失不训练 mask 路径 |
| 1B | 与 1A 相同 | 动作损失也反向训练 U-Net；未测试 |
| 2A | 预测 mask + pooled RGB latent 输入 ACT | 动作损失不训练 mask 路径；未测试 |
| 2B | 预测 mask + pooled RGB latent 输入 ACT | 动作损失训练 mask 路径，pooled RGB 路径 detach |
| SEM-1 | 每视角五类 soft mask 合成语义 RGB 图，同时保留原始 RGB | 语义图作为额外图像 token，分割主要由语义监督 |
| SEM-2 | 只输入语义图，不输入原始 RGB | 最依赖分割质量和遮挡完整性 |
| ASEM-1 | 与 SEM-1 相同 | 动作梯度经过预热、限幅和冲突投影后训练分割网络 |
| SSACT-1 | SEM-1 + 语义动力学 + 五阶段历史模型 | 推理时可启用自适应执行长度和 CLF/QP 残差修正 |
| SEM-1-V2 | 保留五类 soft probability，经 adapter 形成 RGB ResNet 特征残差 | 不增加 Transformer 图像 token；动作梯度不进入分割概率 |
| SSACT-3 | SEM-1-V2 + 阶段/事件/进度/转移/关系头 | 阶段条件化 relation attention + FiLM；当前测试未启用伺服和可变步长 |

当前语义类别为 `background`、`occluder`、`object`、`region`、`tool`，没有机械臂类别。机械臂遮挡物体时，模型难以区分“目标被机械臂暂时挡住”和“目标真实丢失”，这是 SEM-1-V2 和阶段模型容易被语义误导的重要结构性问题。

## 7. 推理配置差异

所有主实验均为 30 FPS、3x4 网格、双臂、front+side、无历史 action fusion。除首条旧 ACT 使用 `latest_nonblocking` 外，其余 841 条均使用 `wait_new_frame`。

| 策略族 | 模型预测长度 | 每次执行/重规划 | 执行方式 | AMP | 其他 |
|---|---:|---:|---|---|---|
| ACT | 60 | 60 | 同步 | 关 | 一次预测基本执行完整 chunk |
| SEM-1/SEM-2/ASEM-1 | 60 | 60 | 同步 | 关 | 与 ACT 保持相同动作节奏 |
| DP | 64 | 24 | 异步 | 开 | DDIM 16 步，warmup 1 帧，历史融合关闭 |
| SSACT-1 | 60 | 动态 1-4 | 同步 | 开 | active servo，最大动作残差 0.015 |
| SSACT-3 当前测试 | 60 | 固定 4 | 同步 | 开 | stage-only，servo 关闭 |

DP checkpoint 的训练配置记录 `horizon=64`、`n_obs_steps=1`、训练噪声步数 100；评估时使用 DDIM 16 步去噪并执行前 24 步。ACT 和 Mask-ACT 均预测并执行 60 步。这个比较反映的是各策略当前完整控制栈，不是只替换网络后、执行节奏完全相同的纯架构对比。

## 8. 数据版本与分割配置

| 项目 | 旧 setup | newsetup |
|---|---|---|
| 训练数据 | `newdata_3object` | `bettersetup` |
| ACT 数据视图 | `outputs/train/dataset_views/act_newdata_3object_front_side` | `outputs/train/dataset_views/ACT-bettersetup-front-side` |
| DP 数据视图 | `outputs/train/dataset_views/DP-1F-Full` | `outputs/train/dataset_views/DP-1F-Full-bettersetup` |
| 评估分割预览/自动指标 | `mycode/tool/seg_v1/best.pt` | `mycode/tool/seg_v2/best.pt` |
| 策略图像 | front + side | front + side |
| 语义类别 | 五类 | 五类，仍无机械臂类别 |

外部分割模型路径主要用于 GUI 预览和自动评估；纯 ACT/DP 不把该分割结果作为策略输入。Mask-ACT 的策略语义由 checkpoint 内部 U-Net 产生，不能把 GUI 预览模型与策略内部语义网络混为一谈。

## 9. 配置和可追溯性问题

- 历史 metadata 保存了 checkpoint 路径，但没有保存 checkpoint 哈希、Git commit、相机 serial、实际 `/dev/videoX`、V4L2 source mode、原始分辨率或缩放方式。
- 同一路径可能被重训权重覆盖。初版 DP 与 eval20 DP 都记录 `DPF_3object/100000/pretrained_model`，但不能证明文件内容相同。
- 1A 和 2B 的历史 checkpoint 目录当前缺失，已有结果可以分析，但无法严格复现。
- 当前 `ACT_NEWSETUP.json`、`DP_NEWSETUP.json`、`MASK_SEM1_NEWSETUP.json` 的默认保存目录是 `eval/bettersetup`，而现有实验实际位于 `eval/newsetup`。直接启动会把后续结果拆到另一个目录。
- 当前 `MASK_SEM1.json`、`MASK_SEM2.json` 默认保存到 `eval/object3color2`，但完整语义结果实际位于 `eval20`。历史实际设置应以每条 `evaluation_metadata.json` 为准。
- `object3color2/act` 的 screw 自动阈值、最近帧投票和侧视回退逻辑在采集中途修改过，因此自动失败原因跨整批不完全同口径。
- 全部 842 条记录中只有 4 条人工结果与自动最终判定不一致，整体一致率约 99.5%。最终成功率仍以人工 `result` 为主。

后续每条评估建议额外固化：checkpoint SHA256、Git commit、训练数据版本、分割模型 SHA256、相机 serial、实际 V4L2 profile、输入 resize/crop 参数和完整运行配置快照。

## 10. 当前判断与下一步

### 可以直接采用的结论

1. ACT 仍应作为主基线和部署默认策略。
2. 在旧 setup 中，SEM-1 有一定价值，但总体没有超过 ACT；SEM-2 明显证明只依赖语义不够。
3. DP 在两套完整 72 次协议中都弱于 ACT，newsetup 差距更明显。
4. `y=0` 和最终布料恢复是跨策略的主要数据缺口。
5. 机械臂未分割、目标被遮挡时语义失真，是后续语义策略必须解决的问题。

### 应该补做

1. 固定原生 `640x360@30` 和相机 profile 后，newsetup SEM-1、SEM-1-V2 从 cube 第 1 次重新测试，每格 2 次。
2. 加入机械臂/遮挡类别或不确定性门控后，再测试 SEM-1-V2；旧的 0/12 不应作为最终结论。
3. SSACT-3 先确认界面和 runtime log 中自适应步长、servo/CLF-QP 均真实开启，再做 3-5 个诊断 episode；通过后才扩展到完整网格。
4. 针对 ACT 的 `y=0`、screw 和布料恢复阶段补采数据，并独立报告阶段成功率。

### 不建议继续补齐

1. object3color2 的初版 DP、1A、2B 应作为早期失败探索归档。
2. 1B、2A 没有测试数据，且已有 SEM 系列提供了更清晰的语义路线，不建议重新开启。
3. ASEM-1 当前 4 次全部失败，应先修复训练机制，而不是直接补满 72 次。
4. SSACT-1 当前存在阶段锁定、未标定 CLF 和超时问题，应由 SSACT-3 或后续修正版替代。

## 11. 相关报告

- `result/act/REPORT.md`：`object3color2/act` 的 360 次大样本详细报告和图表。
- `eval/newsetup/REPORT.md`：newsetup 的阶段表现、失败模式和位置分析。
- 本文件：全部实验的完成度、配置差异和跨代总结。

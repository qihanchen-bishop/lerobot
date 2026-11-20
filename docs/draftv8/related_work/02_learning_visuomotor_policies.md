# Learning Visuomotor Policies：文献检索与叙述设计

检索日期：2026-11-19  
用途：为 Related Work 第二点 **Learning Visuomotor Policies** 建立方法分类，并吸收原 **Distribution Shift and Task-Level Analysis** 中与策略学习相关的内容。

## 1. 分类原则

本段不再讨论“双臂任务有哪些”，而是比较 visuomotor policies 的学习范式。分类首先依据策略获得监督的方式，而不是网络结构：

1. **交互式学习**：利用机器人自主交互产生的奖励或预测信号，包括 model-free RL 和 learned-model planning；
2. **示范式学习**：利用专家状态—动作轨迹进行 imitation learning；
3. **混合式学习**：先用示范建立基础行为，再以 RL、残差策略或交互纠正提升性能。

flat policy 与 hierarchical policy 是与上述分类正交的“策略组织方式”：RL、IL 或混合方法都可以采用单一策略，也可以组合 primitives、subgoals 和 high-level selectors。

这种分类比原稿的“端到端策略—层级策略”更完整，因为它能够先比较不同学习信号的优势与代价，再讨论序列任务如何组织。

## 2. 交互式学习：奖励优化与学习模型规划

### 2.1 基于奖励的策略学习

| 优先级 | 论文 | 方法要点 | 主要优势 | 主要限制 | 当前 BibTeX |
|---|---|---|---|---|---|
| 核心 | [End-to-End Training of Deep Visuomotor Policies](https://www.jmlr.org/papers/v17/15-522.html), Levine et al., JMLR 2016 | Guided policy search 将 trajectory-centric RL 产生的监督用于训练从图像到力矩的 CNN policy | 证明视觉与控制可以端到端联合优化，并避免单独设计视觉状态估计 | 依赖局部控制器、轨迹优化和真实机器人训练条件，不易直接扩展到任意长任务 | 缺失，建议键 `levine2016visuomotor` |
| 核心 | [Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation](https://proceedings.mlr.press/v87/kalashnikov18a.html), Kalashnikov et al., CoRL 2018 | QT-Opt 以大规模真实抓取尝试训练闭环视觉 Q-function | 能围绕任务奖励主动发现重抓、探测、扰动恢复等策略，而不受专家动作上限约束 | 使用超过 58 万次真实抓取尝试，反映出真实机器人 RL 的数据、复位和安全成本 | 缺失，建议键 `kalashnikov2018qtopt` |
| 核心 / 领域 | [FF-SRL](https://doi.org/10.1109/IROS58592.2024.10801658), Dall'Alba et al., IROS 2024 | 构建 GPU surgical RL environment 以加速可形变组织任务的重复训练 | 直接说明外科 RL 需要专门的高吞吐仿真环境 | 论文动机明确指出大量真实手术系统试验通常不安全且不可行；仿真真实性又成为额外条件 | 已有 `dallalba2024ffsrl` |

基于奖励的方法能够直接优化任务成功，并通过探索获得示范中没有的纠正或恢复行为；代价是需要可计算奖励、大量交互和可靠复位，真实接触任务中的探索还可能缓慢或不安全。这个限制在医疗手术中尤其关键：真实组织和手术系统不能作为可反复试错的训练场，而可靠模拟又需要处理组织形变、接触和视觉变化。因此，本文选择 imitation learning 并不只是算法偏好，而是与可获得的专家示范和缺少安全 RL 交互环境相匹配。

### 2.2 基于学习模型的视觉规划

| 优先级 | 论文 | 方法要点 | 主要优势 | 主要限制 | 当前 BibTeX |
|---|---|---|---|---|---|
| 核心 | [Deep Visual Foresight for Planning Robot Motion](https://doi.org/10.1109/ICRA.2017.7989324), Finn and Levine, ICRA 2017 | 从无标签机器人视频学习 action-conditioned prediction model，再以 model-predictive control 选择动作 | 自主交互数据不需要动作标签或人工奖励塑形，并能对新物体重新规划 | 视频预测误差会进入规划；接触、遮挡和可形变物体使长时域预测尤其困难 | 缺失，建议键 `finn2017foresight` |

这一路线把学习对象从 policy 转为 environment dynamics，再通过规划产生动作。它提高了目标复用和可解释性，但性能受模型准确性和规划时域限制，因此不宜与直接策略学习混为一类。

## 3. 示范式学习：行为克隆与动作序列建模

| 优先级 | 论文 | 方法要点 | 主要优势 | 主要限制 | 当前 BibTeX |
|---|---|---|---|---|---|
| 核心 | [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://www.roboticsproceedings.org/rss19/p016.html), Zhao et al., RSS 2023 | ACT 用 Transformer 预测 action chunks，并以 temporal ensembling 执行 | 以较少真实示范学习精细双臂任务，降低逐步动作预测的有效时域 | 依赖示范质量和覆盖；动作损失不直接约束策略采用何种任务状态作决策 | 已有 `zhao2023act` |
| 核心 | [Diffusion Policy](https://www.roboticsproceedings.org/rss19/p026.html), Chi et al., RSS 2023 | 条件去噪建模多峰动作轨迹，并采用 receding-horizon control | 能表示多模态和高维连续动作，适合接触丰富的 visuomotor behavior | 推理和闭环执行仍受离线数据覆盖限制 | 已有 `chi2023diffusion` |
| 核心 | [Behavior Generation with Latent Actions](https://proceedings.mlr.press/v235/lee24y.html), Lee et al., ICML 2024 | VQ-BeT 用分层向量量化将连续 action chunks 变为离散 latent actions | 相比 Diffusion Policy 保留多模态生成能力并实现约 5 倍推理加速 | 仍观察到高误差与 mode collapse 失败；离散 codebook 不能消除闭环累积误差 | 已有 `lee2024vqbet` |
| 核心 | [Consistency Policy](https://www.roboticsproceedings.org/rss20/p071.html), Prasad et al., RSS 2024 | 将训练好的 Diffusion Policy 蒸馏为少步 consistency sampler | 在资源受限平台上将真实任务推理延迟降低约 9 倍 | 牺牲部分动作多模态，训练稳定性较弱，且教师蒸馏增加训练时间 | 已有 `prasad2024consistency` |
| 核心 | [BAKU](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ff887781480973bd3cb6026feb378d1e-Abstract-Conference.html), Haldar et al., NeurIPS 2024 | 将多感官/语言条件、Transformer trunk、ACT 式 action chunking 和可替换 action head 组合为多任务策略 | 相比 MT-ACT 改善跨任务数据共享，并在模拟与真实任务上提高成功率 | 对部分精细操作仍较弱；作者指出尚未解决多技能串联 | 已有 `haldar2024baku` |
| 核心 | [RDT-1B](https://proceedings.iclr.cc/paper_files/paper/2025/hash/49f80e4d2471ad4f2edf4f5f1ab62339-Abstract-Conference.html), Liu et al., ICLR 2025 | 用统一物理动作空间和 1.2B diffusion transformer 进行跨机器人预训练，再适配语言条件双臂任务 | 将 Diffusion Policy 的多模态动作建模扩展到大规模跨 embodiment 迁移 | 预训练使用超过百万条轨迹及 48 张 H100，数据与算力成本高 | 已有 `liu2025rdt` |
| 补充 | [REMAC](https://arxiv.org/abs/2601.20130), Wang et al., ICLR 2026 | 用 masked action chunking 学习修正异步推理下的 chunk 内感知—执行失配 | 不增加部署延迟地提高实时执行的连续性与鲁棒性 | 解决的是异步执行失配，不提供示范覆盖外的专家纠正 | 已有 `wang2026remac` |
| 补充 | [Legato](https://www.roboticsproceedings.org/rss22/p058.html), Liu et al., RSS 2026 | 在训练中学习 action-chunk continuation，减少边界跳变和虚假 mode switching | 改善轨迹平滑性与完成时间 | 聚焦 chunk 衔接，不能单独解决长序列中的任务级错误恢复 | 已有 `liu2026legato` |
| 可选 | [InterACT](https://proceedings.mlr.press/v270/lee25a.html), Lee et al., CoRL 2024 / PMLR 2025 | 用分层注意力建模双臂关节与视觉输入的相互依赖 | 说明双臂耦合可以在单个 action-chunk policy 内建模 | 改善架构交互建模，不等于显式监督策略保留具体任务状态 | 已有 `lee2025interact` |

ACT 与 Diffusion Policy 之后的工作并不只是在修改视觉表征：VQ-BeT 和 Consistency Policy 分别从动作离散化与蒸馏侧降低生成延迟，BAKU 改善多任务共享，RDT-1B 扩展跨机器人预训练，而 REMAC 和 Legato 直接处理异步 chunk 执行中的失配与不连续。这些改进各有明确边界：它们可能牺牲 mode coverage、增加蒸馏或大规模预训练成本，或仅改善执行平滑性；仍未获得示范状态覆盖之外的纠正信号。

IL 将人的任务知识直接压缩到示范中，通常比从奖励探索更适合真实机器人，也便于学习长序列和接触丰富行为。然而，策略只能学习示范呈现的行为和状态；闭环执行产生的新历史会带来 covariate shift，且较低的离线动作误差不保证策略使用了正确的环境依据。

DAgger 在 learner-visited states 上追加专家标注（`ross2011`），DART 则对示范注入优化噪声以覆盖可能执行误差（`laskey2017dart`）。二者说明 distribution shift 可以从数据覆盖侧缓解，但需要额外交互或改变采集过程；因此该问题应在 IL 分类内部讨论，而不是作为独立 Related Work 小节。

## 4. 混合式学习：用示范初始化、用交互改进

| 优先级 | 论文 | 组合方式 | 主要优势 | 主要限制 | 当前 BibTeX |
|---|---|---|---|---|---|
| 核心 | [AW-Opt](https://proceedings.mlr.press/v164/lu22a.html), Lu et al., CoRL 2021 / PMLR 2022 | 结合 advantage-weighted regression 与 QT-Opt，统一利用 demonstrations 和 heterogeneous offline data | IL 提供有效初始行为，RL 进一步优化任务回报 | 仍依赖奖励、大规模机器人经验和离线数据质量 | 缺失，建议键 `lu2022awopt` |
| 核心 | [From Imitation to Refinement—Residual RL for Precise Assembly](https://arxiv.org/abs/2407.16677), Ankile et al., 2024 | 冻结 chunked BC policy，并以在线 RL 学习逐步 residual corrections | 保留 BC 的长时域计划能力，同时获得精细闭环修正和恢复 | 需要额外 RL rollouts、奖励、仿真训练及 sim-to-real 处理 | 已有 `ankile2024resip` |
| 可选 | [Relay Policy Learning](https://proceedings.mlr.press/v100/gupta20a.html), Gupta et al., CoRL 2019 / PMLR 2020 | 从非结构化示范初始化 goal-conditioned policies，再用 RL 连接长任务 | 同时利用示范初始化和 RL 探索，适合长时程子任务衔接 | 需要目标层级、环境交互和强化学习微调 | 已有 `gupta2020relay` |

混合方法说明 IL 与 RL 的优缺点具有互补性：示范减少无效探索，奖励优化突破示范质量和局部覆盖的限制。但这种改进并非免费，通常引入奖励定义、交互数据、安全探索和复位条件。

## 5. 正交维度：扁平策略与显式层级

无论采用 RL、IL 还是混合学习，序列行为都可以由单个策略产生，也可以显式拆成 primitives 或 subgoals。

| 优先级 | 论文 | 组织方式 | 优势 | 限制 | 当前 BibTeX |
|---|---|---|---|---|---|
| 可选 | [Neural Task Programming](https://arxiv.org/abs/1710.01813), Xu et al., ICRA 2018 | 从视觉示范学习任务层级与可组合程序 | 提高跨任务组合和结构复用 | 需要推断显式程序结构 | 已有 `xu2018ntp` |
| 核心 | [Multistage Cable Routing Through Hierarchical Imitation Learning](https://doi.org/10.1109/TRO.2024.3353075), Luo et al., T-RO 2024 | 固定低层 primitives，由视觉高层策略选择、重试和推进 | 技能复用清晰，高层能够组织恢复和重试 | 依赖预定义 primitive 集与高层 selector | 已有 `luo2024cable` |
| 可选 | [Goal-Conditioned Dual-Action Imitation Learning](https://doi.org/10.1109/TRO.2024.3372778), Kim et al., T-RO 2024 | 在预先划分的子任务中切换反应式局部动作与全局轨迹 | 兼顾大范围移动和接触阶段反应性 | 依赖子任务划分、动作类型标签和专用结构 | 已有 `kim2024dualaction` |

显式层级有利于技能复用、阶段推进和失败恢复，但需要 primitive definitions、skill boundaries、subgoals 或 high-level selection。单一策略避免这些结构性标注和接口，却必须在共同表征中维持整个序列所需的决策信息。

## 6. 正文的一段式论证链

正文建议压缩为以下六步：

1. 以监督来源概括 interaction-based、demonstration-based 和 hybrid learning；
2. 用 guided policy search、QT-Opt 和 Deep Visual Foresight 说明 RL/learned-model control 的自主改进能力及交互代价，再以 surgical RL simulator 说明这种代价在手术环境中更加突出；
3. 用 ACT 和 Diffusion Policy 说明 IL 的真实机器人数据效率与动作序列建模能力；
4. 将 covariate shift 作为 IL 的主要限制，并用 DAgger、DART 说明数据侧解决方式；
5. 用 AW-Opt 或 residual RL 说明混合方法的互补性，再以 cable routing 简短说明显式层级这一正交维度；
6. 定位本文：固定离线示范、单一 action-chunk policy，不依赖奖励交互、动力学规划或 skill selector，研究 task-directed representation supervision。

正文不需要引用表中全部论文。建议保留以下五组：

- reward-based RL 及手术环境限制：`levine2016visuomotor,kalashnikov2018qtopt,dallalba2024ffsrl`；
- learned-model planning：`finn2017foresight`；
- demonstration-based IL：`zhao2023act,chi2023diffusion`；
- coverage 与混合改进：`ross2011,laskey2017dart,lu2022awopt,ankile2024resip`；
- explicit hierarchy：`luo2024cable`。

## 7. 当前最值得保留的研究缺口表述

基于奖励的学习可以通过交互发现纠正行为，学习动力学模型可以支持目标条件规划，而模仿学习能以较少真实示范获得复杂动作序列；它们分别受到交互和奖励成本、模型预测误差以及示范覆盖与闭环偏移的限制。混合式与层级式方法能够提升精度、恢复或技能复用，但进一步引入奖励、在线 rollouts、技能边界或高层选择器。对于可形变组织和手术机器人，真实试错通常不安全且难以重复，构建足够真实的 RL 仿真环境本身也是重要难题。本文因此利用可获得的专家示范，在固定有限数据和单一 action-chunk policy 下研究任务相关表征监督，而不依赖在线试错或专用手术仿真器。

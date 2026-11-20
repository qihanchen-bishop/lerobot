# Task-Relevant Representations for Robot Learning：文献检索与叙述设计

检索日期：2026-11-19  
用途：为 Related Work 第三点 **Task-Relevant Representations for Robot Learning** 建立文献分类，并吸收原 **Distribution Shift and Task-Level Analysis** 中与表示覆盖、信息保留和任务结果相关的内容。

## 1. 本段需要回答什么

这一段的核心不是泛泛说明“表征很重要”，而是区分三个层次：

1. 通用视觉预训练能否提供适合机器人控制的特征？
2. 对象、关键点和语义分割能否把与操作有关的空间结构显式提供给策略？
3. 如何保证动作生成所用的内部表示确实保留了后续决策需要的任务状态？

本文的切入点位于第三层，但方法同时使用了第二层的语义输入。因此正文必须明确区分“输入中存在语义信息”和“联合策略表征可读出特定任务信息”。

## 2. 通用或预训练视觉表征

| 优先级 | 论文 | 方法与结论 | 对本段的作用 | 当前 BibTeX |
|---|---|---|---|---|
| 核心 | [R3M: A Universal Visual Representation for Robot Manipulation](https://proceedings.mlr.press/v205/nair23a.html), Nair et al., CoRL 2022 / PMLR 2023 | 从大规模人类视频学习时序和语言对齐特征，作为冻结视觉模块提高下游数据效率 | 代表“通用 manipulation-relevant pretraining”路线 | 缺失，建议键 `nair2023r3m` |
| 核心 | [On Pre-Training for Visuo-Motor Control](https://proceedings.mlr.press/v202/hansen23c.html), Hansen et al., ICML 2023 | 简单从头训练基线可与多种冻结预训练表征竞争；预训练域与控制域之间存在 domain gap，微调可缓解 | 支持“通用预训练不自动等于下游任务相关” | 缺失，建议键 `hansen2023pretraining` |
| 可选 | [What Makes Pre-Trained Visual Representations Successful for Robust Manipulation?](https://proceedings.mlr.press/v270/burns25a.html), Burns et al., CoRL 2024 / PMLR 2025 | 比较多种预训练模型在视觉偏移下的控制鲁棒性，发现 emergent segmentation ability 与泛化表现相关 | 连接预训练表征、分割能力和 OOD 鲁棒性，但不应被表述为分割足以保证控制 | 缺失，建议键 `burns2025robust` |

**总结判断：** 通用预训练主要提升特征复用和数据效率，但其不变性由预训练数据和目标决定，不一定与当前任务的决策变量一致。

## 3. 对象中心、空间和语义结构化表征

这类方法通过对象 proposal、关键点、3-D object features 或 foundation-model masks，减少策略直接从原始像素发现操作结构的负担。

| 优先级 | 论文 | 表征形式 | 对本段的作用 | 当前 BibTeX |
|---|---|---|---|---|
| 核心 | [Transporter Networks](https://proceedings.mlr.press/v155/zeng21a.html), Zeng et al., CoRL 2020 / PMLR 2021 | 通过局部特征运输表示 pick-and-place 的视觉空间对应 | 代表面向操作几何的结构化视觉表征 | 已有 `zeng2021transporter` |
| 核心 | [VIOLA](https://proceedings.mlr.press/v205/zhu23a.html), Zhu et al., CoRL 2022 / PMLR 2023 | 将预训练检测器的 object proposals 作为先验，并由 Transformer 选择任务相关区域 | 直接支持对象中心先验能够改善有限示范下的视觉模仿 | 已有 `zhu2023viola` |
| 核心 | [GROOT](https://proceedings.mlr.press/v229/zhu23b.html), Zhu et al., CoRL 2023 | 学习对象中心 3-D 表征以跨场景和实例泛化 | 代表语义对象与三维几何结合 | 已有 `zhu2023groot` |
| 核心 | [SAM-E](https://proceedings.mlr.press/v235/zhang24c.html), Zhang et al., ICML 2024 | 微调 SAM 特征并结合 sequence imitation，提取 embodied manipulation 的 task-relevant features | 与本文使用分割基础模型和序列策略的技术背景接近 | 已有 `zhang2024same` |
| 可选 | [Task-Oriented Hierarchical Object Decomposition for Visuomotor Control](https://proceedings.mlr.press/v270/qian25b.html), Qian et al., CoRL 2024 / PMLR 2025 | 按对象及部件组织可组合表征，为不同任务选择对应实体 | 强化“任务无关的单一全局表征难以忽略无关信息” | 缺失，建议键 `qian2025hodor` |
| 可选 | [Choose What to Observe](https://arxiv.org/abs/2603.07875), Ding et al., 2026 | 用开放词汇分割和深度构造 task-aware semantic--geometric observations | 与本文语义输入最接近，但本文还监督内部 query 表征可恢复任务描述量 | 已有 `ding2026choose` |

### 语义分割需要明确指出的边界

语义分割有三个不能被正文模糊掉的问题：

1. **类别相关不等于决策相关。** 分割图标识对象身份和轮廓，但策略可能需要的是可见面积、工具—目标位移、目标—区域距离等关系量。
2. **输入可用不等于内部保留。** 即使语义图与 RGB 一起输入，高容量动作网络仍可能忽略这些通道，或在压缩时丢失后续决策所需关系。
3. **掩码误差会转化为几何误差。** 漏检、粘连、遮挡传播错误或稳定但错误的分割，都可能产生看似合理而实际错误的描述量。本文的质量筛选用于控制监督质量，但不能把分割结果当成无误真值。

因此，正文可以肯定 semantic/object priors 的价值，但研究缺口应落在“如何约束动作生成表征保留选定任务关系”，而不是笼统批评语义分割。

## 4. 直接约束任务相关性与抑制捷径

| 优先级 | 论文 | 方法要点 | 对本段的作用 | 与本文的区别 | 当前 BibTeX |
|---|---|---|---|---|---|
| 核心 | [Causal Confusion in Imitation Learning](https://proceedings.neurips.cc/paper/2019/hash/947018640bf36a2bb609d3557a285329-Abstract.html), de Haan et al., NeurIPS 2019 | BC 可能从示范相关性中选择错误的因果解释 | 说明低动作预测误差不保证使用了正确决策依据 | 理论诊断，不提供本文这种物理描述量监督 | 已有 `dehaan2019` |
| 核心 | [Fighting Copycat Agents](https://proceedings.neurips.cc/paper/2020/hash/1b113258af3968aaf3969ca67e744ff8-Abstract.html), Wen et al., NeurIPS 2020 | 移除历史中对前一动作的冗余信息，保留预测下一动作所需信息 | 代表通过表示正则抑制时间捷径 | 关注 action-history shortcut，不针对暴露几何 | 已有 `wen2020copycat` |
| 核心 | [Object-Aware Regularization for Addressing Causal Confusion](https://proceedings.neurips.cc/paper/2021/hash/17a3120e4e5fbdc3cb5b5f946809b06a-Abstract.html), Park et al., NeurIPS 2021 | 以对象感知的随机屏蔽防止策略利用与动作相关的 nuisance variables | 连接对象结构与 causal confusion | 促使均匀关注语义对象，不直接规定需保留的物理关系 | 缺失，建议键 `park2021oreo` |
| 核心 | [Task-Relevant Adversarial Imitation Learning](https://proceedings.mlr.press/v155/zolna21a.html), Zolna et al., CoRL 2020 / PMLR 2021 | 约束 adversarial discriminator，避免用任务无关视觉特征区分专家与非专家 | 说明 task relevance 可通过训练目标施加 | 约束的是奖励判别器，不是行为克隆的 action-generating queries | 已有 `zolna2021trail` |
| 可选 | [Rethinking Latent Redundancy in Behavior Cloning](https://proceedings.mlr.press/v267/bai25e.html), Bai et al., ICML 2025 | 以信息瓶颈减少 BC latent redundancy，同时保留操控有用信息 | 代表直接塑造策略 latent 的路线 | 不指定可解释的可见性和空间描述量 | 已有 `bai2025ibbc` |
| 可选 | [CIVIL](https://doi.org/10.1007/s10514-026-10266-3), Dai et al., Autonomous Robots 2026 | 由示范者通过语言和物理标记指出决策相关对象与位置，学习紧凑因果视觉表征 | 与“显式教学哪些信息重要”高度相关 | 依赖额外示范者标注交互；本文从离线语义 masks 自动构造描述量 | 已有 `dai2026civil` |

**总结判断：** 现有工作已经认识到 imitation learning 会利用捷径，也提出对象感知、信息瓶颈或判别器约束。本文的差异不是首次提出“task-aware”，而是把它操作化为一组从联合策略 query 中可恢复的可见性与交互几何描述量。

## 5. task-level analysis 应如何融入本段

原第四点中关于表示的理论工作应放在本段结尾，而不是独立成节：

- [Support and Invertibility in Domain-Invariant Representations](https://proceedings.mlr.press/v89/johansson19a.html), Johansson et al., AISTATS 2019：分布对齐若缺少支持与可逆性条件，可能损失预测所需信息；已有 `johansson2019support`。
- [On Learning Invariant Representations for Domain Adaptation](https://proceedings.mlr.press/v97/zhao19a.html), Zhao et al., ICML 2019：仅追求 domain invariance 可能导致目标任务性能恶化；已有 `zhao2019invariant`。
- [Approximate Information State](https://www.jmlr.org/papers/v23/20-1165.html), Subramanian et al., JMLR 2022：压缩状态的价值取决于其对决策和未来演化的近似充分性；已有 `subramanian2022ais`。

这些工作支持一个克制而准确的落点：压缩或对齐表示可能改善覆盖，但也可能合并需要不同动作的历史。本文的 task-level analysis 因此同时讨论 representation-space coverage 和 expert-action recoverability，并把它们连接到完整任务失败；它给出的是条件性设计解释，不是“有语义监督就能保证成功”。

## 6. 正文的一段式论证链

建议压缩为以下五步：

1. 通用预训练表征提升复用和数据效率，但不一定与当前任务的决策变量对齐；
2. 对象中心和语义方法显式提供对象与空间结构；
3. 指出分割图仍有“类别不等于关系、输入不等于保留、掩码误差传播”三项边界；
4. 现有 task-relevant objectives 主要抑制捷径或塑造奖励/latent，但通常不直接监督本文关心的物理任务状态；
5. 定位本文：语义输入负责结构化观察，descriptor-supervised queries 负责使任务状态在动作生成表征中可恢复，task-level analysis 再说明 coverage 与 recoverability 如何共同影响完整任务。

正文建议保留四组引用：

- 通用预训练及其边界：`nair2023r3m,hansen2023pretraining`；
- 对象/语义结构：`zhu2023viola,zhu2023groot,zhang2024same`；
- 任务相关约束：`dehaan2019,wen2020copycat,park2021oreo,zolna2021trail,bai2025ibbc`；
- 表示分析：`johansson2019support,zhao2019invariant`。

## 7. 当前最值得保留的研究缺口表述

通用预训练、对象中心编码和语义分割可以减少从像素学习操作结构的负担，但它们主要改变策略可见的信息或视觉先验，并不保证动作生成表征保留某个序列任务后续决策所需的可见性与空间关系。已有任务相关正则能够抑制部分视觉或时间捷径，却通常不直接规定可恢复的物理任务状态。本文将语义观察与描述量监督分开：前者提供对象级结构，后者约束联合策略 query 保留目标暴露和交互几何，并通过条件性任务级分析说明这种信息保留的作用与边界。

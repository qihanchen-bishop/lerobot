# 方法章节叙述与实验验证规划

回答时间：2026-09-22 11:34:32 HKT（UTC+08:00）

## 回答内容

建议把方法组织成一条完整的信息处理链：

> **获得可靠的语义监督 → 学习任务相关表示 → 利用执行历史及时纠偏。**

我查看了 `sam2` 的[迭代流程文档](/home/qihan/data/sam2/segdata/SAM2_不确定性驱动自动迭代标注流程.md)、提示生成与转换工具，以及[质量评估代码](/home/qihan/data/sam2/tools/yolo_sam2/evaluate_segmentation_quality.py:359)。已有流程比稿件目前描述的完整，但**流程文档中的设计、代码已具备的能力、最终数据实际执行过的步骤，需要通过每轮记录进一步区分**。

## 一、方法章节结构

建议第四章采用五节，三个核心模块各占一节。

### A. Framework Overview

开头不重复理论推导，直接说明：

> 本方法通过质量驱动的自动标注构建语义监督，通过语义输入和关系监督查询形成任务相关表示，再通过历史条件残差修正执行中的动作块。

配一张总览图，区分**离线标注、策略训练、在线执行**，标明冻结关系和监督来源。交代三个模块顺序训练，部署不使用未来帧或真实标签。

### B. Quality-Aware Iterative Semantic Annotation

本节回答：**如何以有限人工投入，获得可靠且覆盖困难交互状态的标签？**

按“种子→扩展→诊断→局部修复→验收”的顺序写：
- 初始离散提示生成与人工审核，SAM2传播得到种子标签。
- 训练分视角YOLO提示器，扩展至其余视频。
- 计算时序、面积、质心、碎片和可见性异常；双向一致性仅在实际生成了独立双向结果时使用。
- 将问题帧合并为片段，补充提示后局部重跑；超过预算才交人工处理。
- 保留版本、修复记录与人工开销，训练最终在线分割器。

**理论承接：**该流程针对伪标签误差项，而不是直接保证策略成功；高一致性也可能对应稳定误分。

优先考虑三项改进：
1. **区分“确认不可见”和“标签未知”**，防止空mask全部成为零面积监督。
2. **增加任务关系级诊断**，关注质心、工具端点、接触边界的错误，而非只看整体IoU；这用于优先复核，不冒充理论敏感性常数。
3. **保护真实转场、保留随机审计**，不能因刚暴露时面积变化大就删除，也不能仅审核低分样本。

代码目前将缺失双向指标的分量设为1，再参与综合平均；建议改为按可用指标聚合并保留缺失标记，避免“未测量”被解释为“高置信”。离线是否重试可以离散分流，**训练中的质量权重仍可保持连续，不必引入硬控制门控**。

### C. Semantic-Augmented Policy with Relation-Supervised Queries

本节回答：**标签如何转化为策略能够利用的任务信息？**

先写冻结分割器生成soft语义图，与RGB经共享视觉骨干形成视觉token；再写查询与辅助几何头，以及动作解码器如何读取查询隐藏状态。明确视角顺序、类别对应、位置编码和无效关系处理。

**理论承接：**语义提供对象结构先验，查询监督促进关系可读出；对应关系估计误差与表示损失，但不直接控制决策实现误差。

第一版保持现有结构，**不同时更换语义通道、视觉融合和查询机制**。部署零潜变量下的几何读出需要单独评价；查询不是显式阶段分类器，也不是完成判断器。

### D. History-Conditioned Residual Feedback

本节回答：**已有计划形成后，如何利用更新证据修正未来动作？**

依次写：冻结基础计划→新图像和因果记录→固定窗口GRU→未来槽位残差→接收、有效性与执行。明确缓存查询来自旧计划，而非每次反馈重新计算；残差加到原基础动作，不累计旧修正。

**理论承接：**历史提供潜在增量信息，编码与训练决定能否利用，延迟和槽位匹配决定是否生效。GRU是实现选择，限幅不是稳定性保证。

优先评估真实更新间隔与训练缓存的差异；若明显，再考虑按实测延迟分布构建缓存或时间间隔扰动，不先引入新网络。

### E. Training and Execution Protocol

集中列出三阶段训练、损失、冻结模块、数据划分、动作单位和时序参数。方法正文不夹杂大量实验安排；具体对照统一放第五章。

## 二、实验如何递进

建议先固定当前方法完成核心对照，再引入上述改进，避免无法归因。

| 验证层次 | 最小对照 | 核心证据 |
|---|---|---|
| 自动标注 | 单轮传播、质量筛选、质量驱动迭代 | 独立审核集的分割/关系误差、困难帧保留率、人工时间 |
| 语义与查询 | RGB、RGB＋语义、再加无监督查询、再加监督查询 | 成功率、部署模式关系精度、关键事件错误 |
| 历史反馈 | 同一基础checkpoint：无反馈、当前输入MLP、GRU | 成功率、同槽动作误差、修正方向与幅度 |
| 及时执行 | GRU输出屏蔽、正常采用、受控延迟 | 匹配调度下的成功率差、有效采用与过期情况 |
| 完整方法 | RGB/语义查询 × 无反馈/GRU | 两条路径的独立贡献及是否存在交互作用 |

标注对照应使用共同种子和独立审核集，分别报告**同人工预算下的质量**或**同质量目标下的人工成本**。下游只需先比较两种代表性标签版本，不必每轮都重训所有策略。

MLP与GRU共享最新观测、本体、计划及速度等输入，比较额外序列历史；速度本身已有历史来源，要披露。查询屏蔽和历史置乱只能作为依赖性诊断，结合重新训练的消融解释。

## 三、推进顺序

先完成现有流程的证据盘点与独立标签审计；随后锁定语义/查询对照，再完成同基础策略的反馈与调度对照；最后才做标注优化、延迟适配和跨平台扩展。

这样文章的贡献不是三个模块的拼接，而是：**以可追溯的监督建立任务关系表示，并检验这些表示与新增执行证据如何共同改善闭环执行。** 理论负责提出条件，方法负责给出可实现机制，实验负责判断这些机制是否真正奏效。

---

# 理论部分叙述逻辑整理

回答时间：2026-10-03 22:12:11 HKT（UTC+08:00）

## 回答内容

### 一条核心主线

理论部分只需回答一个问题：

> 在有限离线示范下，动作拟合误差为什么不能直接保证任务完成，以及任务关系表示和历史条件反馈分别在什么条件下能够弥补这一差距？

全文应按以下逻辑递进：

1. **先区分训练目标与部署目标。** 模仿学习最小化示范动作误差，但真正目标是任务完成概率，两者不是同一个量。
2. **再建立二者的条件性联系。** 动作误差只有在部署分布被示范分布覆盖、且任务后果对动作偏差不过度敏感时，才能约束失败概率。
3. **由覆盖困难引出任务表示。** 表示可以合并与任务无关的视觉差异，但不能把需要不同动作的状态错误合并，因此需要显式保留任务相关关系。
4. **由静态表示的局限引出历史反馈。** 计划生成后环境仍在变化；新观测和动作--响应历史只有被正确编码、转化为有效修正并及时作用于匹配槽位时，才会提高完成率。

因此，三个模块不是并列堆叠，而是依次处理：**监督可靠性 → 任务信息保留 → 执行期信息利用**。

### 建议章节结构与关键公式

#### A. Problem Formulation：明确理论对象

定义模仿训练风险与部署成功率：

$$
\widehat R_{\mathrm{IL}}(\theta)
=\sum_{i,t}w_{it}\mathcal L_{\mathrm{IL}}(\pi_\theta;\mathcal H_t^i,\zeta_i),
\qquad
J(\pi)=\Pr_\pi(\Psi=1).
$$

本节结论应直接写明：降低 $\widehat R_{\mathrm{IL}}$ 不自动推出提高 $J(\pi)$；后文需要把动作、表示和反馈统一映射到任务后果。

#### B. From Imitation Error to Task Performance：动作误差何时有意义

先以性能差分恒等式建立共同尺度：

$$
F(\pi)-F(\pi_0)
=T\,\mathbb E_{\bar\mu_\pi}
\left[A_t^0(x,\pi(x))\right],
\qquad F(\pi)=1-J(\pi).
$$

它说明，决定任务表现的是**部署策略实际访问状态上的动作后果**，而不是所有帧等权的动作误差。若同时满足：

$$
\bar\mu_\pi(\mathcal B)\le C\nu(\mathcal B),
\qquad
|A_t^{\mathrm E}(x,a)|\le L\|a-\pi_{\mathrm E}(x)\|,
$$

则有：

$$
F(\pi)\le F(\pi_{\mathrm E})+TL\sqrt{C R_\nu(\pi)}.
$$

这里 $C$ 表示示范对部署状态的覆盖程度，$L$ 表示任务后果对动作误差的敏感程度。该式给出第一个明确结论：**动作误差只有在覆盖和敏感性受控时，才是任务表现的可靠代理。** 有限示范、遮挡、多阶段转移和接触过程恰好会使这两个条件难以满足。

完整历史的覆盖通常过于严格。将历史编码为表示 $Z$ 后可得到：

$$
\mathbb E_{\bar\mu_\pi}e(x)^2
\le C_\phi R_\nu(\pi)+D_\phi.
$$

表示可能减小覆盖常数 $C_\phi$，但若它合并了需要不同动作的状态，则条件误差差异 $D_\phi$ 会增大。因此这一式只负责引出下一问题：**表示究竟保留了哪些与任务后果有关的信息？**

#### C. Task-Relevant Representation：语义与QToken为什么可能有效

把表示的影响拆成“信息是否保留”和“信息是否被策略利用”：

$$
\mathcal L_\mu(\pi_Z)-\mathcal L_\mu^*(\mathcal F_h)
=\Delta_{\mathrm{rep}}(\mathcal U)
+\epsilon_{\mathrm{dec}}(\pi_Z).
$$

若任务后果可以由关系 $G$、必要上下文 $\chi$ 和动作 $a$ 近似，近似误差为 $\epsilon_{\mathrm{rel}}$；后果对关系误差的敏感性为 $K_G$；表示中的关系读出误差为 $\delta_G$，则：

$$
\mathcal L_\mu(\pi_Z)-\mathcal L_\mu^*(\mathcal F_h)
\le 2(\epsilon_{\mathrm{rel}}+K_G\delta_G)
+\epsilon_{\mathrm{dec}}(\pi_Z).
$$

这给出语义与QToken有效的三个条件：关系定义足以解释任务后果，使 $\epsilon_{\mathrm{rel}}$ 小；分割和关系监督可靠，使 $\delta_G$ 小；策略确实利用查询表示，使 $\epsilon_{\mathrm{dec}}$ 小。语义并未增加新的物理观测，而是在有限数据下提供任务相关归纳偏置；质量感知标注主要控制伪标签误差及其对 $\delta_G$ 的影响。

#### D. History-Conditioned Feedback：新增信息如何形成实际纠偏

按信息包含关系
$\mathcal F_p\subseteq\mathcal F_c\subseteq\mathcal F_h$，定义新观测和历史的潜在价值：

$$
\Delta_{\mathrm{new}}
=\mathcal L_\mu^*(\mathcal F_p)-\mathcal L_\mu^*(\mathcal F_c),
\qquad
\Delta_{\mathrm{hist}}
=\mathcal L_\mu^*(\mathcal F_c)-\mathcal L_\mu^*(\mathcal F_h).
$$

二者非负，但“存在信息”不等于“模型能利用信息”。候选修正的收益分解为：

$$
\Gamma_\mu
=\epsilon_{\mathrm{base}}+\Delta_{\mathrm{new}}+\Delta_{\mathrm{hist}}
-\Delta_{\mathrm{rep}}(\mathcal U)-\epsilon_{\mathrm{dec}}(\widehat u).
$$

最后，只有及时到达且匹配当前计划槽位的修正才被采用。令 $\Lambda$ 为采用指示量，$g_t(x)$ 为候选修正相对基础动作的任务收益，则：

$$
J(\pi)-J(\pi_0)
=T\,\mathbb E_{\bar\mu_\pi}[\Lambda g_t(x)].
$$

结合关系表示误差和未采用概率 $p_{\mathrm{miss}}$，得到全文最终的充分条件：

$$
J(\pi)-J(\pi_0)\ge T\left[
\epsilon_{\mathrm{base}}+\Delta_{\mathrm{new}}+\Delta_{\mathrm{hist}}
-2(\epsilon_{\mathrm{rel}}+K_G\delta_G)
-\epsilon_{\mathrm{dec}}(\widehat u)-p_{\mathrm{miss}}
\right].
$$

右侧为正时，反馈才有理论上的净收益：新增信息价值和基础策略的改进空间，必须超过表示损失、决策实现误差与时延损失。

### 写作时应强调的边界

- 理论给出的是**成立条件和可检验机制**，不是从网络结构直接证明成功率必然提高。
- 语义/QToken应通过独立关系误差、困难状态表现和成功率验证，不能只报告辅助训练损失。
- GRU的价值应通过同一基础checkpoint下的无反馈、当前帧MLP和历史GRU对比验证，并报告修正采用率与延迟。
- 最终反馈界针对固定基础策略及同一部署分布；不同策略下的局部收益不能直接相加。

建议每节开头先给结论，再给定义和推导，结尾用一句话引出下一节。这样整章形成：**代理目标为何不足 → 需要保留什么信息 → 新信息如何被及时用于纠偏 → 何时真正提高任务完成率**。

## 2026-10-05 14:59:45 HKT | 第三章引导段修改建议

可以删除最后一句中展开收益与损失项的后半句，具体条件留到后文推导。开头保留研究目的，再从模仿误差与任务表现的关系引出两条改进途径。建议正文采用以下版本：

### 英文正文

This section studies conditions for improving task performance from limited demonstrations. We begin by examining when imitation error bounds task-failure probability and why lower training error alone does not guarantee higher task success. This motivates two complementary paths: preserving task-relevant information in policy representations and using new observations and execution history for timely feedback. For the first, we analyze how accurately preserving task relations can limit representation loss, providing a conditional basis for semantic guidance and relation supervision. For the second, we characterize the decision value of new observations and history and derive conditions under which feedback improves task completion.

### 对应中文

本章研究有限示范下改善任务表现的条件。我们首先分析模仿误差何时能够约束任务失败概率，以及为什么更低的训练误差本身不能保证更高的任务成功率。由此引出两条互补的改进途径：在策略表示中保留任务相关信息，以及利用新观测与执行历史实现及时反馈。对于前者，我们分析准确保留任务关系如何约束表示损失，为语义引导与关系监督提供条件性依据。对于后者，我们刻画新观测与历史的决策价值，并推导反馈提高任务完成概率的条件。

这里区分了总体模仿误差与有限示范上的训练误差，避免把后文的理论界直接表述成训练损失的保证。未修改论文正文。

## 2026-10-05 15:49:13 HKT | 第三章引导段修订

### 英文正文

This section studies conditions for improving task performance from limited demonstrations. Starting from the relationship between imitation error and task success, we examine two paths for improvement: preserving information needed to distinguish the task consequences of different actions, and using new observations and execution history to produce effective corrections. We first establish conditions under which action error bounds task-failure probability and explain why improved coverage in representation space alone does not guarantee better task performance. We then bound representation loss in terms of task-relation estimation error, providing a conditional basis for semantic guidance and relation supervision. Finally, we characterize the potential decision value of new observations and history and derive a sufficient condition for feedback to improve task completion.

### 对应中文

本章研究有限示范下改善任务表现的条件。我们从模仿误差与任务成功率的关系出发，考察两条改进途径：保留区分不同动作的任务后果所需的信息，以及利用新观测与执行历史生成有效修正。首先，我们建立动作误差约束任务失败概率的条件，并说明为什么仅改善表示空间中的覆盖仍不足以保证更好的任务表现。随后，我们根据任务关系估计误差给出表示损失的上界，为语义引导与关系监督提供条件性依据。最后，我们刻画新观测与历史的潜在决策价值，并推导反馈提高任务完成概率的充分条件。

本版不预设两条途径具有互补或协同增益；以“考察”而非“保证”引出改进途径。具体假设与误差项留到各节展开。未修改论文正文。

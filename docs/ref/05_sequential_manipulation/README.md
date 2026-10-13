# Sequential manipulation tasks：8 篇相关论文对比

检索与下载日期：2026-10-09。用途：为 `docs/draftv6` 的 Introduction 第一段建立任务背景，并为后续 Related Work 提供线索。

本次选择依据是与“步骤顺序、前置条件、对象关系、遮挡暴露、准备—执行依赖”的相关性；不是按引用量排名，也不是只收录标题含 sequential 的论文。下文明确区分论文自己的任务表述与针对本项目的分析判断。论文信息来自原文、作者/机构页面、正式论文集及出版方登记的 DOI 元数据。

## 1. 结论与阅读顺序

推荐将本文任务称为 **sequential manipulation tasks（序列操作任务）**，并用 **with interdependent steps（包含相互依赖的步骤）** 限定研究范围。不能仅凭 sequential 一词就推断所有子任务有唯一固定顺序，或每个步骤都依赖全部前序步骤。

最值得先读：**01 Liang → 08 BiPreManip → 05 Object Search → 02 C-LEARN**。它们分别支持前置条件、准备—执行关系、可见性与可达性关系，以及多步操作中的几何约束。

## 2. 书目信息对比

表中的标题链接指向已下载的本地 PDF；“来源”指向可核查的外部页面。作者按原文顺序列出。

| 编号与论文 | 作者 | 正式发表时间与刊物/会议 | 保存版本与来源 |
|---|---|---|---|
| 01 [Learning Sensorimotor Primitives of Sequential Manipulation Tasks from Visual Demonstrations](pdf/01_Liang_2022_Sequential_Sensorimotor.pdf) | Junchi Liang; Bowen Wen; Kostas Bekris; Abdeslam Boularias | 2022-05，ICRA 2022，8591–8597 | arXiv v1，2022-03-08；[原文](https://arxiv.org/abs/2203.03797)；[正式 DOI](https://doi.org/10.1109/ICRA46639.2022.9811703) |
| 02 [C-LEARN: Learning Geometric Constraints from Demonstrations for Multi-Step Manipulation in Shared Autonomy](pdf/02_PerezDArpino_2017_CLEARN.pdf) | Claudia Pérez-D’Arpino; Julie A. Shah | 2017，ICRA 2017（会议 05-29 至 06-03；MIT 条目 Date Issued 为 2017-06） | 作者稿；[MIT 机构库与发表信息](https://dspace.mit.edu/entities/publication/ac1ec897-262b-4fbc-a84d-bbd95baa9cda) |
| 03 [Neural Task Programming: Learning to Generalize Across Hierarchical Tasks](pdf/03_Xu_2018_Neural_Task_Programming.pdf) | Danfei Xu; Suraj Nair; Yuke Zhu; Julian Gao; Animesh Garg; Li Fei-Fei; Silvio Savarese | 2018-05，ICRA 2018，3795–3802 | arXiv v2，2018-03-14（首次预印本 2017）；[原文](https://arxiv.org/abs/1710.01813)；[正式 DOI](https://doi.org/10.1109/ICRA.2018.8460689) |
| 04 [Relay Policy Learning: Solving Long-Horizon Tasks via Imitation and Reinforcement Learning](pdf/04_Gupta_2020_Relay_Policy_Learning.pdf) | Abhishek Gupta; Vikash Kumar; Corey Lynch; Sergey Levine; Karol Hausman | CoRL 2019；正式论文集出版年 **2020**，PMLR 100:1025–1037 | 正式论文集 PDF；[PMLR 条目](https://proceedings.mlr.press/v100/gupta20a.html) |
| 05 [Object Search by Manipulation](pdf/05_Dogar_2014_Object_Search.pdf) | Mehmet R. Dogar; Michael C. Koval; Abhijeet Tallavajhula; Siddhartha S. Srinivasa | 2014-01，Autonomous Robots，36(1):153–167 | 期刊作者稿；本条不是同名 ICRA 2013 会议版；[CMU 期刊条目](https://publications.ri.cmu.edu/object-search-by-manipulation-2) |
| 06 [Mechanical Search: Multi-Step Retrieval of a Target Object Occluded by Clutter](pdf/06_Danielczuk_2019_Mechanical_Search.pdf) | Michael Danielczuk; Andrey Kurenkov; Ashwin Balakrishna; Matthew Matl; David Wang; Roberto Martín-Martín; Animesh Garg; Silvio Savarese; Ken Goldberg | 2019，ICRA 2019 | arXiv v1，2019-03-04；[原文](https://arxiv.org/abs/1903.01588)；[作者项目与会议标注](https://ai.stanford.edu/mech-search/) |
| 07 [Preparatory Manipulation Planning Using Automatically Determined Single and Dual Arm](pdf/07_Wan_2020_Preparatory_Manipulation.pdf) | Weiwei Wan; Kensuke Harada; Fumio Kanehiro | 2020-01，IEEE Transactions on Industrial Informatics，16(1):442–453 | arXiv v1，2018-12-08，预印本标题末词为 Arms；[预印本](https://arxiv.org/abs/1812.03274)；[正式 DOI](https://doi.org/10.1109/TII.2019.2892772) |
| 08 [BiPreManip: Learning Affordance-Based Bimanual Preparatory Manipulation through Anticipatory Collaboration](pdf/08_Shen_2026_BiPreManip.pdf) | Yan Shen; Feng Jiang; Zichen He; Xiaoqi Li; Yuchen Liu; Zhiyu Li; Ruihai Wu; Hao Dong | 2026，CVPR 2026 | 下载的是 arXiv v1，2026-03-23；[可读正文](https://arxiv.org/html/2603.21679v1)；[CVF 正式收录来源](https://openaccess.thecvf.com/content/CVPR2026/papers/Shen_BiPreManip_Learning_Affordance-Based_Bimanual_Preparatory_Manipulation_through_Anticipatory_Collaboration_CVPR_2026_paper.pdf) |

版本说明：07 的 DOI 含 2019，但卷期发表年份是 2020；04 的会议年份和论文集年份不同。08 的 CVF 条目可检索，直接下载受到 403 限制，因此保存公开 arXiv 稿，并未将其冒充最终出版 PDF。此次选集没有 T-RO 论文；07 是 T-II，05 是 Autonomous Robots，不应误标为 T-RO。

## 3. 与序列操作相关的观点和任务实例

以下“论文观点/任务例子”是原文概述；“本项目联系与边界”是本次阅读分析，不是原作者对本项目的评价。

| 编号 | 原文使用的任务概念 | 论文观点与任务例子 | 顺序或依赖体现在哪里 | 本项目联系与边界 |
|---|---|---|---|---|
| 01 | Sequential manipulation tasks | 学习连接低层技能，并判断子目标及下一技能的前置条件；例子包括蘸漆后涂刷、堆叠、移除轮胎螺母后取下轮胎。 | 对象类别、相对位姿及已执行动作历史共同影响步骤选择。 | **最直接的术语依据。** 与工具—目标关系和执行历史都相关；不能声称已有工作普遍未结合这两类信息。 |
| 02 | Multi-step / functional manipulation | 用关键帧序列和每步的几何约束表示任务；例子包括固定箱体后抽取圆柱、开门后在受限空间按按钮。 | 除了步骤排列，还要满足执行中的空间与多臂约束。 | 可用于说明动作序列必须符合实际几何关系；其共享自主与规划设定不同于本文连续视觉动作策略。 |
| 03 | Sequential tasks with hierarchical and compositional structures | 从任务示范递归分解子任务，测试物体分类、积木堆叠和桌面清理。 | 研究任务长度、子任务组合/排列及目标变化；执行利用当前观测。 | 支持序列操作的上位背景；其组合泛化不等同于本文关注的遮挡暴露后执行调整。 |
| 04 | Multi-stage, long-horizon manipulation | 从非结构化 play 示范初始化层级策略，再用 RL 改进；厨房例子涉及微波炉、水壶、开关、柜门。 | 将多个目标组成多阶段行为；低层按固定时长运行。 | 可说明示范学习处理多阶段任务的路线；这些目标不能自动解释为彼此都有严格物理前置条件。 |
| 05 | Object search by manipulation | 联合考虑对象间可见性与可达性关系，移动遮挡或阻挡对象以找到目标。 | 某对象既可能挡住视线，也可能阻止机械臂接近另一对象。 | **最适合支持遮挡—暴露动机。** 文中目标是搜索，不覆盖本文完整工具移动及恢复任务。 |
| 06 | Mechanical search / multi-step retrieval | 通过推、吸取、抓取等动作，定位并取出被杂物遮挡的已知目标。 | 动作改变杂物配置和目标暴露状态，下一动作需要面对更新后的场景。 | 提供“操作改变后续感知与操作条件”的实例；不要将本文整个任务直接命名为 mechanical search。 |
| 07 | Preparatory manipulation | 通过重抓取、放置或交接调整对象姿态，为后续使用做准备；例子包括电钻定向、装配工件定向。 | 准备动作使目标对象进入适于后续操作的配置。 | 支持“先建立操作条件”的动机；定向后的钻孔等是应用动机，不应一概写成该文完整实验已执行。 |
| 08 | Bimanual / collaborative preparatory manipulation | 一臂先重配置对象，使另一臂的目标操作可行；例子包括推至桌边再抓取，以及调整瓶/笔姿态后开盖。 | 不对称双臂角色，准备动作与随后目标动作存在依赖。 | **与双臂准备—执行结构最接近。** 该文主要重配置目标对象；本文还涉及遮挡物、工具与目标之间的关系。 |

上述每行的外部来源和本地原文见第 2 节同编号链接；快速原文定位见下一节。

## 4. 原文定位与阅读重点

页码均为本地 PDF 文件页码，不是正式论文集连续页码。

| 编号 | 建议阅读位置 | 阅读时核对的问题 |
|---|---|---|
| 01 | 第 1 页 Abstract / Introduction；第 2 页问题表述；第 5–6 页实验 | 作者如何把前置条件、相对位姿与历史联系起来？其 LSTM 历史用于何种决策？ |
| 02 | 第 1 页 Introduction / Fig. 1；后续任务表示及执行章节 | 为什么顺序正确仍不足以保证操作符合几何约束？哪些环节属于共享自主？ |
| 03 | 第 1–2 页 Introduction；第 6–8 页堆叠和桌面清理实验 | 任务长度、拓扑、语义三种泛化各指什么？闭环执行与任务分解如何配合？ |
| 04 | 第 1–2 页 Introduction；第 6 页 §5 Experimental Results | 方法为何利用示范初始化再强化学习？任务目标集合与严格步骤依赖有何区别？ |
| 05 | 第 1–2 页 Introduction；§2 对关系的形式化 | 可见性和可达性如何区分？对象之间的阻挡怎样影响动作顺序？ |
| 06 | 第 1 页 Introduction / Fig. 1；第 2 页 §III Problem Statement | 搜索目标与终止条件是什么？物理交互怎样改变下一步决策的依据？ |
| 07 | 第 1 页 Introduction / Fig. 1；后续 regrasp graph 描述 | 为什么需要准备动作？规划终点是对象准备姿态还是整个功能任务完成？ |
| 08 | 第 1–2 页 Introduction / Fig. 1–2；§3.5 | 准备臂如何服务于目标操作？准备完成后怎样重新使用更新观测？ |

尤其注意 01：其对象类别与 6D 位姿来自视觉模块，LSTM 记忆动作历史；涂刷例子说明，仅凭这些当前几何观测不能判断刷子是否已经蘸漆。因而，它是本文后续 Related Work 中需要认真比较的已有工作，而不是仅用于第一段的泛泛背景引用。[原文](https://arxiv.org/abs/2203.03797)

## 5. 对 draftv6 第一段的写作建议

### 5.1 如何定义本文关注的范围

下面是基于本项目任务的建议性界定，不是对全部领域强加统一定义：

> 本文关注包含相互依赖步骤的机器人序列操作任务，其中前序交互改变对象状态及其关系，为后续操作建立必要条件。

英文可用：

> We consider sequential manipulation tasks with interdependent steps, in which preceding interactions change object states and relations to establish the conditions required for subsequent operations.

“前置条件”在这里描述任务结构，不意味着本文方法实现了显式条件分类器、阶段识别器或符号规划器。

### 5.2 第一段中不同论点应引用谁

| 想支撑的论点 | 优先引用 | 使用边界 |
|---|---|---|
| 序列操作涉及技能连接与前置条件 | 01；需要更宽背景时补 03 | 不把所有序列操作都说成唯一固定顺序 |
| 前序操作为后续操作建立空间/物理条件 | 02、07、08 | 准备性操作是本文所关注的依赖结构之一 |
| 移动遮挡物改变目标可见性及操作可达性 | 05、06 | 在本实验中分别核对视觉遮挡与物理阻挡，不能混为一谈 |
| 多阶段任务可由示范驱动的学习方法处理 | 01、03、04 | 04 包含 RL 微调，不作为纯离线模仿学习的效果证明 |

第一段通常不需要把 8 篇全部引用。建议以 **01 + 05/06 + 08** 为核心，按具体句子选择 3–4 篇；其余留给 Related Work。

### 5.3 与遮挡—暴露—执行实验的对应（本项目分析）

1. **遮挡暴露：**遮挡物操作改变目标可见情况，也可能改变工具接近条件；应以实际几何情况区分两者。
2. **工具执行：**目标暴露及工具—目标关系共同影响后续动作的适用性，不能仅凭“已执行拉布动作”认定准备成功。
3. **后续恢复：**完整任务还包含遮挡物恢复等最终要求，因此 mechanical search 或 preparatory manipulation 都只能概括其中一部分。
4. **后文研究承接：**第一段提出依据实际交互结果调整行为的任务要求；第二段再讨论有限示范策略在执行偏差下的困难。第一段不提前宣称语义表示或历史反馈是唯一解决方案。

可供修改的中文段落草稿（括号编号对应本目录论文，尚未替换为 LaTeX 引用键）：

> 机器人序列操作任务涉及多个相互依赖的步骤，前序操作引起的对象状态变化为后续操作建立必要条件〔01、08〕。在遮挡环境中，机器人可能需要先移动其他对象，使目标暴露并具备后续操作条件〔05、06〕。例如，在借助工具移动被遮挡目标的任务中，移开遮挡物的实际效果会影响目标的可见性及后续工具操作。因而，策略需要依据交互过程中对象状态及其关系的变化调整后续行为，使各步操作共同推进任务完成。

其中第三句是对本项目实验的说明，不是声称这些文献研究了完全相同的任务。正式写作时可以将首句收窄为“本文关注的机器人序列操作任务……”，避免将特定依赖结构泛化到所有序列任务。

## 6. 对后续研究缺口表述的约束

以下是基于本次文献阅读对本文写作的建议，不是新的文献结论：

- 不宜写“现有研究没有利用对象关系和历史”。01 已联合使用，03 也明确使用观测反馈。
- 不宜将“有先后顺序”直接等同于“长程”，或者将“有多个目标”直接等同于“前置条件依赖”。
- 任务名称不决定方法结构。本文可以研究序列操作，同时采用连续动作策略与残差反馈，而非显式技能规划。
- 本文差异应落实到理论分析对象、表示监督目标、反馈预测目标、信息条件及实际实现上。不能由这 8 篇任务背景文献推断整个领域缺少理论或执行纠正。
- “执行偏差后能否持续推进任务”可以作为研究问题；更强的恢复保证或阶段完成保证需要另外的理论与实验支撑。

## 7. 文件结构、版本与下载核验

- `pdf/`：8 篇论文全文 PDF，均已通过 `pdfinfo` 解析并提取文本。
- `text/`：由 PDF 提取的检索文本。公式、双栏阅读顺序或字符可能有误，引用以 PDF 为准。
- `sources/`：成功取得的作者 LaTeX 源码压缩包。它与 PDF 原文是两种不同交付物；没有源码的论文仍保留完整 PDF。
- `download_manifest.json`：下载地址、文件大小、PDF SHA-256、源码下载状态及失败原因。
- `publication_metadata.json`：01、03、07 的 Crossref DOI 元数据，用于核对正式发表信息。
- `download_status.md`：每篇 PDF 页数与源码可用性汇总。

LaTeX 源码获取不保证全部成功：部分 arXiv 下载返回 429，部分传输中断；未将错误页面或不完整压缩包保存为源码。02、05 未找到公开 LaTeX 包。详情以下载清单为准。

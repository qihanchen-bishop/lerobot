# Bimanual Sequential Manipulation：文献检索与分类

检索日期：2026-11-19  
用途：为 Related Work 第一小节 **Bimanual Sequential Manipulation** 建立文献池。本文档以任务与问题定位为主；同一论文可以在第一节从任务角度引用，再在第二小节 **Learning Visuomotor Policies** 从方法角度展开。

## 1. 检索边界

本小节需要回答的不是“有哪些双臂算法”，而是以下三个问题：

1. 双臂机器人已经能够完成哪些具有多个连续操作的任务？
2. 哪些工作研究了双臂角色互补、准备性操作，以及前序操作对后续操作的影响？
3. 当场景包含可形变物体、遮挡和目标暴露时，还会出现哪些未被一般双臂任务充分体现的问题？

文献按四类整理：

- 基础综述、任务分类与基准；
- 一般双臂序列任务；
- 角色互补与准备性操作；
- 可形变、遮挡与目标暴露。

优先级说明：

- **核心**：建议进入第一小节正文；
- **可选**：用于补充某个论点，篇幅不足时可删；
- **跨节**：第一节引用其任务设置与挑战，第二节再讨论其策略结构与能力边界。

## 2. 基础综述、任务分类与基准

| 优先级 | 论文 | 主要内容 | 对本小节的作用 | 当前 BibTeX |
|---|---|---|---|---|
| 可选 | [Dual Arm Manipulation—A Survey](https://doi.org/10.1016/j.robot.2012.07.005), Smith et al., RAS 2012 | 系统梳理双臂协作、抓取、规划与控制，指出双臂系统在非结构环境中的协调复杂性 | 可作为双臂操作总体背景，但年代较早，不宜承担本文的主要问题定位 | 缺失，建议键 `smith2012dual` |
| 核心 | [A Bimanual Manipulation Taxonomy](https://doi.org/10.1109/LRA.2022.3196158), Krebs and Asfour, RA-L 2022 | 从对称性、两臂目标及相互关系等方面对双臂行为分类 | 支持区分对称协作、非对称角色和共同任务目标；本文属于角色随阶段变化的非对称协作 | 缺失，建议键 `krebs2022taxonomy` |
| 可选 | [PerAct2: Benchmarking and Learning for Robotic Bimanual Manipulation Tasks](https://arxiv.org/abs/2407.00278), Grotz et al., 2024 | 扩展 RLBench，构建语言条件双臂 6-DoF 操作任务和基线 | 表明双臂任务已形成专门基准；更适合用于说明任务覆盖面 | 缺失，建议键 `grotz2024peract2` |
| 可选 | [BiGym: A Demo-Driven Mobile Bi-Manual Manipulation Benchmark](https://proceedings.mlr.press/v270/chernyadev25a.html), Chernyadev et al., CoRL 2024 / PMLR 2025 | 包含 40 个家庭环境任务，从简单到复杂厨房清理，并提供人类示范 | 可用于说明双臂序列操作正从桌面技能扩展到复杂家庭任务 | 缺失，建议键 `chernyadev2025bigym` |

## 3. 一般双臂序列任务

这组工作证明端到端学习策略可以执行精细、动态或长序列双臂任务，但多数工作的任务主线是动作协调、数据采集或策略表达能力，并未单独研究动作表征是否保留前序交互所形成的任务状态。

| 优先级 | 论文 | 代表任务或特点 | 对本文定位的价值 | 当前 BibTeX |
|---|---|---|---|---|
| 核心 / 跨节 | [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://roboticsproceedings.org/rss19/p016.html), Zhao et al., RSS 2023 | 双臂传递、插入、拉开密封袋、电池安装、开杯盖等精细多步骤任务；提出 ACT | 第一节用来说明精细双臂任务的阶段性与接触依赖；第二节再讨论 ACT 的动作分块 | 已有 `zhao2023act` |
| 核心 | [Mobile ALOHA: Learning Bimanual Mobile Manipulation using Low-Cost Whole-Body Teleoperation](https://proceedings.mlr.press/v270/fu25b.html), Fu et al., CoRL 2024 / PMLR 2025 | 炒菜、收纳锅具、乘电梯、清洗锅具等长序列全身双臂任务 | 说明双臂序列操作的应用范围与长时程特点；任务依赖多个连续步骤，但论文重点是移动双臂数据采集和协同训练 | 已有 `fu2024mobilealoha`，目前是 arXiv 条目，可后续更新正式 PMLR 信息 |
| 可选 | [Universal Manipulation Interface](https://roboticsproceedings.org/rss20/p045.html), Chi et al., RSS 2024 | 从在野人类示范学习动态、双臂、精细和长序列行为 | 说明复杂双臂任务可以由统一示范接口学习；其主要贡献属于数据接口和策略学习 | 已有 `chi2024umi` |
| 可选 | [You Only Teach Once](https://roboticsproceedings.org/rss21/p149.html), Zhou et al., RSS 2025 | 从单次双目人类视频生成训练数据，覆盖开抽屉后取放、倒水、拧瓶盖、开箱等长序列双臂任务 | 适合说明长序列双臂任务的数据效率问题；方法细节应放第二节 | 已有 `zhou2025yoto` |
| 可选 / 跨节 | [ALOHA Unleashed: A Simple Recipe for Robot Dexterity](https://proceedings.mlr.press/v270/zhao25b.html), Zhao et al., CoRL 2024 / PMLR 2025 | 在 ALOHA 2 上学习涉及可形变物体和复杂接触动力学的双臂任务 | 第一节连接一般双臂学习与可形变任务；第二节讨论“大规模数据＋表达性策略”的路线 | 缺失，建议键 `zhao2025alohaunleashed` |
| 跨节 | [InterACT: Inter-dependency Aware Action Chunking with Hierarchical Attention Transformers for Bimanual Manipulation](https://proceedings.mlr.press/v270/lee25a.html), Lee et al., CoRL 2024 / PMLR 2025 | 用分层注意力建模双臂关节和视觉输入之间的相互依赖 | 第一节引用双臂交互依赖的任务特征；第二节比较其分层注意力与 ACT 类方法 | 缺失，建议键 `lee2025interact` |

## 4. 角色互补与准备性操作

这组工作与本文的任务结构最接近。它们不只要求两臂同时运动，而是强调两臂承担不同角色，或者由一个操作建立另一个操作所需的条件。

| 优先级 | 论文 | 主要任务结构 | 可支持的论点 | 与本文的区别 | 当前 BibTeX |
|---|---|---|---|---|---|
| 核心 | [Stabilize to Act: Learning to Coordinate for Bimanual Manipulation](https://proceedings.mlr.press/v229/grannen23a.html), Grannen et al., CoRL 2023 | 一只手稳定环境，另一只手执行切菜、拉拉链等操作；根据状态重新稳定 | 双臂协作常体现为非对称的“辅助臂—执行臂”角色，而不是两个独立单臂策略的简单组合 | 重点是稳定角色和 acting policy；本文中两臂角色随暴露、目标操作和恢复阶段改变 | 缺失，建议键 `grannen2023stabilize` |
| 核心 | [BiPreManip: Learning Affordance-Based Bimanual Preparatory Manipulation through Anticipatory Collaboration](https://openaccess.thecvf.com/content/CVPR2026/html/Shen_BiPreManip_Learning_Affordance-Based_Bimanual_Preparatory_Manipulation_through_Anticipatory_Collaboration_CVPR_2026_paper.html), Shen et al., CVPR 2026 | 先预测主臂的目标交互，再由辅助臂执行预抓取、旋转或推到桌边等准备动作 | 最直接支持“前序操作为后续操作建立条件”，也是本文必须正面区分的工作 | 以显式 affordance 和主/辅臂分工完成准备后操作；本文关注单个视觉模仿策略在完整暴露—执行—恢复过程中保留任务状态 | 已有 `shen2026bipremanip` |
| 核心 | [Goal-Conditioned Dual-Action Imitation Learning for Dexterous Dual-Arm Robot Manipulation](https://doi.org/10.1109/TRO.2024.3372778), Kim et al., T-RO 2024 | 香蕉剥皮被划分为多个 reaching 和 peeling 子任务，并结合全局轨迹与局部反应动作 | 说明可形变物体上的长序列双臂任务需要稳定的全局运动和接触阶段的闭环反应 | 依赖人工子任务划分、动作类型切换和专门分类器；本文不使用显式阶段或全局/局部动作标签 | 已有 `kim2024dualaction` |
| 可选 | [Preparatory Manipulation Planning Using Automatically Determined Single and Dual Arm](https://doi.org/10.1109/TII.2019.2892772), Wan et al., TII 2020 | 通过单臂或双臂重定向物体，使钻孔、装配、包装等后续任务可执行 | 提供“准备性操作”的经典规划视角，说明最终任务的可行性依赖前置配置 | 依赖显式几何与规划；本文从有限视觉示范学习闭环策略 | 已有 `wan2020preparatory` |
| 可选 | [Constraints Extraction from Asymmetrical Bimanual Tasks and Their Use in Coordinated Behavior](https://doi.org/10.1016/j.robot.2017.12.011), Pais Ureche and Billard, RAS 2018 | 从人类示范提取非对称双臂任务约束，区分辅助与主动操作 | 可作为非对称角色分工的较早工作 | 更关注约束提取与协调执行，不研究长序列中的视觉任务状态 | 缺失，建议键 `paisureche2018constraints` |

## 5. 可形变物体、遮挡与目标暴露

### 5.1 双臂可形变物体操作

| 优先级 | 论文 | 主要内容 | 对本文的作用 | 当前 BibTeX |
|---|---|---|---|---|
| 核心 | [FabricFlowNet: Bimanual Cloth Manipulation with a Flow-based Policy](https://proceedings.mlr.press/v164/weng22a.html), Weng et al., CoRL 2021 / PMLR 2022 | 以光流表示当前布料与目标形态关系，并在单臂/双臂动作间选择 | 说明布料的高维形变、自遮挡和非刚体变化使视觉双臂控制困难；也说明任务相关结构化表示的价值 | 缺失，建议键 `weng2022fabricflownet` |
| 核心 | [Bimanual Deformable Bag Manipulation Using a Structure-of-Interest Based Neural Dynamics Model](https://arxiv.org/abs/2401.11432), Zhou et al., T-Mech 2025 | 用 structure of interest 压缩并预测袋口等关键区域的动力学 | 说明无需重建完整可形变状态，可以围绕任务相关结构建模 | 已有 `zhou2025bag` |
| 可选 | [SpeedFolding: Learning Efficient Bimanual Folding of Garments](https://doi.org/10.1109/IROS47612.2022.9981402), Avigal et al., IROS 2022 | 学习双臂抓取姿态和折叠动作，实现高效衣物折叠 | 可用于展示典型双臂布料任务，但与序列条件依赖的联系弱于 FabricFlowNet | 缺失，建议键 `avigal2022speedfolding` |
| 可选 | [MOTORCYCLE 1.0: Automating Bimanual Cable Routing Around Fixtures on the NIST Task Board](https://autolab.berkeley.edu/assets/publications/media/CASE_2025_MOTORCYCLE_1_0_CAMREADY.pdf), Azulay et al., CASE 2025 | 学习线缆追踪并通过双臂滑动规划控制松弛，完成绕夹具布线 | 说明双臂可用于管理可形变线性物体的张力和空间约束 | 缺失，建议键 `azulay2025motorcycle` |
| 可选 / 跨节 | [Multistage Cable Routing Through Hierarchical Imitation Learning](https://doi.org/10.1109/TRO.2024.3353075), Luo et al., T-RO 2024 | 组合低层技能和高层选择完成多夹具线缆布设，并处理恢复与重试 | 第一节可作为非双臂但任务结构相近的多阶段可形变操作；第二节再讨论层次化模仿与恢复 | 已有 `luo2024cable` |

### 5.2 遮挡、暴露与场景改变

| 优先级 | 论文 | 主要内容 | 对本文的作用 | 与本文的区别 | 当前 BibTeX |
|---|---|---|---|---|---|
| 核心 | [Mesh-based Dynamics with Occlusion Reasoning for Cloth Manipulation](https://roboticsproceedings.org/rss18/p011.html), Huang et al., RSS 2022 | 从单幅深度图重建包含不可见区域的完整布料网格，再用于规划展开和规范化 | 直接证明自遮挡会使可形变物体状态估计和下游规划困难 | 目标是恢复完整布料状态和形态；本文保留 RGB，并监督少量与后续决策相关的任务描述量 | 缺失，建议键 `huang2022medor` |
| 核心 | [Object Search by Manipulation](https://doi.org/10.1007/s10514-013-9372-x), Dogar et al., Autonomous Robots 2014 | 通过移动遮挡物改善目标可见性和可达性 | 建立“通过操作改变场景以支持后续目标任务”的问题脉络 | 主要处理刚性物体搜索和规划，不涉及可形变遮挡下的完整后续操作与恢复 | 已有 `dogar2014search` |
| 可选 | [Mechanical Search: Multi-Step Retrieval of a Target Object Occluded by Clutter](https://doi.org/10.1109/ICRA.2019.8794143), Danielczuk et al., ICRA 2019 | 反复感知并推动、吸取或抓取刚性杂物，最终检索被遮挡目标 | 支持“暴露和检索本身就是多步操作” | 重点是刚性杂物中的目标检索，而不是暴露后继续操作并恢复场景 | 已有 `danielczuk2019mechanical` |
| 核心 | [Autonomous Tissue Retraction in Robotic Assisted Minimally Invasive Surgery](https://doi.org/10.1109/LRA.2020.3013914), Attanasio et al., RA-L 2020 | 规划组织抓取和牵开以暴露目标区域 | 提供可形变覆盖物下“物理暴露”的真实应用动机 | 以达到暴露为终止目标，没有研究暴露后操作与恢复的完整依赖 | 已有 `attanasio2020retraction` |
| 核心 | [Learning from Demonstrations for Autonomous Soft-Tissue Retraction](https://arxiv.org/abs/2110.00336), Pore et al., ISMR 2021 | 从示范学习软组织牵开策略以暴露目标区域 | 说明学习方法可用于接触丰富、物性变化明显的暴露任务 | 同样以暴露为主要目标，未联合后续目标操作 | 已有 `pore2021retraction` |
| 可选 | [Learning-Based Adaptive Control for Surgical Robotic Exposure Task on Deformable Tissues](https://arxiv.org/abs/2605.17927), Liu et al., 2026 | 根据组织视觉边界在线调整牵开控制并选择抓取点 | 强化“暴露需要根据形变结果闭环调整”的论点 | 聚焦 ROI 暴露与控制收敛，而非完整序列策略的任务信息保留 | 已有 `liu2026exposure` |
| 可选 | [Active Vision Might Be All You Need: Exploring Active Vision in Bimanual Robotic Manipulation](https://arxiv.org/abs/2409.17435), Chuang et al., ICRA 2025 | 学习移动相机以改善双臂操作中的观测 | 可用于区分“移动传感器改善可见性”和“通过接触改变场景以建立可见性与操作空间” | 本文的暴露动作同时改变感知条件和物理可操作性 | 已有 `chuang2025activevision` |

## 6. 建议的第一小节核心引用集合

第一小节正文不需要覆盖以上全部论文。建议优先保留以下 10--12 篇，分别承担不同论证功能：

1. **双臂任务的基本类型**：Krebs and Asfour 2022；
2. **代表性精细/长序列双臂任务**：ACT 2023、Mobile ALOHA 2025，可选 UMI 2024 或 YOTO 2025；
3. **非对称角色与前置条件**：Stabilize to Act 2023、BiPreManip 2026、GC-DA 2024；
4. **可形变与遮挡困难**：FabricFlowNet 2022、MEDOR 2022、Bimanual Bag 2025；
5. **暴露作为准备性操作**：Object Search by Manipulation 2014、Attanasio et al. 2020 或 Pore et al. 2021。

篇幅更紧时，可以压缩为八篇：ACT、Mobile ALOHA、Stabilize to Act、BiPreManip、GC-DA、FabricFlowNet、MEDOR、Tissue Retraction。

## 7. 当前采用的一段式组织

正文采用 T-RO 范文的黑体引导加单段形式，逻辑压缩为四步：

1. 用任务分类界定双臂操作从对称协调到角色互补；
2. 用代表性学习系统说明精细和长序列任务已经能够被执行；
3. 从稳定、准备性操作推进到可形变与遮挡场景中的条件建立问题；
4. 指出现有工作多把整形、检索或暴露作为终点，而本文把暴露、目标操作和恢复组成一个完整序列。

这段只回答“现有任务研究了什么、本文任务补充了什么”，具体策略范式和表征方法分别留给后两点。

## 8. 建议补入 `references.bib` 的文献

优先级从高到低：

1. `grannen2023stabilize` — Stabilize to Act；
2. `weng2022fabricflownet` — FabricFlowNet；
3. `huang2022medor` — Mesh-based Dynamics with Occlusion Reasoning；
4. `krebs2022taxonomy` — A Bimanual Manipulation Taxonomy；
5. `zhao2025alohaunleashed` — ALOHA Unleashed；
6. `chernyadev2025bigym` — BiGym；
7. `avigal2022speedfolding` — SpeedFolding；
8. `azulay2025motorcycle` — MOTORCYCLE 1.0；
9. `grotz2024peract2` — PerAct2；
10. `lee2025interact` — InterACT（主要用于第二小节）。

正式加入 BibTeX 前，应以论文正式出版页面重新核对作者顺序、卷页、年份和 DOI。特别是 CoRL 2024 论文的 PMLR 页面使用 2025 年出版年份，正文引用年份需要统一。

## 9. 当前最值得保留的研究缺口表述

现有工作已经展示了精细、动态和长序列的双臂操作，也研究了稳定—执行和准备—操作等非对称协作形式。可形变遮挡进一步使前序操作的结果具有不确定性，但相关工作通常将布料整形、目标检索或区域暴露本身作为终止目标。本文关注的是一个不同但相邻的问题：在暴露、目标操作和恢复组成的完整序列中，视觉模仿策略能否保留前序交互形成的任务状态，并据此选择后续动作。

# T-RO 学习控制论文阅读指南

整理日期：2026-10-12。以下五篇均已核对为 IEEE Transactions on Robotics 期刊论文，选择重点是多阶段操作、双臂可形变物体操作、模仿学习、感知反馈与残差学习。建议按编号阅读，前两篇与当前任务最接近。

本目录保存公开的 arXiv 作者版本，不保证与出版社最终排版逐字相同。期刊年份以 DOI 注册信息为准，不以 arXiv 首次上传或更新年份为准。每个 PDF 均已检查页数、标题与文本可提取性。引用条目见 [references.bib](references.bib)。

## 1 多阶段操作与失败恢复

**Multistage Cable Routing Through Hierarchical Imitation Learning**，Jianlan Luo 等，T-RO 40: 1476–1491, 2024。

[本地 PDF](01_Multi_Stage_Cable_Routing_TRO_2024.pdf) · [期刊 DOI](https://doi.org/10.1109/TRO.2024.3353075) · [作者版本 v5](https://arxiv.org/abs/2307.08927v5) · [项目主页](https://sites.google.com/view/cablerouting)

通过视觉分层模仿学习完成多阶段线缆布设，高层选择和重试低层技能，处理可形变物体与执行失败。

建议重点读 III System and Task Setup、IV Cable Routing via Hierarchical Imitation、VI Experimental Evaluation。尤其关注其低层策略、高层策略和历史信息消融，以及不同夹具配置下的评价。适合借鉴“长序列困难为什么需要恢复能力”的引言逻辑，以及整体成功与局部阶段表现如何分开报告。它的高层技能选择不同于你的连续残差纠偏。

本地文件：16 页；下载自 https://arxiv.org/pdf/2307.08927v5 。arXiv 标题使用 Multi-Stage，期刊注册标题使用 Multistage。

## 2 双臂可形变物体操作与动作生成方式

**Goal-Conditioned Dual-Action Imitation Learning for Dexterous Dual-Arm Robot Manipulation**，Heecheol Kim、Yoshiyuki Ohmura、Yasuo Kuniyoshi，T-RO 40: 2287–2305, 2024。

[本地 PDF](02_Goal_Conditioned_Dual_Action_TRO_2024.pdf) · [期刊 DOI](https://doi.org/10.1109/TRO.2024.3372778) · [作者版本 v3](https://arxiv.org/abs/2203.09749v3)

以双臂剥香蕉为代表任务，结合目标条件下的局部反应动作与全局轨迹生成，处理精细接触和长程执行的不同需求。

建议重点读 III Preliminaries、IV Method、V Evaluation，特别是 V-B Ablation Studies 如何先提出具体问题，再安排对应消融。适合参考双臂序列任务的定义、子任务划分、阶段结果与完整任务表现的组织。其局部／全局动作不是你的“冻结基础策略＋历史残差”，不要将两者等同。

本地文件：19 页；下载自 https://arxiv.org/pdf/2203.09749v3 。v3 更新于 2025 年，但期刊发表年份为 2024 年。

## 3 额外感知信息与反馈价值的验证

**Multimodal and Force-Matched Imitation Learning With a See-Through Visuotactile Sensor**，Trevor Ablett 等，T-RO 41: 946–959, 2025。

[本地 PDF](03_Multimodal_Force_Matched_Imitation_TRO_2025.pdf) · [期刊 DOI](https://doi.org/10.1109/TRO.2024.3521864) · [作者版本 v5](https://arxiv.org/abs/2311.01248v5) · [项目主页](https://papers.starslab.ca/sts-il/)

围绕接触密集操作，将视觉和触觉信息接入模仿学习，并研究力匹配及感知模式切换。

建议重点读 III Methodology、IV Experiments、V Limitations。适合参考如何分别验证数据采集改进、输入模态和控制机制，而不是仅比较完整系统。对你最有价值的是“额外信息是否有效”的实验组织；触觉带来的新测量与从 RGB 导出的语义归纳偏置不同，理论解释不能直接照搬。

本地文件：14 页；下载自 https://arxiv.org/pdf/2311.01248v5 。DOI 含 2024，但正式卷次年份为 2025。

## 4 基础模型与学习残差的分工

**TossingBot: Learning to Throw Arbitrary Objects With Residual Physics**，Andy Zeng 等，T-RO 36(4): 1307–1319, 2020。

[本地 PDF](04_TossingBot_Residual_Physics_TRO_2020.pdf) · [期刊 DOI](https://doi.org/10.1109/TRO.2020.2988642) · [作者版本 v3](https://arxiv.org/abs/1903.11239v3) · [MIT 期刊版本记录](https://dspace.mit.edu/entities/publication/9e2decc6-1568-4d38-b308-54b6332e2409)

结合解析物理控制与视觉学习残差，联合学习抓取和投掷，使基础模型承担可建模部分、网络补偿其不足。

建议重点读 III Method Overview、IV Learning Residual Physics for Throwing、VI Evaluation、VII Discussion and Future Work。适合参考残差模块的动机、与纯模型／纯学习方案的比较，以及不同物体和目标位置的实验安排。这里学习的是物理控制参数残差，不是动作块执行期间的历史条件反馈。

本地文件：13 页，页眉为 IEEE Transactions on Robotics；下载自 https://arxiv.org/pdf/1903.11239v3 。不要误用项目主页上较早的 RSS 会议版本来判断期刊篇幅与结构。

## 5 数学建模到算法与实验的衔接

**Motion Planning Diffusion: Learning and Adapting Robot Motion Planning With Diffusion Models**，Joao Carvalho 等，T-RO 41: 4881–4901, 2025。

[本地 PDF](05_Motion_Planning_Diffusion_TRO_2025.pdf) · [期刊 DOI](https://doi.org/10.1109/TRO.2025.3593109) · [作者版本 v3](https://arxiv.org/abs/2412.19948v3) · [作者机构发表记录](https://www.dfki.de/en/web/research/projects-and-publications/publication/16337)

学习轨迹分布先验，将其与任务代价引导结合，并使用 B-spline 表示轨迹。与当前任务的直接联系弱于前四篇，但适合学习“问题建模、表示选择、求解算法、对应实验”的写法。

建议重点读 III Robot Motion Planning with Diffusion Models、IV Experimental Evaluation，尤其是任务设置、数据和评价指标如何分开交代。它是学习辅助运动规划，不是常用的 Diffusion Policy，也不是你的 GRU 反馈方法。

本地文件：21 页；下载自 https://arxiv.org/pdf/2412.19948v3 。该公开版本仍有 Submitted to IEEE Transactions on Robotics 页眉；期刊身份、卷页和 DOI 已由作者机构页面及 Crossref 注册记录确认，不能将这个 PDF 称为出版社最终排版版。

## 对当前论文的阅读重点

先用前两篇对照你的任务动机、任务成功定义和阶段评价；用第三篇检查语义输入与其他模块的消融是否可归因；用第四篇检查残差学习的分工和泛化实验；第五篇主要参考数学建模与算法叙述的衔接。

这些论文的写作结构并不完全一致，也不是都依靠长篇定理证明。值得借鉴的是每个研究问题如何对应方法选择和验证证据，而不是模仿公式数量。当前三类物体均在示范中出现，分类别结果应说明表现一致性；未见位置的结果才对应该设置下的位置泛化。

## 核验说明

五个 DOI 的 Crossref 记录均返回期刊名称 IEEE Transactions on Robotics，卷、页和年份与上述条目一致。PDF 页数依次为 16、19、14、13、21。01 在全文文本提取时出现少量嵌入对象数字格式警告，但 PDF 可解析并能提取正文；保留原始下载文件，不重写论文内容。

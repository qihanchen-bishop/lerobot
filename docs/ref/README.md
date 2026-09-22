# 理论分析与方法参考文献

更新：2026-09-16。原有 8 篇，本轮新增 12 篇，共 20 个独立 PDF，按主要用途分类。跨类别的论文只保存一份。

- [C、D 小节规划](CD_SECTION_PLAN.md)：衔接 A、B 的核心观点、叙述顺序、数学条件与实验对应。
- [专题阅读笔记](CD_READING_NOTES.md)：核读位置、文献依据、与本方法的区别。
- [来源与校验清单](papers_manifest.json)：PDF 页数、SHA-256、来源、版本和解析警告。

本轮先整理参考资料与规划，随后按作者追加要求完成 C、D 的中文理论正文和补充证明，
并将待补验证写入实验章节：[正文 PDF](../draftv6/main.pdf)、
[补充证明](../draftv6/representation_feedback_proofs.tex)、
[实验协议](../draftv6/experiments.tex)。
笔记区分重点核读与背景参考，不表示逐页审核全部参考论文的证明。

## 01 模仿学习理论

目录：`01_imitation_theory/`。用于 B 的分布偏移、误差传播与任务后果分析。

| 编号 | 文献与版本 | 本地文件 | 原始来源 | 用途 |
|---|---|---|---|---|
| 01 | Ross et al., A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning, AISTATS 2011 | [PDF](01_imitation_theory/01_DAgger_Ross_AISTATS_2011.pdf) | [PMLR](https://proceedings.mlr.press/v15/ross11a.html) | DAgger；示范与部署分布 |
| 02 | Tu et al., On the Sample Complexity of Stability Constrained Imitation Learning, L4DC 2022 | [PDF](01_imitation_theory/02_Stability_Constrained_IL_Tu_L4DC_2022.pdf) | [PMLR](https://proceedings.mlr.press/v168/tu22a.html) | 增量稳定性与样本复杂度 |
| 03 | Pfrommer et al., TaSIL: Taylor Series Imitation Learning, NeurIPS 2022 | [PDF](01_imitation_theory/03_TaSIL_Pfrommer_NeurIPS_2022.pdf) | [NeurIPS](https://papers.neurips.cc/paper_files/paper/2022/hash/7f10c3d66c3b7863a9cda255dcac5bb7-Abstract-Conference.html) | 局部误差传播与专家导数匹配 |
| 08 | Schulman et al., Trust Region Policy Optimization, ICML 2015 | [PDF](01_imitation_theory/08_TRPO_Schulman_ICML_2015.pdf) | [PMLR](https://proceedings.mlr.press/v37/schulman15.html) | 性能差分与策略改进条件 |
| 19 | Cheng and Boots, Convergence of Value Aggregation for Imitation Learning, AISTATS 2018 | [PDF](01_imitation_theory/19_Value_Aggregation_Cheng_AISTATS_2018.pdf) | [PMLR](https://proceedings.mlr.press/v84/cheng18c.html) | B 已引用；价值聚合 |
| 20 | Ross and Bagnell, Reinforcement and Imitation Learning via Interactive No-Regret Learning, arXiv 2014 v1 | [PDF](01_imitation_theory/20_AggreVaTe_Ross_arXiv_2014_v1.pdf) | [arXiv](https://arxiv.org/abs/1406.5979v1) | AggreVaTe；错误的后续代价 |

这些工作不直接证明本文满足覆盖、稳定性或专家可查询条件。19、20 本轮补齐原文，主要服务已有 B。

## 02 任务相关表示

目录：`02_task_representations/`。用于 C 的归纳偏置、关系表示与任务进程。

| 编号 | 文献与版本 | 本地文件 | 原始来源 | 用途 |
|---|---|---|---|---|
| 04 | Zhang et al., Learning Invariant Representations for Reinforcement Learning without Reconstruction, ICLR 2021 | [PDF](02_task_representations/04_DBC_Zhang_ICLR_2021.pdf) | [arXiv](https://arxiv.org/abs/2006.10742) | DBC；行为相关而非像素相似 |
| 05 | de Haan et al., Causal Confusion in Imitation Learning, NeurIPS 2019 | [PDF](02_task_representations/05_Causal_Confusion_deHaan_NeurIPS_2019.pdf) | [NeurIPS](https://papers.neurips.cc/paper_files/paper/2019/hash/947018640bf36a2bb609d3557a285329-Abstract.html) | 错误相关性与额外输入的风险 |
| 09 | Zeng et al., Transporter Networks: Rearranging the Visual World for Robotic Manipulation, CoRL 2020 / PMLR 2021 | [PDF](02_task_representations/09_Transporter_Zeng_CoRL_2020.pdf) | [PMLR](https://proceedings.mlr.press/v155/zeng21a.html) | 空间结构先验 |
| 10 | Zhu et al., VIOLA: Imitation Learning for Vision-Based Manipulation with Object Proposal Priors, CoRL 2022 / PMLR 2023 | [PDF](02_task_representations/10_VIOLA_Zhu_CoRL_2022.pdf) | [PMLR](https://proceedings.mlr.press/v205/zhu23a.html) | 对象先验、上下文与历史 |
| 11 | Fu et al., Learning Task Informed Abstractions, ICML 2021 | [PDF](02_task_representations/11_TIA_Fu_ICML_2021.pdf) | [PMLR](https://proceedings.mlr.press/v139/fu21b.html) | TIA；相关与无关因素的分解条件 |
| 12 | Zakka et al., XIRL: Cross-embodiment Inverse Reinforcement Learning, CoRL 2021 / PMLR 2022 | [PDF](02_task_representations/12_XIRL_Zakka_CoRL_2021.pdf) | [PMLR](https://proceedings.mlr.press/v164/zakka22a.html) | 视觉进程、时间对应与目标距离 |

## 03 历史与信息状态

目录：`03_history_information/`。用于 D 的增量信息、历史压缩与错误捷径。

| 编号 | 文献与版本 | 本地文件 | 原始来源 | 用途 |
|---|---|---|---|---|
| 06 | Subramanian et al., Approximate Information State for Approximate Planning and Reinforcement Learning in Partially Observed Systems, JMLR 2022 | [PDF](03_history_information/06_AIS_Subramanian_JMLR_2022.pdf) | [JMLR](https://www.jmlr.org/papers/v23/20-1165.html) | AIS；有用历史压缩的条件 |
| 13 | Wen et al., Fighting Copycat Agents in Behavioral Cloning from Observation Histories, NeurIPS 2020 | [PDF](03_history_information/13_Fighting_Copycat_Wen_NeurIPS_2020.pdf) | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2020/hash/1b113258af3968aaf3969ca67e744ff8-Abstract.html) | 历史导致重复上一动作 |
| 14 | Chuang et al., Resolving Copycat Problems in Visual Imitation Learning via Residual Action Prediction, ECCV 2022 | [PDF](03_history_information/14_RAP_Chuang_ECCV_2022.pdf) | [ECVA](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136990386.pdf) | RAP；动作差分监督历史编码 |

## 04 动作块与执行反馈

目录：`04_chunk_feedback/`。用于 D 的直接方法对照、时延与槽位对齐。

| 编号 | 文献与版本 | 本地文件 | 原始来源 | 用途 |
|---|---|---|---|---|
| 15 | Zhao et al., Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware, RSS 2023 | [PDF](04_chunk_feedback/15_ACT_Zhao_RSS_2023.pdf) | [RSS](https://www.roboticsproceedings.org/rss19/p016.html) | ACT；动作块与时间集成 |
| 16 | Black et al., Real-Time Execution of Action Chunking Flow Policies, NeurIPS 2025 | [PDF](04_chunk_feedback/16_RTC_Black_NeurIPS_2025.pdf) | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/300ccb2187dedd4edcc07f7e76d8e553-Abstract-Conference.html) | RTC；已承诺前缀与异步衔接 |
| 17 | Sendai et al., Leave No Observation Behind: Real-time Correction for VLA Action Chunks, arXiv 2025 v1 | [PDF](04_chunk_feedback/17_A2C2_Sendai_arXiv_2025_v1.pdf) | [arXiv](https://arxiv.org/abs/2509.23224v1) | A2C2；计划条件实时残差，直接近邻 |
| 18 | Johannink et al., Residual Reinforcement Learning for Robot Control, arXiv 2018 v2；后发表于 ICRA 2019 | [PDF](04_chunk_feedback/18_Residual_RL_Johannink_arXiv_2018_v2.pdf) | [arXiv](https://arxiv.org/abs/1812.03201v2) | 传统控制与学习残差；在线 RL |

## 05 建模与论证写法

目录：`05_formulation_writing/`。

| 编号 | 文献与版本 | 本地文件 | 原始来源 | 用途 |
|---|---|---|---|---|
| 07 | Bobu et al., Quantifying Hypothesis Space Misspecification in Learning From Human–Robot Demonstrations and Physical Corrections, IEEE T-RO 2020 | [PDF](05_formulation_writing/07_Hypothesis_Misspecification_Bobu_TRO_2020.pdf) | [作者稿](https://arxiv.org/abs/2002.00941v2)、[DOI](https://doi.org/10.1109/TRO.2020.2971415) | 假设失效、分析、方法和实验的组织 |

现有资料保持原路径：[IEEE/T-RO 风格笔记](IEEE_TRO_FORMULATION_STYLE_NOTES.md)、[TaSIL 作者与阅读路线](TaSIL_AUTHORS_AND_READING_GUIDE.md)。

## 使用与验证

C 优先：VIOLA → Transporter → DBC → TIA → XIRL。D 优先：A2C2 → ACT / RTC → Copycat / RAP → AIS。正文只引用直接支持对应论点的文献，不必全部列入 C、D。

CoRL 的会议年和 PMLR 出版年分别列出；正式 BibTeX 采用所引版本的官方元数据。

20 个 PDF 的 `pdfinfo`、`pdftotext -layout` 均返回成功；未逐页检查排版。DAgger 有注释目标警告，DBC 有交叉引用重建提示，TIA 有字体权重警告；DBC 已另外检查第 3 页图像及相关正文，未修改 PDF 字节。

原有 8 个根目录 PDF 路径保留为相对符号链接，指向分类文件，既有引用仍有效。来源、页数、散列见校验清单；既有文件不推测 arXiv 修订号。

资料仅用于阅读与引用，版权归原权利人；公开获取不等于获准重新分发。本轮没有提交或同步 GitHub。

# 理论分析参考文献

下载日期：2026-09-10。共 8 篇，均已通过 `pdfinfo` 解析检查和文本提取检查。第 4、7 篇为 arXiv 作者公开版本，其余为会议或期刊官网 PDF。本文档中的阅读重点是针对当前工作的建议，不代表这些论文直接证明本方法有效。

## 文献与阅读重点

1. **A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning**
   - Ross 等，AISTATS 2011；9 页。
   - [本地 PDF](01_DAgger_Ross_AISTATS_2011.pdf)；[官方页面](https://proceedings.mlr.press/v15/ross11a.html)。
   - 关注专家分布与策略执行分布的差异、误差累积及 DAgger 的保证。不能将分类误差界直接套到 ACT 的连续动作 L1 误差。

2. **On the Sample Complexity of Stability Constrained Imitation Learning**
   - Tu 等，L4DC 2022；12 页。
   - [本地 PDF](02_Stability_Constrained_IL_Tu_L4DC_2022.pdf)；[官方页面](https://proceedings.mlr.press/v168/tu22a.html)。
   - 关注增量稳定性如何影响模仿学习的样本复杂度及时间跨度依赖。当前模型未显式满足其稳定性约束，不能直接继承保证。

3. **TaSIL: Taylor Series Imitation Learning**
   - Pfrommer 等，NeurIPS 2022；13 页。
   - [本地 PDF](03_TaSIL_Pfrommer_NeurIPS_2022.pdf)；[官方页面](https://papers.neurips.cc/paper_files/paper/2022/hash/7f10c3d66c3b7863a9cda255dcac5bb7-Abstract-Conference.html)。
   - 重点学习“分析误差传播条件，再设计训练目标，最后验证”的叙述方式。其策略导数匹配与我们的 QToken 几何监督不是同一个目标。

4. **Learning Invariant Representations for Reinforcement Learning without Reconstruction**
   - Zhang 等，ICLR 2021；16 页；DBC。
   - [本地 PDF](04_DBC_Zhang_ICLR_2021.pdf)；[arXiv 页面](https://arxiv.org/abs/2006.10742)。
   - 关注任务相关表示、无关视觉变化与 bisimulation。语义分割可以作为归纳偏置，但不自动满足该文的奖励和转移保持条件。

5. **Causal Confusion in Imitation Learning**
   - de Haan 等，NeurIPS 2019；12 页。
   - [本地 PDF](05_Causal_Confusion_deHaan_NeurIPS_2019.pdf)；[官方页面](https://papers.neurips.cc/paper_files/paper/2019/hash/947018640bf36a2bb609d3557a285329-Abstract.html)。
   - 关注为什么增加输入或历史不必然改善策略，以及伪相关与分布变化的关系。语义辅助监督本身不是因果识别保证。

6. **Approximate Information State for Approximate Planning and Reinforcement Learning in Partially Observed Systems**
   - Subramanian 等，JMLR 2022；83 页。
   - [本地 PDF](06_AIS_Subramanian_JMLR_2022.pdf)；[官方页面](https://www.jmlr.org/papers/v23/20-1165.html)。
   - 优先阅读第 2 至 4 节：历史压缩何时能保留决策所需信息，以及近似误差如何影响规划。普通 GRU 并不自动构成满足这些条件的近似信息状态。

7. **Quantifying Hypothesis Space Misspecification in Learning From Human–Robot Demonstrations and Physical Corrections**
   - Bobu 等，IEEE Transactions on Robotics 2020；作者接受稿 21 页。
   - [本地 PDF](07_Hypothesis_Misspecification_Bobu_TRO_2020.pdf)；[arXiv 页面](https://arxiv.org/abs/2002.00941v2)；[DOI](https://doi.org/10.1109/TRO.2020.2971415)。
   - 借鉴 TRO 写作结构：指出原有假设的失效，建立可分析模型，设计对应机制，再针对条件做实验。其不确定性定义不等同于分割质量分数。

8. **Trust Region Policy Optimization**
   - Schulman 等，ICML 2015；9 页；补充参考。
   - [本地 PDF](08_TRPO_Schulman_ICML_2015.pdf)；[官方页面](https://proceedings.mlr.press/v37/schulman15.html)。
   - 查阅 performance difference identity 及策略改进条件。使用这个恒等式不代表当前方法属于强化学习，也不自动获得 TRPO 的改进保证。

## 建议阅读顺序

先读 TaSIL、DBC、AIS、Bobu，分别关注分析驱动设计、任务相关表示、历史信息和 TRO 叙述结构。随后补齐 DAgger 与稳定性约束模仿学习，TRPO 用于核对性能差异公式。

每篇记录四项：使用什么假设、实际证明什么、方法如何对应分析、实验验证了哪些条件。将可借鉴的分析与当前模型尚未满足的假设分开，不将相关理论写成本方法已有的保证。

## 文件检查

8 个文件均可解析，页数依次为 9、12、13、16、12、83、21、9。DAgger PDF 在文本提取时有原文件注释链接目标警告，但正文可以提取；此警告不属于下载失败。

文件仅用于论文阅读与引用，版权归原权利人。下载作者公开稿不意味着可以重新分发或用于其他用途；未执行 GitHub 同步。

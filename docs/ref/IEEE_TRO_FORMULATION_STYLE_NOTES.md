# Formulation 的符号与假设写作核对

核对日期：2026-09-15。适用文件：docs/draftv6/formulation.tex。

## 协作约定

- 当前以中文稿为编辑基准；用户修改并确认中文后，才翻译为英文。
- 翻译不擅自改变假设强度、符号含义、研究范围或结论；数学问题另行指出和讨论。
- 区分期刊明确要求、相关论文的惯例与本文自己的建模选择。
- 本次核对不等于整篇稿件已经满足投稿要求。

## 官方依据

1. [T-RO Information for Authors](https://www.ieee-ras.org/publications/t-ro/t-ro-information-for-authors/)，Manuscript Preparation：使用 Transactions 双栏模板；Regular/Survey 使用 IEEEtran 的 journal 模式，默认 10 pt。官网鼓励首次投稿者参考近期论文的风格和技术水平。未在该页面发现必须使用某种集合字母或集合字体的规定。
2. [IEEE Editorial Style Manual for Authors](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE-Editorial-Style-Manual-for-Authors.pdf)，2024-07-29 版，Math / Displayed Equations：变量通常使用斜体，向量通常使用粗斜体；文字性缩写下标使用正体。Nomenclature 为可选部分，不是每篇文章都必须添加的符号表。
3. [IEEE Editing Mathematics](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/Editing-Mathematics.pdf)，2023-10-27 版，第 5 页 E 节：定理类标题有专门的层级样式，同样适用于 Lemmas、Hypotheses、Propositions、Definitions、Conditions 等；第 6 页说明向量在作者加以区分时通常使用粗体。公式应纳入句法，并保持编号和排版一致。

“所有集合必须用花体”“Assumption 必须逐节编号”“所有向量不加粗都不合规”均不是上述来源支持的结论。
本文使用花体表示主要空间，是作者的可读性约定，不是 IEEE 强制规定。当前保留后续理论采用的普通数学斜体状态和指令记号，不在单个小节中孤立改成粗体。全篇向量字体如需统一，应另作范围明确的排版修改。

## 相关论文的实际写法

| 论文与位置 | 可借鉴之处 | 不应直接移用的内容 |
|---|---|---|
| [Tu 等，L4DC 2022](https://proceedings.mlr.press/v168/tu22a.html)，第 2 节、Assumptions 4.1–4.2；[本地 PDF](02_Stability_Constrained_IL_Tu_L4DC_2022.pdf) | 系统方程旁给出状态和输入维数、策略的定义域和值域；假设明确对象、条件和量词，后续定理显式引用 | 控制仿射结构、稳定性和专家可实现性不是本文一般建模的已知事实 |
| [TaSIL，NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/7f10c3d66c3b7863a9cda255dcac5bb7-Paper-Conference.pdf)，第 2 节、Assumptions 3.1–3.2；[本地 PDF](03_TaSIL_Pfrommer_NeurIPS_2022.pdf) | 首次出现时解释状态、输入、动力学映射和扰动；假设内明确常数及适用范围，解释放在假设之外 | 状态可用性、专家导数和光滑性不能因引用该文而获得 |
| Bobu 等，T-RO 2020，第 III 节；[本地 PDF](07_Hypothesis_Misspecification_Bobu_TRO_2020.pdf) | 先说明人和机器人的作用，再引入输入集合、轨迹和代价函数；符号随叙述定义 | 该文研究目标推断，不是固定示范视觉行为克隆；这里只借鉴定义顺序 |

这些是相关理论与 T-RO 论文中的具体示例，不以“优秀论文都如此”替代逐篇核对。假设内容必须来自本研究所需条件，不能仅为形式上像理论论文而增设。

## 本次修正

- 删除无衬线的空间记号。它们不是数学错误，但本稿没有充分的采用理由。
- 明确定义状态空间、指令集合、各视角图像空间、联合图像空间、本体观测空间及完整观测空间。
- 指令变量 u 属于动作集合 A，并不要求两者首字母相同；关键是明确 A 的含义和维数。
- 完整观测空间用花体 Y，避免与后文代表信息集合的花体 O 重名；普通斜体 O 仅表示观测概率核。
- 显式说明时间、视角、示范索引、维数、轨迹长度、任务标识、参数空间及各函数的作用。
- 示范轨迹用 zeta，避免与后文表示执行时刻的 tau 重用；任务标识用 eta，避免与延迟 ell 重用。
- 区分完成指示变量与“该变量等于 1”的完成事件。
- 使用 IEEEtran 下的 newtheorem 环境，自动生成“假设 1（视觉可评估性）”及引用，不手工模拟样式。

## 数学边界

视觉可评估性是预先指定评价域中的理想可辨识性假设，不是分割准确性的事实，也不是收敛或成功率改进定理。若观测噪声使成功与失败不可区分，精确判别假设可能不成立。本次未擅自将其换成近似判别假设；是否引入允许错误率，需作者确认并同步调整论证。

评价域不能按测试结果事后筛选，不能因为某条轨迹不满足可见性条件就默认将其从成功率统计中剔除。经验风险公式是逐步模仿的建模示例，不宣称现有 ACT 训练严格采用该轨迹加权方式或找到了全局最优解。

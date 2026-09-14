# V6 中文期刊结构稿

基于 V5 重组，保留 V5 不变。采用 IEEEtran journal 双栏格式；这是中文工作稿，不代表已满足全部 TRO 投稿要求。

## 文件入口
- main.tex：论文入口，方法型短标题。
- main.pdf：构建产物。
- analysis.tex：第三章总入口，全部分析与证明在本章内，无独立证明附录。
- method.tex：第四章方法和在线执行算法。
- experiments.tex：第五章实验协议与结果占位。

## 六章结构
1. Introduction
2. Related Work
3. Problem Formulation and Theoretical Analysis
4. Method
5. Experimental Validation
6. Discussion and Conclusion

所有待补说明用 `\\draftnote{...}`，PDF 中呈现为中文括号包围的下划线文字。引言、相关工作、最终结果和结论不虚构补全。V5 的近似成功率表未作为最终结果带入。

第三章保留覆盖、任务敏感性、几何近似充分性、历史条件投影和时延对齐条件；证明从 V5 附录移入对应分析。理论平方风险与实现 Smooth L1/L1 损失明确区分。方法超参数沿用 V5，并标记最终配置核验项。

## 构建
当前使用独立 Conda 环境 `latex`（Tectonic 0.17.0），不依赖或修改机器人训练环境。
构建脚本优先使用现有 Tectonic/XeLaTeX，否则自动调用该 Conda 环境，无需手动激活。
```bash
bash docs/draftv6/build.sh
```
新机器可安装：
```bash
conda create -n latex -c conda-forge tectonic=0.17.0 -y
```
首次编译需要联网下载并缓存 TeX 资源。系统需提供 `Droid Sans Fallback` 中文字体及
`/usr/share/fonts/opentype/urw-base35/` 下的 NimbusRoman 字体；字体设置见 `main.tex`。
当前模板有字体初始化回退与第三方 `algorithmic.sty` 编码告警；已检查输出字体嵌入，
未发现正文缺字、溢出或未解析引用。

## 叙述主线
核心问题是在有限示范下，通过语义引导的任务关系表示与历史条件反馈改善动作块策略的任务完成表现。
表示与执行信息是两个互补缺口；计划连续性和动态响应只作为反馈机制的分析维度。
引言和摘要已按此主线展开，理论证明保留；实验新增表示×反馈交叉设计及示范规模协议。
30/60/90条规模、扰动对照与机制诊断均是待验证协议，不是新增实测结果。
固定90条示范下的改善不自动支持样本效率，额外语义资源必须披露。
本次叙述修改已重新编译到 `main.pdf`（9页），并抽查首页及实验对照表排版。

## 待完成
补齐独立评估日志、训练种子、试验次数及置信区间后再填写结果。新增参考文献需实际核对后引用，不能用文献题目推断其定理。发布前删除或解决所有 draftnote，并将中文工作稿转为英文。

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
```bash
bash docs/draftv6/build.sh
```

## 后续
补齐独立评估日志、训练种子、试验次数及置信区间后再填写结果。新增参考文献需实际核对后引用，不能用文献题目推断其定理。发布前删除或解决所有 draftnote，并将中文工作稿转为英文。

# 中文核心方法草稿 v2

主文件：`main.tex`

参考文献：`references.bib`

推荐使用 XeLaTeX 编译中文文档：

```bash
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

安装了 `latexmk` 时可直接运行：

```bash
latexmk -xelatex -interaction=nonstopmode main.tex
```

第二版在原草稿基础上明确拆分暴露、分离、运输、恢复和完成状态，并加入：

- 基于语义几何而非固定时间生成阶段监督；
- Phase、Progress、Event 和 Transition 四类输出头；
- Stay、Advance、Rollback 和 Uncertain 切换预测；
- 对 ACT 使用软阶段嵌入和残差 FiLM/条件注意力；
- 带滞回、最小驻留时间和动作块截断的受约束状态机；
- 失败恢复数据、质量加权监督和动作梯度隔离要求。

正文同时保留自动语义标注、标签质量评估、可变视角融合、语义预测 ACT 和图像空间视觉伺服。实验结果、作者信息、机构信息和最终超参数尚未填写。

没有本地 TeX Live 时，也可以使用 Tectonic：

```bash
tectonic -X compile main.tex
```

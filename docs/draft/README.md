# 中文核心方法草稿

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

正文聚焦自动语义标注、标签质量评估、自动阶段学习、可变视角语义预测 ACT 和图像空间视觉伺服。实验结果、作者信息、机构信息和最终超参数尚未填写。

没有本地 TeX Live 时，也可以使用 Tectonic：

```bash
tectonic -X compile main.tex
```

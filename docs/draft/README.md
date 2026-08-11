# LaTeX 理论初稿

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

正文当前是一份理论与方法初稿，实验结果、作者信息、机构信息和最终超参数尚未填写。

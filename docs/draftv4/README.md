# T-RO 中文理论与方法初稿

日期：2026-09-09。主文件 `main.tex`，IEEEtran journal 10pt双栏，中文由xeCJK支持。
这是投稿准备阶段的中文工作稿，不是可直接提交的英文终稿。
未填写作者身份；Introduction、Related Work、Experiments仅保留空标题，摘要和结论未添加。
相关工作章节留空不代表无需引用：理论基础和方法依赖的引用保留在正文及references.bib。

## 文件

- `theory.tex`：问题定义、假设、命题与证明、适用边界。
- `method.tex`：SAM2质量迭代、冻结U-Net、QToken、GRU残差及SF-v2时序。
- `references.bib`：基础参考文献，Related Work扩写时进一步补全。
- `main.pdf`：编译预览。

运行 `bash docs/draftv4/build.sh`。需要Tectonic或XeLaTeX，字体Droid Sans Fallback和Nimbus Roman。
当前显式使用 `/usr/share/fonts/opentype/urw-base35/` 内的OTF字体，避免Tectonic误选同名字体的Type1文件。迁移机器时调整main.tex的字体路径。
不修改IEEEtran栏宽、行距或页边距。中文字体仅便于讨论；最终英文稿删除xeCJK并依据投稿时官方要求核对匿名、页数和模板。
官方要求：https://www.ieee-ras.org/publications/t-ro/t-ro-information-for-authors/

## 理论审查重点

1. 语义由固定RGB映射产生，不凭空增加条件信息；没有伪造“语义必然降低Bayes风险”定理。
2. 几何风险界显式保留近似充分性、预测误差和动作读出误差。ACT并非纯几何控制器，最后一项不可省略。
3. 历史收益是嵌套信息集上的条件均值风险差；GRU压缩与拟合误差可能抵消收益，没有证明GRU架构必要。
4. 残差改进给出精确平方风险条件；有界投影是总体最优参照，不假设训练必达最优。
5. 质量加权推导包含选择偏差和标签噪声偏差；不把启发式分数称为逆方差或校准概率。
6. 延迟覆盖按同一计划、同一目标时间推导，明确首轮/跨计划/连续丢帧等边界。
7. 实际Smooth L1/MAE与理论MSE区分；离线专家分布不等于实际策略访问分布。
8. 标准概率论工具不宣称为原创定理；没有闭环稳定性、任务成功或“首次提出任务”的未核实主张。

方法依据上一版 `../draftv3/semantic_annotation_unet_qtoken_gru_methods_zh.md` 及现有实现核对。
SF-v2不是持续GRU隐藏状态，不做全局阶段硬切换，C的query来自已生效慢计划。
用户反馈的最新实机收益不在本稿实验章节代写，后续应按实际记录填入并检验理论条件。

## 本次检查

- Tectonic编译成功，PDF为5页IEEE双栏；正文29个编号公式。
- 已检查正文页面，未发现跨栏溢出或正文遮挡；日志无未定义引用、缺字或Overfull box。
- IEEEtran初始化阶段有TU/ptm字体回退提示；正文已显式指定Nimbus Roman OTF和中文字体，不影响本机PDF生成。这不是最终英文模板字体验收。
- 交叉引用无重复/缺失，三个要求留空的章节没有正文。
- 用数值样例检查残差风险恒等式、历史风险分解、有界投影，用枚举检查重叠覆盖条件；均通过。数值检查不替代证明，也不验证任务假设。
- 没有修改策略代码、训练结果或旧draft；未同步GitHub。

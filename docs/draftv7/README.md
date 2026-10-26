# draftv7：中文实验论述与英文图表

本稿由 draftv6 复制后修改；本次没有改动 draftv6，也未同步 Git。

- 阅读入口：[main.pdf](main.pdf)，主文件：[main.tex](main.tex)。
- 实验正文及评估附录保留中文；实验图表、表头和图题为英文。
- 策略结果截至 2026-10-23 补采完成；2026-10-26补入finaldata标注迭代记录。
- 理论源文件与35个编号理论公式保持不变。方法已同步硬语义图、双视角读出与质量损失归一化。
- 原 v7 已去掉历史反馈分支；旧完整稿保存在 draftv6。

## 章节与图表

实验按标注迭代、整体表现、模块消融、物体与位置变化、执行瓶颈与时间、理论联系组织。
统计细节、实现配置、失败计数和完整条件结果放在 evaluation_details.tex。

五方法共480次有效试验，每方法96次；四类物体、12位置、每条件两次。
纳入ACT、DP、RGBQtoken、segNoQtoken和viewQtoken，不包含VIOLA。
已有人工修正与无效中止替换记录保留，不重新筛选结果。

figures/results_20261023 包含五组英文 PDF/PNG：

1. overall_objects_en：总体区间及四类物体，附录。
2. ablation_en：语义输入与监督查询四格对照。
3. position_en：同位置跨物体汇总。
4. time_failure_en：累计完成与失败阶段。
5. object_position_en：完整物体--位置矩阵，附录。

旧中文图与生成器保留，但当前论文不引用这些中文图。
英文图表由 results 中冻结的 CSV/JSON 生成，无需模型权重或视频。
来源与产物哈希保存在 results/english_assets_manifest.json。

figures/annotation_finaldata/annotation_iteration_en 展示R0--R3的固定保留视频异常率及伪标签扩充。
results/annotation_finaldata 保存SAM2来源哈希、逐视频诊断、像素混淆矩阵和模型谱系的精简快照。
快照核对两视角R3与finaldata_seg/models导出模型一致；不重新推理，不修改SAM2工程。
保留视频没有人工像素真值，种子参考不是独立测试集。图中异常率是自动诊断，不是人工错误率。

## 重建与验证

在仓库根目录执行：

~~~bash
.cache/viola/venv/bin/python docs/draftv7/build_experiment_assets_en.py
.cache/viola/venv/bin/python docs/draftv7/build_annotation_assets.py
bash docs/draftv7/build.sh
python3 docs/draftv7/verify_draft.py
~~~

绘图仅需 numpy 和 matplotlib，使用本机已有环境，不修改策略环境。
验证默认检查冻结数据、图表哈希、理论保留、引用及编译日志，显式报告原机器数据缺失。
加 --require-originals 后严格要求原始 JSONL 可访问；本机缺少这些其他机器路径，不能声称完成原始视频复审。
v6 源文件和理论仍严格核对，包含运行路径的 .log/.blg 等易变构建记录不作为源文件完整性判据。
标注生成器加 --refresh 可重新读取SAM2记录并校验源报告与导出模型，默认只使用冻结快照。

## 审阅入口与待定事项

- [写作参考、理论对应与补充验证建议](experiment_revision_notes.md)。
- [方法实现核对与尚未验证部分](method_implementation_notes.md)。
- [验证记录](verification_zh.md)。

未扩写摘要、相关工作、引言中的仿真表述、讨论中的第二任务及案例占位。
当前新增证据只覆盖所报告的一项实机任务，整稿定稿时需统一这些部分。
成功率支持方法在所测任务上的有效性，不直接证明覆盖、敏感性或专家选择假设。
本轮未新增跨训练种子实验、独立像素真值测试、质量筛选消融或逐查询消融。

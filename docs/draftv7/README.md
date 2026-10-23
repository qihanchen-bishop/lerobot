# draftv7：保留 v6 英文正文，更新中文实验

本稿由 `../draftv6` 复制后修改，原版 draftv6 未改动。

- 阅读入口：[main.pdf](main.pdf)；LaTeX 主文件：[main.tex](main.tex)。
- 实验正文及评估附录使用中文，数据截至 2026-10-23 16:26 补采完成。
- 非实验章节保留 v6 的原文与语言，仅删除残差分支、历史条件残差反馈及其依赖表述。v6 原有的中文注释、讨论占位和空摘要也保持原有状态。
- 建模、动作分析、表示分析、动作分析证明、参考文献库和编译脚本与 v6 字节一致。
- 残差理论和证明不参与编译；方法框架图重新绘为不含残差分支的英文图。完整旧内容仍在 `../draftv6/`，供下一篇使用。
- 独立结果表格：[../experiment_results_20261023/README.md](../experiment_results_20261023/README.md)。

## 实验数据和图片

五种方法共480次有效实机试验，每方法96次，四类物体 × 12位置 × 每条件两次。
纳入ACT、DP、RGBQtoken、segNoQtoken、viewQtoken；未纳入VIOLA。
segNoQtoken方块的边界记录按人工确认失败计入；纸球无效中止已替换为正常执行的补采失败记录。

`experiments.tex` 包含总体、类别、位置、消融、重复表现、时间、失败阶段和推理开销。
`evaluation_details.tex` 说明有效记录、标签修正及统计口径。
`figures/results_20261023/` 包含五组中文 PDF/PNG 图：

1. `overall_objects_zh`：总体及四类物体；
2. `ablation_zh`：语义输入与查询监督的四格对照；
3. `position_zh`：同位置跨物体汇总；
4. `time_failure_zh`：累计完成与失败阶段；
5. `object_position_zh`：完整物体—位置矩阵。

图表及动态表格来自 `results/` 的冻结CSV/JSON。原始统计脚本与已执行notebook位于项目本地的 `result/final_five_method_comparison_20261023/`（该原目录不随此次提交上传）。
本稿携带冻结的逐试验CSV、汇总、检验值、人工复核和验证结果，可独立重建图表。

```bash
/home/romilab/.cache/codex/eval-analysis-venv/bin/python docs/draftv7/build_experiment_assets.py
python docs/draftv7/build_result_tables.py
bash docs/draftv7/build.sh
python docs/draftv7/verify_draft.py
```

`verify_draft.py` 的原始记录哈希检查依赖本机原始数据路径；CSV、图表和表格重建不依赖图像帧或模型权重。
编译沿用独立latex环境，图表使用独立分析环境。

## 保留原稿带来的待定事项

遵照本次范围，方法正文未改为最终checkpoint的实现描述。最终测试实际使用硬语义图、三个共享查询的双视角八维读出、目标像素到固定区域的最大距离，以及按几何有效坐标数归一化的辅助损失；这些差异已在实验章节“本轮测试的实现配置”集中说明。

v6引言中的仿真表述、讨论中的第二任务及失败案例占位未扩写为新结果。当前新增证据仅覆盖所报告的一项实机任务。摘要、相关工作占位、讨论定稿及方法与最终实现的统一仍应后续单独修订。

本文保留条件性理论，实机成功率不直接验证覆盖、动作敏感性或专家选择假设。
未新增跨训练种子实验、独立分割精度评估或逐查询消融；参考文献库本次未重新做全库出版信息核验。

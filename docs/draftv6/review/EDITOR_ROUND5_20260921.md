# 第五轮交审：真实空CSV及分层统计口径

## 完成

`results_templates/`新增七份仅表头CSV，没有假数据、默认零值或虚构方法结果：

1. trial_results.csv：单次试验、原始结果、配对区组、训练版本和来源。
2. main_summary.csv：逐checkpoint逐条件或声明层级的加权汇总。
3. representation_ablation.csv：关系/视角/可见性分层的表示诊断。
4. history_ablation.csv：固定回放和槽位上的历史/当前信息诊断。
5. runtime_summary.csv：任务与事件分层的端到端和槽位采用统计。
6. budget_summary.csv：真实训练预算、额外监督资源及主结果关联。
7. stratum_weights.csv：跨条件/运行的预定归一化权重。

README逐类解释字段、行粒度、关联、单位、缺失值、分母和指标边界。全部复用原trial_id及summary_id，不新增重复实验。运行及原始日志清单是数据交付要求，不新写分析程序。

正文主表改为seed/checkpoint逐层填写；Wilson明确限定同checkpoint、同条件独立可交换重复。跨条件与种子汇总需预定权重和分层区间，不能对合并总数直接套Wilson。模板另指出同基础下不同残差种子具有嵌套依赖。

## 验证

- awk检查七份CSV只有一行表头，无空列或重复列名，均通过。
- git diff --check通过。
- 现有build.sh编译成功；PDF17页。
- 无Overfull、未解析引用或缺字告警；既有字体与algorithmic编码告警仍在。
- 已查看第11页主表：seed/checkpoint标题完整，无溢出；表格位于讨论/附录之前。

请复审CSV可填写性及分层统计口径。配置、真实结果、实景素材、最终成功判据仍待作者；没有运行硬件或修改训练代码。

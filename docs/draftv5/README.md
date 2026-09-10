# Draft v5：从模仿误差到任务完成

论文标题：**Semantic-Guided Imitation Learning with History-Conditioned Feedback: Analysis, Framework, and Experimental Validation**

中文对应：**结合历史条件反馈的语义引导模仿学习：理论分析、方法框架与实验验证**。

中文 IEEEtran 工作稿，面向 T-RO 稿件准备。保留 draftv3、draftv4；未修改训练或机器人代码。

## 文件

- [main.pdf](main.pdf)：编译后的完整稿。
- [argument_chain_zh.md](argument_chain_zh.md)：理想保证、现实缺口、语义关系监督与历史纠偏之间的逐步连接。
- [main.tex](main.tex)：入口，正文分文件组织。
- [viewpoint_and_sources_zh.md](viewpoint_and_sources_zh.md)：先修正观点，再解释文献依据及推导边界。
- [validation_plan_zh.md](validation_plan_zh.md)：理论条件、可测证据、反例和对照实验。
- [check_draft.py](check_draft.py)：公式数值自检、引用完整性与排版错误检查。
- [evidence_to_experiments_zh.md](evidence_to_experiments_zh.md)：最新初步结果、抽象任务定义和优先补充对照。
- 理论任务定义在 `task.tex`，具体双臂判据在实验部分的 `task_instance.tex`，两者分开。

## 主线

1. 任务成功是目标，动作模仿损失是代理目标，问题定义不预设快慢结构。
2. 泛化误差、部署覆盖和任务敏感性共同决定动作误差能否支持成功率保证。
3. 同样来自 RGB 的语义图不增加原始观测信息，而是借助外部监督提供归纳偏置。
4. 几何监督 query 要求任务关系可读出，但可读不等于被动作使用，更不等于完整阶段判断。
5. 最新观测与历史可能带来可预测的残差；收益需超过编码、拟合、延迟的代价。
6. Temporal ensemble 也有新观测。区别是预测混合与计划条件修正，不是开环与闭环的二分。

## 编译和检查

```bash
bash docs/draftv5/build.sh
python3 docs/draftv5/check_draft.py
```

依赖当前机器的 Tectonic（或 XeLaTeX）、IEEEtran、xeCJK、NimbusRoman OTF 和 Droid Sans Fallback。
已有字体路径警告意味着换机器须配置字体；不是任意环境可复现的容器构建。

正文方法规格继承 draftv4/v2 的实现审计记录；本轮额外核对了本地 ACT temporal ensemble 的实际加权代码。
未重新审计所有部署分支、重算实机成功率或测量硬件延迟。
实验章节现已记录用户提供的近似成功率和位置划分，但没有推定样本数或显著性。
布料复原是布料占整幅参考画面的面积比例，不是对目标物体的遮挡比例。
数学自检不是形式化证明，也不证明真实机器人满足文中的覆盖、平滑或任务充分性假设。

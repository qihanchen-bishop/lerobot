# SF-v2-S6-D3-H9：A/B/C 重训练

复用v1三套冻结基础策略和bettersetup_v5数据，单独重建缓存并从零训练残差模块，不覆盖v1。

- ACT chunk=60，slow_stride=30，slow_delay=3。
- fast_stride=6：5Hz更新；fast_delay=3：目标从观察后100ms开始。
- correction_steps=9：每次预测9步，相邻结果重叠3步，最新有效结果覆盖相同目标槽位，不累加。
- plan_steps=9：控制输入包含全部9个对应基础动作，避免输出后3步没有对应计划输入。
- history=8：固定特征窗口约覆盖1.4秒，不是持续隐藏状态GRU。
- history_gap=0.3秒：正常0.2秒更新不清空历史；明显断流时清空。
- 接收结果仍检查观察年龄<=0.15秒；已接受残差可使用至观察后0.4秒，覆盖最晚11/30秒的目标。计划切换仍清空旧残差。
- seed=1000，同一72/18 episode划分，优化器、损失、限幅和早停策略同v1；A保留MLP对照。

观察第6帧对应目标第9–17帧，第12帧对应第15–23帧。慢计划切换后，旧计划的更远残差作废。
只进行离线训练和shadow验证，不执行机器人。历史跨度、目标长度和控制输入维度均变化，不把结果当作仅更新频率的严格单变量消融。

## 运行

```bash
tmux new-session -d -s sf-v2-s6-d3-h9-abc -c /home/qihan/data/lerobot \
  'PYTHON=/home/qihan/miniconda3/envs/lerobot/bin/python bash run_slow_fast_residual_v2_queue.sh'
```

队列A→验收→B→验收→C→验收，失败即停。每阶段运行缓存、smoke、恢复、正式训练、shadow和audit。
模型与日志：`outputs/train/residual/SF-v2-S6-D3-H9-{A,B,C}-FS/`。
缓存：`outputs/cache/slow_fast_v2_s6_d3_h9_{A,B,C}/`。
总日志：`outputs/train/residual/SF-v2-S6-D3-H9.queue.log`；结束后总退出码为同前缀`.queue.exit_code`。
核对进程确认中断后可用`RESUME=1`重启队列。单阶段验收使用`PROFILE=v2-s6-d3-h9 bash run_slow_fast_residual_stage.sh A audit`。

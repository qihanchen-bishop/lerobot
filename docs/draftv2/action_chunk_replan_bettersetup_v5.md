# bettersetup_v5 action chunk replan 分析

## 技术摘要

- `bettersetup_v5` 有 90 条 episode、53,723 帧；按 30 Hz 估算，中位 episode 时长约 19.28 s。
- 固定 60 action 并不等价于固定任务进度：60 步终点相对当前 action 的单关节变化中位数为 38.53 deg，p95 为 81.78 deg。
- 数据支持把 replan 条件从“固定执行 K 步”扩展为“固定步数 + 运动预算 + 跟踪误差 + 语义阶段切换”。
- SO101 末端空间统计使用公开 URDF 的串联链 FK；图像里的 `tool` 只作为视觉语义，不参与 FK。

## 不同步长下的关节变化

| K steps | final max-joint p50 | final max-joint p95 | final max-joint p99 | cumulative L2 path p50 | cumulative L2 path p95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.20 | 2.99 | 3.90 |  |  |
| 4 | 4.70 | 11.42 | 14.99 | 6.17 | 15.23 |
| 8 | 8.98 | 21.28 | 27.19 | 12.67 | 29.51 |
| 16 | 16.21 | 36.36 | 44.82 | 26.39 | 55.25 |
| 24 | 21.99 | 47.25 | 59.25 | 40.87 | 79.27 |
| 60 | 38.53 | 81.78 | 103.15 | 107.41 | 181.47 |

## Motion budget 触发会怎样

下面的 K 表示从当前 action 开始，累计运动首次超过阈值时已经执行了多少帧；如果 60 帧内都没超过，则记为 60。

| Final max-joint budget | median first K | p75 first K | p90 first K | <=4 steps | never within 60 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 deg | 7 | 12 | 25 | 22.2% | 4.1% |
| 12 deg | 11 | 19 | 38 | 4.3% | 6.4% |
| 16 deg | 15 | 26 | 53 | 0.7% | 8.9% |
| 20 deg | 20 | 36 | 60 | 0.1% | 12.4% |
| 30 deg | 35 | 60 | 60 | 0.0% | 27.2% |
| 40 deg | 57 | 60 | 60 | 0.0% | 48.2% |

| Cumulative L2 path budget | median first K | p75 first K | p90 first K | <=4 steps | never within 60 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 deg | 6 | 10 | 18 | 27.2% | 3.2% |
| 15 deg | 9 | 14 | 24 | 6.0% | 4.4% |
| 20 deg | 12 | 18 | 30 | 0.6% | 5.3% |
| 30 deg | 18 | 26 | 41 | 0.0% | 6.6% |
| 45 deg | 26 | 37 | 58 | 0.0% | 9.6% |
| 60 deg | 34 | 48 | 60 | 0.0% | 15.2% |

## 末端空间变化

| Arm | 1-step translation p50 | 1-step translation p95 | 60-step translation p50 | 60-step translation p95 | 60-step rotation p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| left | 0.3 mm | 4.5 mm | 52.3 mm | 137.6 mm | 29.9 deg |
| right | 3.0 mm | 8.0 mm | 94.0 mm | 212.1 mm | 46.6 deg |


## 判断

只要机械臂运动幅度过大就重新规划，这个想法是合理的，但它应该作为 motion budget 或安全触发条件，而不是替代闭环判断。大动作可能是正确任务动作，例如快速 uncover 或 transport；此时 replan 未必解决问题，反而可能造成策略抖动。

更合适的在线逻辑是：固定最小 replan 间隔仍保留；当已执行 action 的累计关节变化或末端位移超过预算时提前 replan；当当前关节状态和旧 chunk 计划位置偏差过大时提前 replan；当语义阶段变化时立刻 replan。安装新 chunk 时继续使用 overlap fusion，避免硬切换带来的动作跳变。

可以先试的保守阈值是：关节空间使用 `max_joint_final_displacement >= 16-20 deg` 或 `cumulative_joint_L2_path >= 20-30 deg`；末端空间使用左臂 `20-30 mm`、右臂 `40-60 mm` 的累计平移预算。这些阈值来自示范分布的中位到 p75 附近，目标是让快动作比固定 24/60 步更早重规划，同时避免 4 步内频繁抖动。

SO101 的在线版本应把左右 5 维关节 action 分别做 FK，统计末端平移、旋转和速度。图像里的 `tool` 是视觉标注对象，可以用于语义阶段和接触关系判断，但不应替代真实末端运动学。

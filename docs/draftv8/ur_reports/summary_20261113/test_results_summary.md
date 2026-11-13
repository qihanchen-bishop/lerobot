# 双 UR 策略测试结果汇总

生成时间：2026-11-13T16:51:31+08:00。来源：`/home/romilab/Projects/gello-ur/eval/mergedata`。

纳入 60 条已人工确认测试；成功 38 条。另有 7 个已丢弃目录和 4 个无确认结果的残留目录未纳入；这 4 个目录与已丢弃运行同名，仅含复位记录，不是额外 4 次独立测试。原始数据未修改。

## 统计口径

- 按策略目录、任务代码分组；最终人工确认的 `evaluation_result.json` 为准，与运行元数据及 `eval_results.jsonl` 交叉核对。每个运行目录仅统计一次。
- 成功率 = 成功次数 / 已确认测试次数。未测试显示“—”，不是 0% 。`b/b1` 没有已确认测试。
- 完成时间仅对成功测试统计：执行时间为策略开始至停止，不含复位；全流程时间为开始至复位结束。失败耗时单独列在明细，不混入成功完成时间。
- 缺少 `policy_duration_s` 的失败记录采用其保留记录时长，明细标明字段来源；没有完整复位时间的记录不推算总时长。
- 视频可能裁掉复位段，因此录像时长与实际全流程时间不同。所有时间单位为秒；多选失败原因分别计数，原因合计不一定等于失败次数。
- 保存结果中的旧 `status=awaiting_result` 字段可能未更新；本次使用结果文件、`result_saved_at` 和索引确认，不误判为待确认。

## 按策略汇总

| 策略 | 任务 | 成功次数 | 测试次数 | 失败次数 | 成功率 | 成功平均执行时间(s) | 成功执行中位数(s) | 成功执行范围(s) | 成功平均全流程时间(s) | 失败原因及次数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| merge_Qtoken_RGBD | 全部 | 19 | 20 | 1 | 95.0% | 93.12 | 89.25 | 78.79–129.22 | 102.29 | 超时 1 |
| merge_rgbd_act | 全部 | 8 | 20 | 12 | 40.0% | 89.33 | 88.88 | 76.42–100.88 | 97.35 | 夹取掉落 1；超出限幅 11 |
| merge_rgbd_dp | 全部 | 11 | 20 | 9 | 55.0% | 109.62 | 110.01 | 93.34–121.54 | 114.10 | 夹取失败 4；超出限幅 4；超时 1 |
| 全部策略 | 全部 | 38 | 60 | 22 | 63.3% | 97.10 | 94.29 | 76.42–129.22 | 104.67 | 夹取掉落 1；夹取失败 4；超出限幅 15；超时 2 |

## 按策略 × 任务汇总

| 策略 | 任务 | 成功次数 | 测试次数 | 失败次数 | 成功率 | 成功平均执行时间(s) | 成功执行中位数(s) | 成功执行范围(s) | 成功平均全流程时间(s) | 失败原因及次数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| merge_Qtoken_RGBD | o | 5 | 5 | 0 | 100.0% | 88.50 | 83.52 | 79.76–108.50 | 98.22 | — |
| merge_Qtoken_RGBD | g | 5 | 5 | 0 | 100.0% | 93.79 | 94.97 | 84.22–103.25 | 103.30 | — |
| merge_Qtoken_RGBD | b | 0 | 0 | 0 | — | — | — | — | — | — |
| merge_Qtoken_RGBD | o1 | 5 | 5 | 0 | 100.0% | 88.97 | 80.94 | 78.79–112.26 | 97.93 | — |
| merge_Qtoken_RGBD | g1 | 4 | 5 | 1 | 80.0% | 103.22 | 100.32 | 83.04–129.22 | 111.57 | 超时 1 |
| merge_Qtoken_RGBD | b1 | 0 | 0 | 0 | — | — | — | — | — | — |
| merge_rgbd_act | o | 2 | 5 | 3 | 40.0% | 94.45 | 94.45 | 88.02–100.88 | 102.26 | 超出限幅 3 |
| merge_rgbd_act | g | 1 | 5 | 4 | 20.0% | 90.63 | 90.63 | 90.63–90.63 | 98.10 | 超出限幅 4 |
| merge_rgbd_act | b | 0 | 0 | 0 | — | — | — | — | — | — |
| merge_rgbd_act | o1 | 2 | 5 | 3 | 40.0% | 88.88 | 88.88 | 88.17–89.60 | 96.55 | 夹取掉落 1；超出限幅 2 |
| merge_rgbd_act | g1 | 3 | 5 | 2 | 60.0% | 85.78 | 82.52 | 76.42–98.41 | 94.35 | 超出限幅 2 |
| merge_rgbd_act | b1 | 0 | 0 | 0 | — | — | — | — | — | — |
| merge_rgbd_dp | o | 3 | 5 | 2 | 60.0% | 112.81 | 113.14 | 104.13–121.17 | 116.65 | 夹取失败 1；超时 1 |
| merge_rgbd_dp | g | 2 | 5 | 3 | 40.0% | 94.31 | 94.31 | 93.34–95.28 | 98.36 | 夹取失败 2；超出限幅 1 |
| merge_rgbd_dp | b | 0 | 0 | 0 | — | — | — | — | — | — |
| merge_rgbd_dp | o1 | 4 | 5 | 1 | 80.0% | 112.14 | 111.17 | 104.69–121.54 | 117.21 | 超出限幅 1 |
| merge_rgbd_dp | g1 | 2 | 5 | 3 | 40.0% | 115.09 | 115.09 | 110.01–120.17 | 119.78 | 夹取失败 1；超出限幅 2 |
| merge_rgbd_dp | b1 | 0 | 0 | 0 | — | — | — | — | — | — |

## 失败原因

| 策略 | 人工确认失败原因 | 次数 |
| --- | --- | --- |
| merge_Qtoken_RGBD | 超时 | 1 |
| merge_rgbd_act | 夹取掉落 | 1 |
| merge_rgbd_act | 超出限幅 | 11 |
| merge_rgbd_dp | 夹取失败 | 4 |
| merge_rgbd_dp | 超出限幅 | 4 |
| merge_rgbd_dp | 超时 | 1 |

## 可比性与数据说明

- 这是当前已确认测试的描述性结果，不是随机对照或显著性检验；每个已测试任务、每个策略仅 5 次。任务难度、运行顺序与人工干预可能影响比较。
- 三种策略均选择了右夹爪人工兜底模式；“选中”不代表每轮实际使用。不能仅凭最终标签宣称完全自主成功率。
- 限幅模式记录：merge_Qtoken_RGBD: {'clamp': 20}；merge_rgbd_act: {'clamp': 19, 'stop': 1}；merge_rgbd_dp: {'clamp': 20}。超出限幅是人工保存的失败标签，不等同于每条都有控制器保护停机。
- 各策略复位耗时不同，比较策略完成速度优先看“不含复位”的成功执行时间，不用录像长度替代。
- `run_20261112_145947_449426` 已按人工确认纠正为成功；其约 10.14 秒复位录像此前被裁掉，未恢复。实际全流程时间仍有记录，报告没有伪造视频长度。

## 每轮明细

| 策略 | 任务 | 结果 | 开始时间 | 运行目录 | 策略执行/失败耗时(s) | 耗时字段来源 | 全流程时间(s) | 复位时间(s) | 已保留录像时长(s) | 失败原因 | 停止原因 | 错误 | 复位录像保留 | 复位录像已裁掉 | 历史标签修正 | 限幅模式 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| merge_Qtoken_RGBD | o1 | 成功 | 2026-11-12T13:51:19.307+08:00 | run_20261112_135119_302116 | 93.61 | policy_duration_s | 103.53 | 9.92 | 103.53 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_Qtoken_RGBD | o1 | 成功 | 2026-11-12T14:08:42.122+08:00 | run_20261112_140842_122029 | 79.27 | policy_duration_s | 86.88 | 7.62 | 86.88 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_Qtoken_RGBD | o1 | 成功 | 2026-11-12T14:12:15.569+08:00 | run_20261112_141215_569827 | 78.79 | policy_duration_s | 86.85 | 8.06 | 86.85 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_Qtoken_RGBD | o1 | 成功 | 2026-11-12T14:17:32.224+08:00 | run_20261112_141732_224461 | 80.94 | policy_duration_s | 88.87 | 7.93 | 88.87 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_Qtoken_RGBD | o1 | 成功 | 2026-11-12T14:20:24.690+08:00 | run_20261112_142024_690183 | 112.26 | policy_duration_s | 123.52 | 11.26 | 123.52 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_Qtoken_RGBD | g1 | 成功 | 2026-11-12T14:26:19.179+08:00 | run_20261112_142619_179373 | 86.16 | policy_duration_s | 94.17 | 8.01 | 94.17 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_Qtoken_RGBD | o | 成功 | 2026-11-12T14:37:45.664+08:00 | run_20261112_143745_664670 | 89.25 | policy_duration_s | 99.67 | 10.42 | 99.68 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_Qtoken_RGBD | g | 成功 | 2026-11-12T14:40:47.538+08:00 | run_20261112_144047_538415 | 94.97 | policy_duration_s | 104.86 | 9.89 | 104.86 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_Qtoken_RGBD | g | 成功 | 2026-11-12T14:53:49.797+08:00 | run_20261112_145349_797300 | 103.25 | policy_duration_s | 110.89 | 7.64 | 110.89 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_Qtoken_RGBD | g | 成功 | 2026-11-12T14:56:49.340+08:00 | run_20261112_145649_340054 | 96.74 | policy_duration_s | 106.95 | 10.21 | 106.95 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_Qtoken_RGBD | g | 成功 | 2026-11-12T14:59:47.449+08:00 | run_20261112_145947_449426 | 89.78 | policy_duration_s | 99.91 | 10.14 | 89.78 | — | organ_coverage_success | — | False | True | True | clamp |
| merge_Qtoken_RGBD | g | 成功 | 2026-11-12T15:11:14.440+08:00 | run_20261112_151114_439972 | 84.22 | policy_duration_s | 93.89 | 9.68 | 93.90 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_Qtoken_RGBD | o | 成功 | 2026-11-13T10:30:26.380+08:00 | run_20261113_103026_380695 | 81.46 | policy_duration_s | 90.65 | 9.19 | 90.65 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_Qtoken_RGBD | o | 成功 | 2026-11-13T10:32:59.040+08:00 | run_20261113_103259_040245 | 83.52 | policy_duration_s | 92.64 | 9.13 | 92.65 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_Qtoken_RGBD | o | 成功 | 2026-11-13T10:35:32.148+08:00 | run_20261113_103532_148743 | 79.76 | policy_duration_s | 89.61 | 9.85 | 89.61 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_Qtoken_RGBD | o | 成功 | 2026-11-13T10:38:19.531+08:00 | run_20261113_103819_531430 | 108.50 | policy_duration_s | 118.53 | 10.04 | 118.53 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_Qtoken_RGBD | g1 | 成功 | 2026-11-13T10:41:53.643+08:00 | run_20261113_104153_643204 | 129.22 | policy_duration_s | 138.76 | 9.54 | 138.76 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_Qtoken_RGBD | g1 | 失败 | 2026-11-13T10:45:24.156+08:00 | run_20261113_104524_156009 | 180.05 | policy_duration_s | 205.59 | 25.53 | 180.05 | 超时 | maximum_run_time | — | False | True | False | clamp |
| merge_Qtoken_RGBD | g1 | 成功 | 2026-11-13T10:52:01.432+08:00 | run_20261113_105201_431977 | 114.48 | policy_duration_s | 122.29 | 7.81 | 122.30 | — | gui_stop_button | — | True | False | False | clamp |
| merge_Qtoken_RGBD | g1 | 成功 | 2026-11-13T10:56:36.914+08:00 | run_20261113_105636_914832 | 83.04 | policy_duration_s | 91.07 | 8.02 | 91.07 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_rgbd_act | g1 | 成功 | 2026-11-13T11:04:21.380+08:00 | run_20261113_110421_380372 | 82.52 | policy_duration_s | 91.33 | 8.82 | 91.34 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_rgbd_act | g1 | 成功 | 2026-11-13T11:06:52.257+08:00 | run_20261113_110652_257131 | 76.42 | policy_duration_s | 84.29 | 7.87 | 84.29 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_rgbd_act | g1 | 失败 | 2026-11-13T11:09:44.534+08:00 | run_20261113_110944_534447 | 27.57 | duration_s（无策略停止时间字段） | — | — | 27.57 | 超出限幅 | manual_failure_stop | — | False | False | False | clamp |
| merge_rgbd_act | g1 | 失败 | 2026-11-13T11:11:05.987+08:00 | run_20261113_111105_987824 | 15.49 | duration_s（无策略停止时间字段） | — | — | 15.49 | 超出限幅 | tcp_workspace_exceeded | TCP workspace exceeded: left Z=100.070 mm, allowed [45, 100] mm. Both arms stopped; automatic reset disabled. | False | True | False | stop |
| merge_rgbd_act | g1 | 成功 | 2026-11-13T11:12:58.701+08:00 | run_20261113_111258_701398 | 98.41 | policy_duration_s | 107.44 | 9.04 | 107.44 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_rgbd_act | o1 | 失败 | 2026-11-13T11:15:59.099+08:00 | run_20261113_111559_099361 | 73.71 | duration_s（无策略停止时间字段） | — | — | 73.71 | 夹取掉落 | manual_failure_stop | — | False | False | False | clamp |
| merge_rgbd_act | o1 | 成功 | 2026-11-13T11:18:43.942+08:00 | run_20261113_111843_942596 | 89.60 | policy_duration_s | 97.39 | 7.79 | 97.39 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_rgbd_act | o1 | 成功 | 2026-11-13T11:21:14.285+08:00 | run_20261113_112114_285560 | 88.17 | policy_duration_s | 95.71 | 7.55 | 95.72 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_rgbd_act | o1 | 失败 | 2026-11-13T11:23:49.561+08:00 | run_20261113_112349_561783 | 39.89 | duration_s（无策略停止时间字段） | — | — | 39.89 | 超出限幅 | manual_failure_stop | — | False | False | False | clamp |
| merge_rgbd_act | o1 | 失败 | 2026-11-13T11:26:22.991+08:00 | run_20261113_112622_991466 | 45.33 | duration_s（无策略停止时间字段） | — | — | 45.33 | 超出限幅 | manual_failure_stop | — | False | False | False | clamp |
| merge_rgbd_act | o | 成功 | 2026-11-13T11:28:09.864+08:00 | run_20261113_112809_864482 | 88.02 | policy_duration_s | 95.61 | 7.59 | 95.61 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_rgbd_act | o | 失败 | 2026-11-13T11:32:04.582+08:00 | run_20261113_113204_577368 | 34.51 | duration_s（无策略停止时间字段） | — | — | 34.51 | 超出限幅 | manual_failure_stop | — | False | False | False | clamp |
| merge_rgbd_act | o | 成功 | 2026-11-13T11:33:14.422+08:00 | run_20261113_113314_422081 | 100.88 | policy_duration_s | 108.91 | 8.03 | 108.91 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_rgbd_act | o | 失败 | 2026-11-13T11:36:07.198+08:00 | run_20261113_113607_198300 | 37.12 | duration_s（无策略停止时间字段） | — | — | 37.12 | 超出限幅 | manual_failure_stop | — | False | False | False | clamp |
| merge_rgbd_act | o | 失败 | 2026-11-13T11:37:19.701+08:00 | run_20261113_113719_696075 | 58.66 | duration_s（无策略停止时间字段） | — | — | 58.66 | 超出限幅 | manual_failure_stop | — | False | False | False | clamp |
| merge_rgbd_act | g | 成功 | 2026-11-13T11:40:11.547+08:00 | run_20261113_114011_547087 | 90.63 | policy_duration_s | 98.10 | 7.48 | 98.11 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_rgbd_act | g | 失败 | 2026-11-13T11:42:48.882+08:00 | run_20261113_114248_882268 | 40.42 | duration_s（无策略停止时间字段） | — | — | 40.42 | 超出限幅 | manual_failure_stop | — | False | False | False | clamp |
| merge_rgbd_act | g | 失败 | 2026-11-13T11:44:05.753+08:00 | run_20261113_114405_753578 | 32.77 | duration_s（无策略停止时间字段） | — | — | 32.77 | 超出限幅 | manual_failure_stop | — | False | False | False | clamp |
| merge_rgbd_act | g | 失败 | 2026-11-13T11:45:10.757+08:00 | run_20261113_114510_757466 | 40.90 | duration_s（无策略停止时间字段） | — | — | 40.90 | 超出限幅 | manual_failure_stop | — | False | False | False | clamp |
| merge_rgbd_act | g | 失败 | 2026-11-13T11:46:20.762+08:00 | run_20261113_114620_761944 | 47.70 | duration_s（无策略停止时间字段） | — | — | 47.70 | 超出限幅 | manual_failure_stop | — | False | False | False | clamp |
| merge_rgbd_dp | g | 失败 | 2026-11-13T13:15:52.217+08:00 | run_20261113_131552_217601 | 72.71 | duration_s（无策略停止时间字段） | — | — | 72.71 | 夹取失败 | manual_failure_stop | — | False | False | False | clamp |
| merge_rgbd_dp | g | 成功 | 2026-11-13T13:39:34.346+08:00 | run_20261113_133934_346251 | 95.28 | policy_duration_s | 98.85 | 3.56 | 98.85 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_rgbd_dp | g | 成功 | 2026-11-13T13:42:11.721+08:00 | run_20261113_134211_721868 | 93.34 | policy_duration_s | 97.86 | 4.52 | 97.87 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_rgbd_dp | g | 失败 | 2026-11-13T13:44:37.795+08:00 | run_20261113_134437_795103 | 76.37 | duration_s（无策略停止时间字段） | — | — | 76.37 | 超出限幅 | manual_failure_stop | — | False | False | False | clamp |
| merge_rgbd_dp | g | 失败 | 2026-11-13T13:46:24.773+08:00 | run_20261113_134624_773721 | 117.34 | duration_s（无策略停止时间字段） | — | — | 117.34 | 夹取失败 | manual_failure_stop | — | False | False | False | clamp |
| merge_rgbd_dp | o | 失败 | 2026-11-13T13:49:07.348+08:00 | run_20261113_134907_348436 | 79.23 | duration_s（无策略停止时间字段） | — | — | 79.23 | 夹取失败 | manual_failure_stop | — | False | False | False | clamp |
| merge_rgbd_dp | o | 成功 | 2026-11-13T13:50:55.487+08:00 | run_20261113_135055_487091 | 104.13 | policy_duration_s | 108.51 | 4.38 | 108.52 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_rgbd_dp | o | 成功 | 2026-11-13T13:53:39.422+08:00 | run_20261113_135339_422716 | 113.14 | policy_duration_s | 116.69 | 3.55 | 116.69 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_rgbd_dp | o | 失败 | 2026-11-13T13:56:48.633+08:00 | run_20261113_135648_633452 | 180.09 | policy_duration_s | 192.62 | 12.53 | 180.09 | 超时 | maximum_run_time | — | False | True | False | clamp |
| merge_rgbd_dp | o | 成功 | 2026-11-13T14:02:08.938+08:00 | run_20261113_140208_938250 | 121.17 | policy_duration_s | 124.76 | 3.59 | 124.76 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_rgbd_dp | o1 | 成功 | 2026-11-13T14:05:20.416+08:00 | run_20261113_140520_416630 | 113.35 | policy_duration_s | 118.51 | 5.17 | 118.51 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_rgbd_dp | o1 | 成功 | 2026-11-13T14:08:20.627+08:00 | run_20261113_140820_627626 | 121.54 | policy_duration_s | 126.24 | 4.70 | 126.24 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_rgbd_dp | o1 | 成功 | 2026-11-13T14:11:31.773+08:00 | run_20261113_141131_773308 | 108.99 | policy_duration_s | 113.32 | 4.33 | 113.32 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_rgbd_dp | o1 | 失败 | 2026-11-13T14:14:31.351+08:00 | run_20261113_141431_351356 | 75.59 | duration_s（无策略停止时间字段） | — | — | 75.59 | 超出限幅 | manual_failure_stop | — | False | False | False | clamp |
| merge_rgbd_dp | o1 | 成功 | 2026-11-13T14:16:18.424+08:00 | run_20261113_141618_424039 | 104.69 | policy_duration_s | 110.79 | 6.10 | 110.79 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_rgbd_dp | g1 | 失败 | 2026-11-13T14:19:14.873+08:00 | run_20261113_141914_873362 | 44.18 | duration_s（无策略停止时间字段） | — | — | 44.18 | 超出限幅 | manual_failure_stop | — | False | False | False | clamp |
| merge_rgbd_dp | g1 | 失败 | 2026-11-13T14:20:28.037+08:00 | run_20261113_142028_037101 | 89.84 | duration_s（无策略停止时间字段） | — | — | 89.84 | 夹取失败 | manual_failure_stop | — | False | False | False | clamp |
| merge_rgbd_dp | g1 | 成功 | 2026-11-13T14:22:52.045+08:00 | run_20261113_142252_045210 | 120.17 | policy_duration_s | 125.12 | 4.95 | 125.12 | — | organ_coverage_success | — | True | False | False | clamp |
| merge_rgbd_dp | g1 | 失败 | 2026-11-13T14:25:46.921+08:00 | run_20261113_142546_921195 | 56.43 | duration_s（无策略停止时间字段） | — | — | 56.43 | 超出限幅 | manual_failure_stop | — | False | False | False | clamp |
| merge_rgbd_dp | g1 | 成功 | 2026-11-13T16:44:34.639+08:00 | run_20261113_164434_639244 | 110.01 | policy_duration_s | 114.44 | 4.43 | 114.44 | — | organ_coverage_success | — | True | False | False | clamp |

## 排除记录

| 策略 | 运行目录 | 排除原因 |
| --- | --- | --- |
| merge_Qtoken_RGBD | run_20261112_140522_947518 | 未人工确认/无 evaluation_result.json |
| merge_Qtoken_RGBD | run_20261112_142445_705502 | 未人工确认/无 evaluation_result.json |
| merge_Qtoken_RGBD | run_20261112_143007_599359 | 未人工确认/无 evaluation_result.json |
| merge_Qtoken_RGBD | run_20261112_143655_988109 | 未人工确认/无 evaluation_result.json |
| merge_Qtoken_RGBD | run_20261112_140522_947518_ae29a7d6 | 已丢弃 |
| merge_Qtoken_RGBD | run_20261112_142445_705502_481daf06 | 已丢弃 |
| merge_Qtoken_RGBD | run_20261112_143007_599359_c2fd1d9f | 已丢弃 |
| merge_Qtoken_RGBD | run_20261112_143201_899717_c9cc1be8 | 已丢弃 |
| merge_Qtoken_RGBD | run_20261112_143655_988109_caa9f04c | 已丢弃 |
| merge_Qtoken_RGBD | run_20261113_105608_951561_52fabed7 | 已丢弃 |
| merge_rgbd_act | run_20261113_112600_010072_88793768 | 已丢弃 |

## 校验与来源

- 纳入结果文件 60 个；核对结果标签、任务、失败原因、记录时长与索引/元数据一致；源文件 SHA-256 见 `source_manifest.json`。
- 本次只核对结果与时长元数据，没有重新解码全部视频，也未根据视频改判任务成败。
- 校验提示：无结果/索引不一致或待确认缺失项。
- 逐条源文件路径及完整原始数值见 `summary_data.json`；CSV 为同一统计结果，便于 Excel 或后续作图。

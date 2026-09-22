# 实验结果空模板

状态：只有表头，没有任何实测结果。UTF-8 CSV，英文列名；数值未知留空，不能用0替代缺失。
0只表示实际测得的零。路径指向可追溯原始记录或行ID清单。所有表可复用同一批试验，不要求新增重复。

## 行粒度与关联

| 文件 | 一行表示 |
| --- | --- |
| trial_results.csv | 一次真实闭环试验；trial_id唯一，即使失败/无效也保留 |
| main_summary.csv | 一个明确层级和评价口径的结果汇总 |
| representation_ablation.csv | 一个模型、审核划分、视角、关系及事件分层的读出诊断 |
| history_ablation.csv | 一个反馈变体在固定回放集合和对齐槽位上的诊断 |
| runtime_summary.csv | 同配置、条件、任务类型及事件分层的运行时统计 |
| budget_summary.csv | 一个真实训练运行在一个示范预算下的资源及结果关联 |
| stratum_weights.csv | 一个聚合集合内一个条件/训练运行的预定权重 |
| config_registry.csv | 一个训练或推理参数的既有证据与最终确认值；不是实验结果 |

`source_summary_id`关联main_summary；原始trial清单使用唯一trial_id，汇总不可重复计数。
不同训练预算、四组表示、MLP/GRU等可以引用同一主表结果，不重新制造试验。

## 通用字段

- `protocol_id`：预先冻结评价协议版本；含完成事件、时限、域外/不确定处理、停止规则。
- `method_id`：方法变体的稳定标识，不以显示名替代配置。
- `base_checkpoint_id`、`residual_checkpoint_id`：文件哈希或能解析到哈希的ID；无残差时后者空。
- `checkpoint_set_id`：基础、残差及分割器组合的版本清单ID。
- `base_seed_id`、`residual_seed_id`：各自真实训练种子；不知道留空，不从文件名猜。
- `training_run_id`：关联各训练阶段的运行清单ID。同一冻结基础上的多个残差种子有嵌套依赖，不能当独立基础训练。
- `config_id`：完整执行配置ID，含动作单位/归一化、视角、重规划、同步/异步、残差及屏蔽模式。
- `dataset_split_id`、`audit_split_id`：数据/审核划分版本；`subset_id`为预算子集，`subset_seed_id`为其抽样种子。
- `platform_id`、`task_id`、`object_id`、`position_id`：真实条件标识；`condition_id`是这些条件及扰动设置的冻结组合。
- `analysis_id`、`summary_id`：本表唯一行ID；`notes`说明缺失、不适用及限制，不在数值列写文字。
- 所有`*_manifest_path`：来源或成员清单路径，记录ID、版本和选取条件。不得用无成员记录的目录名代替。

## 最终配置登记

`config_registry.csv`覆盖分割训练/推理、基础训练、残差训练和策略推理。
`recorded_value`只放已核对的配置记录或明确代码值，不表示该值已被最终实验采用，也不证明一次训练确实完整执行。
`evidence_path/evidence_key`是证据文件和键/符号；`record_status`区分`recorded_config`、`code_verified`和`unverified`。
`final_value`目前全部为空，`final_status`全部为`pending_author`；作者确认后填实际值并注明对应运行/模型版本，不能自动复制旧记录。
`stage`标识阶段，`view_id`标识front/side/all，`parameter`为参数名，`unit`按实际量纲填写，`config_id`绑定最终配置。
目前Qtoken-FS基础配置中的步数、学习率、batch和seed有直接记录；优化器、调度器等本轮未核对的项留空，不推断为默认ACT。
残差训练参数虽在工作稿有既有审计描述，本登记先保留待核字段，最终须链接实际运行配置而非仅以论文正文作证。
分割checkpoint路径来自基础配置记录，不代表原路径在当前机器可用或其权重哈希已核验。
类别顺序及palette数组用CSV标准引号中的JSON保存；需要结构化CSV解析，不能简单按逗号拆列。
不将UI配置里的目标频率、预设网格、时间上限当作本轮实测设置或运行频率。

## trial_results

- `paired_block_id`：跨方法匹配初始条件与重复编号的区组；`repeat_id`为区组内重复，`run_order`为实际运行顺序。
- `started_at_utc`：ISO 8601 UTC开始时间。
- `outcome`：`success`、`failure`、`unassessable`或`technical_invalid`，不得将算法失败当技术无效。
- `timeout`：真实超时则1，否则0；未知空。
- `completion_time_s`：成功判定时间距开始的秒数，仅成功填；`elapsed_time_s`为所有试验实际持续秒数。
- `failure_type`：预先规定的失败类；`technical_invalid_reason`：外部故障事实，非算法失败借口。
- `included_in_primary`：按冻结协议是否进入主要分母(1/0)；`exclusion_rule_id`记录排除依据，不能无规则删失败。
- `assessor_id`：内部审核人或冻结评价器ID；`adjudication_status`记录未审/一致/仲裁/仍不确定，不作为发布身份信息。
- `video_path`、`log_path`：该试验原始视频和逐槽日志路径。

## main_summary与权重

- `aggregation_level`：`checkpoint_condition`、`checkpoint_standardized`、`multi_run_standardized`。
- 逐条件行填真实`training_run_id`及`condition_id`；聚合行不写伪seed，相关ID留空并由`stratum_manifest_path`列全。
- `source_trial_manifest_path`包含全部纳入/审计的原始trial_id；`outcome_rule_id`明确不确定结果如何进入分母。
- `n_trials`：该口径实际分母；`n_success`为其中成功数；`n_failure`为其中按规则计为失败的数量。
- `n_unassessable`、`n_technical_invalid`：来源集合的原始不确定和技术无效计数，可与最终分类重叠；不能简单加到分母。
- `n_timeout`：主要分母中超时数，是失败的子集；`n_training_runs`：实际训练运行数，不自动等于独立样本数。
- `success_rate`：0到1；逐条件行是成功/分母，标准化行是预定加权率，未必等于合并总成功/总次数。
- `ci_level`：如0.95；`ci_method`：如`wilson_within_condition`、`hierarchical_paired_bootstrap`或`descriptive_only`。
- `ci_lower`、`ci_upper`：0到1，方法不适用或重复不足则留空并解释；不要写虚假的零宽区间。
- `condition_weighting`、`seed_weighting`：条件及独立训练层级的预定规则；`weight_manifest_path`指stratum_weights中权重集合。
- `success_time_n`：有可用完成时间的成功数；`success_time_mean_s`、`success_time_median_s`仅条件于这些成功试验。跨层采用何种加权须在notes指定，不暗示全体试验速度。

Wilson仅用于同checkpoint、同条件下独立可交换的重复。跨条件或训练种子不直接合并后使用Wilson。
分层总体与配对差值保留条件区组及真实训练依赖；同基础checkpoint的不同残差种子须保留嵌套。
聚合前声明目标条件权重和训练层级权重，不能为了结果好看事后修改。缺失层不默认重新归一化。
`stratum_weights`的`weight_set_id`指定集合，`normalized_weight`非负且每个集合总和1；
`weight_basis`说明预定依据，`source_summary_id`指逐层结果。数值未确定时整行不填。

## representation_ablation

- `view_id`、`relation_id`：相机和具体关系；`visibility_stratum`、`object_size_stratum`为预定审核分层。
- `readout_stage`：计划时刻/反馈目标槽位；`latent_mode`：`deployment_zero`或单独标明的训练后验诊断。
- `has_semantics`、`has_relation_supervision`：1/0；`n_query_tokens`为实际数量。
- `probe_config_id`：冻结探针容量、输入、拟合划分、时刻对齐规则；`audit_manifest_path`列出审核帧及真值来源。
- `n_episodes`：审核轨迹数；`n_candidate_frames`为候选审核帧数；`n_valid_frames`为此关系可评价数；`valid_fraction`为后者/前者。
- `relation_error_metric`指定MAE、欧氏距离等；`relation_error_mean`是对人工真值的读出误差；`relation_scale_id`指定坐标、单位及归一化。
- `pseudolabel_error_mean`是伪标签对人工真值误差；`model_to_pseudolabel_error_mean`是对伪标签拟合误差。使用同量纲和声明的指标，不能彼此替代。

## history_ablation

- `feedback_arch`：MLP/GRU等；`current_input_config_id`与`history_input_config_id`完整列明当前、速度及过去记录，避免混淆历史来源。
- `history_window_records`：窗口条数；`trainable_parameters`：可训练参数个数；`output_masked`表示仅发送残差屏蔽(1/0)。关闭反馈计算不等于1。
- `replay_manifest_path`、`slot_alignment_id`、`action_scale_id`、`sample_weighting`共同确定同分布、同槽位、同动作尺度与权重。
- `n_episodes`、`n_valid_slots`：回放轨迹数和有效目标槽位数，不将槽位视为独立轨迹。
- `base_mse`、`corrected_mse`：采用同一平方范数约定的风险；`alignment_term`为两倍基础专家误差与实际修正的平均内积；`correction_energy`为相同权重下修正平方范数均值。
- 应满足base_mse减corrected_mse等于alignment_term减correction_energy；逐维平均时四项同除动作维数。门控前后用独立analysis_id并在notes说明。
- `saturation_fraction`：达到限幅的有效槽位--动作维度对数/全部有效对数，0到1；它不等于任务收益或理论实现误差。

## runtime_summary

- `task_kind`：base/feedback/control；`event_stratum`：全部或预定关键事件，分开汇总。
- `n_jobs`为推理任务数；`n_target_slots`为纳入记录的执行槽位数；`n_trials`为来源闭环试验数。
- `n_adopted_slots`、`n_late_slots`、`n_wrong_plan_slots`、`n_invalid_slots`、`n_no_candidate_slots`：仅反馈槽位填写，按预定互斥优先规则分类；候选多次覆盖不能重复算槽位。规则写入config。
- `latency_definition`明确起止，如观测捕获到可用结果，不能用forward时间冒充端到端。
- `end_to_end_p50_ms/p95_ms/p99_ms`、`submission_interval_p50_ms/p95_ms`：任务端到端延迟和提交间隔分位数。
- `observation_age_p50_ms/p95_ms`：目标槽位执行时距对应观测的年龄；`history_span_p50_ms`为反馈有效历史跨度中位数。
- `hardware_id`：计算设备和运行环境清单ID。不同时间定义不合并；无日志空缺，分位数不能通过平均分位数得到。

## budget_summary

- `budget_scope`：机器人示范预算/全流程预算；明确外部语义资源是否固定。
- `n_demo_episodes/n_demo_frames`：实际示范条数/帧数；`split_manifest_path`列明训练、验证、审核用途。
- `n_segmentation_train_frames`：分割训练帧数；`n_segmentation_manual_frames`：人工标注或修订帧数，重复定义须说明。
- `external_pretraining_id`：额外预训练来源版本；`annotation_hours`：真实记录的标注时间，无法核实时空缺。
- `base_updates/residual_updates`：实际优化更新次数；`training_compute_hours`为声明范围的训练计算时间，`compute_hardware_id`说明设备和数量。
- `source_summary_id`关联对应结果。数据规模标签不提供成功率证据；同一subset/seed跨方法匹配，独立训练不能由重复评测冒充。

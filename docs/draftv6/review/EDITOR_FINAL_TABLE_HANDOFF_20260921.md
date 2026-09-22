# 收尾交接：颜色表与配置登记

稳定PDF：`docs/draftv6/main.pdf`，17页，创建时间2026-09-21 23:08:35 CST。
SHA-256：`dc0e50c56f5125ea23f8d805be05db1f8a6261272b7c318cd20dfacdc2d49dfb`
本版本替代23:05稳定版供最终复审；暂不继续改稿，等待本轮审查结论。

## 完成项

- 颜色表改为编号Table IV，label为tab:semanticpalette，正文两处引用。
- 核对train_mask_act_policy.py:248-255浮点palette与:2252-2267 softmax/einsum路径；表中8-bit通道除255对应e_c，按概率加权而非先argmax。Qtoken-FS保存配置的palette与表一致。
- 主结果caption明确所有语义关系增强组仍保留RGB，表可独立理解。
- 新建results_templates/config_registry.csv，覆盖分割训练/推理、基础/残差训练及策略推理，共82行配置登记。README补字段、证据和最终确认规则。
- 已核对的Qtoken-FS配置及SF_C_GRU UI配置仅入recorded_value并标注来源，不变成最终设置或实测事实。未知优化器/调度器等留空，不按默认ACT补值。final_value全空、final_status全pending_author。

## 验证

标准csv解析验证82行列结构完整、所有最终值为空、未核对值为空。
build.sh成功、git diff --check通过；无Overfull/未解析引用/缺字告警，旧字体与algorithmic告警不变。
第15页颜色表与归一化说明已视觉检查。PDF仍17页，未改字号和边距。

不含新增实测结果、硬件操作、训练代码修改、提交或发布。真实结果/照片、最终配置与评价规则确认、英语转换及投稿前检查仍需作者完成。

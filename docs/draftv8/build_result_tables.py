"""Export standalone, reviewable result tables under docs/. No external packages."""
from pathlib import Path
import csv,json,shutil
P=Path(__file__).resolve().parent;OUT=P.parent/'experiment_results_20261023';OUT.mkdir(exist_ok=True)
for f in (P/'results').glob('*.csv'):shutil.copy2(f,OUT/f.name)
for n in ['summary.json','manual_review.json','validation.json','recording_validation.json']:shutil.copy2(P/'results'/n,OUT/n)
s=json.loads((OUT/'summary.json').read_text())
M=['ACT','DP','RGBQtoken','segNoQtoken','viewQtoken'];O=['cube','paperball','screw','crayon'];ON=['方块','纸球','螺丝','蜡笔（训练未见）']
A={(r['method'],r['object']):r for r in s['summary']}
lines=['# 最终五组实机实验结果','', '数据截至 2026-10-23 16:26 补采完成。共 480 次有效试验：每方法 96 次，四类物体 × 12 个位置 × 每条件两次。VIOLA 未纳入。', '', '## 总体与物体类别','']
def table(head,rows):
 lines.append('| '+' | '.join(map(str,head))+' |');lines.append('| '+' | '.join(['---']*len(head))+' |')
 for r in rows:lines.append('| '+' | '.join(map(str,r))+' |')
 lines.append('')
def rate(r):return f"{r['successes']}/{r['n']}（{r['success_rate_pct']:.2f}%）"
table(['方法']+ON+['总体'],[[m]+[rate(A[m,o]) for o in O+['ALL']] for m in M])
with (OUT/'总体与物体结果.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['方法']+ON+['总体']);w.writerows([[m]+[rate(A[m,o]) for o in O+['ALL']] for m in M])
lines+=['## 训练物体与未见物体',''];table(['方法','训练物体（72次）','蜡笔（24次）'],[[m,rate(A[m,'SEEN']),rate(A[m,'crayon'])] for m in M])
lines+=['## 相同位置下比较不同物体的汇总','', '每格为成功次数 / 8。x 从左向右 0–3，y 从上向下 0–2。','']
table(['位置']+M,[[f"({r['x']},{r['y']})"]+[f"{r[m+'_total']}/8" for m in M] for r in s['positions']])
lines+=['## 每个物体在不同位置的详细对照','', '每格为成功次数 / 2；相同位置的不同物体可沿四张表比较。','']
for o,on in zip(O,ON):
 lines+=['### '+on,''];table(['位置']+M,[[f"({r['x']},{r['y']})"]+[f"{r[m]}/2" for m in M] for r in s['condition_details'] if r['object']==o])
lines+=['## 条件内两次重复',''];rs=[r for r in s['condition_reliability'] if r['object']=='ALL'];table(['方法','条件数','两次成功','一成一败','两次失败'],[[r['method'],r['conditions'],r['both_success'],r['one_success'],r['both_failure']] for r in rs])
lines+=['## 完成时间','', '控制时长不含保存、编码；成功子集各不相同，不能将其均时当作同一任务集合的速度排名。',''];table(['方法','总体成功率95%区间','成功子集均时（秒）','全部试验均时（秒）','20秒内成功'],[[m,f"{A[m,'ALL']['wilson95_low_pct']:.2f}–{A[m,'ALL']['wilson95_high_pct']:.2f}%",f"{A[m,'ALL']['mean_success_control_s']:.2f}",f"{A[m,'ALL']['mean_all_control_s']:.2f}",str(next(t for t in s['timing'] if t['method']==m and t['object']=='ALL')['success_by_20s'])+'/96'] for m in M])
lines+=['## 消融',''];table(['语义输入','查询监督','方法','成功率'],[['无','无','ACT',rate(A['ACT','ALL'])],['无','有','RGBQtoken',rate(A['RGBQtoken','ALL'])],['有','无','segNoQtoken',rate(A['segNoQtoken','ALL'])],['有','有','viewQtoken',rate(A['viewQtoken','ALL'])]])
lines+=['相对 ACT：仅查询 +18.75 个百分点，仅语义 +6.25 个百分点；完整方法比仅查询 +20.83、比仅语义 +33.33 个百分点。增益差 14.58 个百分点只作描述，未做独立交互推断。','', '## 总体方法检验','', '物体—位置区组内精确超几何卷积检验；10 个总体比较统一 Holm 校正。','']
table(['方法一','方法二','方法二多成功次数','Holm校正p'],[[r['left'],r['right'],r['success_count_difference'],f"{r['p_holm']:.6g}"] for r in s['method_tests'] if r['object']=='ALL'])
lines+=['## 位置与类别关联','', '每项 200000 次分层置换，20 项统一 Holm 校正；未检出不等于证明无影响。',''];names={'position':'12位置','object':'物体类别','x':'列x','y':'行y'}
table(['方法','关联因素','Holm校正p'],[[r['method'],names[r['effect']],f"{r['p_holm_20']:.6g}"] for r in s['association_tests']])
lines+=['## 失败阶段',''];table(['方法','未暴露','暴露未分离','分离未入区','曾入区但最终失败','总失败'],[[r['method'],r['never_exposed'],r['exposed_never_separated'],r['separated_never_reached_region'],r['reached_region_final_failure'],r['failures']] for r in s['failure_stages'] if r['object']=='ALL'])
lines+=['## 数据口径与文件','', '- segNoQtoken 方块 (1,0) 的边界误判按人工确认的失败计入。','- 纸球 (3,0) 的无效手动中止排除，补采记录为正常执行失败；最终该方法仍为96次。','- 原始来源、快照哈希和运行配置见 `summary.json`；具体人工复核见 `manual_review.json`。','- CSV 文件保留详细分组、逐试验结果及检验值；不含图像帧或模型权重。','- 论文见 [draftv7 PDF](../draftv7/main.pdf)，图表见 [中文实验图](../draftv7/figures/results_20261023/)。','- 复现表格：`python docs/draftv7/build_result_tables.py`。','']
(OUT/'README.md').write_text('\n'.join(lines))
print('Exported standalone tables:',OUT)

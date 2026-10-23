"""Rebuild Chinese manuscript figures/tables from the frozen 480-trial snapshot.
Run: /home/romilab/.cache/codex/eval-analysis-venv/bin/python docs/draftv7/build_experiment_assets.py
"""
from pathlib import Path
import csv,json,hashlib
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
P=Path(__file__).resolve().parent
OUT=P/'figures'/'results_20261023';OUT.mkdir(exist_ok=True)
FONT='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
plt.rcParams.update({'font.family':['DejaVu Sans', FontProperties(fname=FONT).get_name()],'font.size':9,'axes.unicode_minus':False,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
M=['ACT','DP','RGBQtoken','segNoQtoken','viewQtoken']
O=['cube','paperball','screw','crayon'];ON=['方块','纸球','螺丝','蜡笔（训练未见）']
MN=['ACT','DP','RGB＋Q','语义无Q','语义＋Q（本文）'];C=['#607D8B','#CC6677','#4477AA','#DD9944','#228866']
def read(n):
 with (P/'results'/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
rows=read('trials.csv');S=json.loads((P/'results'/'summary.json').read_text())
assert len(rows)==480
counts=Counter((r['method'],r['object'],int(r['x']),int(r['y'])) for r in rows)
assert len(counts)==240 and set(counts.values())=={2}
def k(m,o=None,x=None,y=None):return sum(int(r['success']) for r in rows if r['method']==m and (o is None or r['object']==o) and (x is None or int(r['x'])==x) and (y is None or int(r['y'])==y))
assert [k(m) for m in M]==[54,50,72,60,92]
A={(r['method'],r['object']):r for r in S['summary']}
for m in M:
 for o in O+['ALL']:
  assert A[m,o]['successes']==k(m,None if o=='ALL' else o)
def save(fig,n):
 for ext in ['pdf','png']:fig.savefig(OUT/f'{n}.{ext}',bbox_inches='tight',facecolor='white')
 plt.close(fig)
fig,(ax,bx)=plt.subplots(1,2,figsize=(7.2,3.15),gridspec_kw={'width_ratios':[1,1.8]},layout='constrained')
v=np.array([100*k(m)/96 for m in M]);lo=[A[m,'ALL']['wilson95_low_pct'] for m in M];hi=[A[m,'ALL']['wilson95_high_pct'] for m in M]
ax.barh(range(5),v,color=C,height=.6,xerr=[v-np.array(lo),np.array(hi)-v],error_kw={'lw':.8,'capsize':2,'ecolor':'#333333'})
ax.set_yticks(range(5),MN);ax.invert_yaxis();ax.set_xlim(0,134);ax.set_xticks([0,50,100]);ax.set_xlabel('成功率（%）');ax.set_title('(a) 总体，每方法96次',fontsize=10)
for i,m in enumerate(M):ax.text(132,i,f'{k(m)}/96',ha='right',va='center',fontsize=8)
x=np.arange(4);w=.15
for i,m in enumerate(M):
 vals=[100*k(m,o)/24 for o in O];bx.bar(x+(i-2)*w,vals,w,color=C[i],label=MN[i])
 for j,o in enumerate(O):bx.text(x[j]+(i-2)*w,vals[j]+1,str(k(m,o)),ha='center',fontsize=7)
bx.set_ylim(0,112);bx.set_yticks([0,25,50,75,100]);bx.set_xticks(x,['方块','纸球','螺丝','蜡笔*']);bx.set_ylabel('成功率（%）');bx.set_title('(b) 分物体，每方法每物体24次',fontsize=10)
bx.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.12),fontsize=7,frameon=False,columnspacing=.8)
save(fig,'overall_objects_zh')
fig,ax=plt.subplots(figsize=(3.45,2.45),layout='constrained')
mat=np.array([[54,72],[60,92]])/96*100
im=ax.imshow(mat,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
for y in range(2):
 for x in range(2):
  i=[[0,2],[3,4]][y][x];ax.text(x,y,f'{MN[i]}\n{k(M[i])}/96\n{mat[y,x]:.2f}%',ha='center',va='center',color='white' if mat[y,x]>70 else '#18252b',fontsize=10)
ax.set_xticks([0,1],['无查询监督','有查询监督']);ax.set_yticks([0,1],['无语义输入','有语义输入']);ax.tick_params(length=0);ax.set_title('ACT骨干的2×2模块对照',fontsize=11)
save(fig,'ablation_zh')
fig,axes=plt.subplots(2,3,figsize=(7.15,4.0),layout='constrained')
for i,(m,ax) in enumerate(zip(M,axes.flat)):
 arr=np.array([[k(m,x=x,y=y) for x in range(4)] for y in range(3)])
 im=ax.imshow(arr/8*100,vmin=0,vmax=100,cmap='YlGnBu');ax.set_title(MN[i],fontsize=10)
 ax.set_xticks(range(4));ax.set_yticks(range(3));ax.set_xlabel('列 x');ax.set_ylabel('行 y')
 for y in range(3):
  for x in range(4):ax.text(x,y,f'{arr[y,x]}/8',ha='center',va='center',color='white' if arr[y,x]>=6 else '#18252b',fontsize=9)
axes.flat[-1].axis('off');axes.flat[-1].text(.05,.85,'四类物体合并\n每格8次测试\n\nx：从左到右 0–3\ny：从上到下 0–2\n\n统一色阶：成功率0–100%',va='top',linespacing=1.5)
save(fig,'position_zh')
fig,axes=plt.subplots(5,4,figsize=(7.15,7.8),layout='constrained')
for i,m in enumerate(M):
 for j,o in enumerate(O):
  ax=axes[i,j];arr=np.array([[k(m,o,x,y) for x in range(4)] for y in range(3)])
  ax.imshow(arr,vmin=0,vmax=2,cmap='YlGnBu');ax.set_xticks(range(4));ax.set_yticks(range(3));ax.tick_params(labelsize=7,length=0)
  if i==0:ax.set_title(ON[j],fontsize=9)
  if j==0:ax.set_ylabel(MN[i]+'\n行 y',fontsize=8)
  if i==4:ax.set_xlabel('列 x',fontsize=8)
  for y in range(3):
   for x in range(4):ax.text(x,y,f'{arr[y,x]}/2',ha='center',va='center',fontsize=8,color='white' if arr[y,x]==2 else '#18252b')
save(fig,'object_position_zh')
fig,(ax,bx)=plt.subplots(1,2,figsize=(7.15,3.0),layout='constrained')
for i,m in enumerate(M):
 times=sorted(float(r['control_duration_s']) for r in rows if r['method']==m and r['success']=='1')
 ax.step([0]+times+[40.1],[0]+[100*j/96 for j in range(1,len(times)+1)]+[100*len(times)/96],where='post',color=C[i],label=MN[i])
ax.set_xlim(0,40);ax.set_ylim(0,102);ax.set_xlabel('策略控制时长（秒）');ax.set_ylabel('已成功完成 / 全部试验（%）');ax.set_title('(a) 累计成功完成比例',fontsize=10);ax.legend(fontsize=7,frameon=False)
f={r['method']:r for r in read('failure_stages.csv') if r['object']=='ALL'}
fields=['never_exposed','exposed_never_separated','separated_never_reached_region','reached_region_final_failure']
labels=['未暴露','暴露但未分离','分离但未入区','曾入区但最终失败'];colors=['#999999','#CCBB88','#CC6677','#882255'];left=np.zeros(5)
for field,label,color in zip(fields,labels,colors):
 vals=np.array([int(f[m][field]) for m in M]);bx.barh(range(5),vals,left=left,color=color,label=label)
 for i,vv in enumerate(vals):
  if vv>=3:bx.text(left[i]+vv/2,i,str(vv),ha='center',va='center',fontsize=8,color='white' if color in ['#CC6677','#882255'] else 'black')
 left+=vals
bx.set_yticks(range(5),MN);bx.invert_yaxis();bx.set_xlim(0,50);bx.set_xlabel('失败次数');bx.set_title('(b) 最远到达阶段划分',fontsize=10);bx.legend(fontsize=7,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.22),ncol=2,columnspacing=.8)
save(fig,'time_failure_zh')
# Exact source-derived tables.
def put(n,lines):(P/'results'/n).write_text('\n'.join(lines)+'\n')
lines=[]
for m in M:
 a=A[m,'ALL'];cells=[m]+[f"{k(m,o)}/24 ({100*k(m,o)/24:.2f})" for o in O]+[f"{k(m)}/96 ({100*k(m)/96:.2f})"]
 if m=='viewQtoken':cells=[r'\textbf{'+x+'}' for x in cells]
 lines.append(' & '.join(cells)+r' \\')
put('main_table_rows.tex',lines)
lines=[]
for m in M:
 r=A[m,'ALL'];t=next(t for t in S['timing'] if t['method']==m and t['object']=='ALL')
 lines.append(f"{m} & {r['wilson95_low_pct']:.2f}--{r['wilson95_high_pct']:.2f} & {r['mean_success_control_s']:.2f} & {r['mean_all_control_s']:.2f} & {t['success_by_20s']}/96 "+r'\\')
put('timing_rows.tex',lines)
lines=[]
for m in M[:-1]:
 t=next(t for t in S['method_tests'] if t['object']=='ALL' and t['left']==m and t['right']=='viewQtoken')
 val=t['p_holm'];exp=int(np.floor(np.log10(val)));mant=val/10**exp
 lines.append(f"{m} & {100*(k('viewQtoken')-k(m))/96:.2f} & ${mant:.2f}\\times10^{{{exp}}}$ "+r'\\')
put('contrast_rows.tex',lines)
manifest={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (P/'results').glob('*') if f.suffix=='.csv' or f.name=='summary.json'}
(P/'results'/'figure_source_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Validated 480 records / 240 method-object-position cells; wrote 5 Chinese figure pairs and 3 table fragments.')

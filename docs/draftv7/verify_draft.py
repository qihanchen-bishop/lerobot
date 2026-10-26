"""Read-only verification of manuscript inputs, preserved source, equations and labels."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,csv,hashlib,json,math,re
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--require-originals',action='store_true',help='Also require original-machine logs and snapshots.')
args=parser.parse_args()
P=Path(__file__).resolve().parent
ROOT=P.parents[1]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
manifest=json.loads((P/'results/draftv6_source_sha256.json').read_text())
# Logs contain paths and runtime diagnostics; they are not preserved manuscript inputs.
volatile_v6=[n for n in manifest if Path(n).suffix in {'.log','.aux','.blg','.out'}]
assert all(digest(P.parent/'draftv6'/n)==v for n,v in manifest.items() if n not in volatile_v6),'draftv6 manuscript changed'
s=json.loads((P/'results/summary.json').read_text())
missing_sources=[]
checked_sources=0
for r in s['sources']:
 for key in ['snapshot','original']:
  source=Path(r[key])
  if not source.is_file():
   missing_sources.append(str(source))
  else:
   assert digest(source)==r['sha256'],r['method']+' '+key+' changed; results need rechecking'
   checked_sources+=1
if args.require_originals:
 assert not missing_sources,'Original-machine sources unavailable: '+', '.join(missing_sources)
# The prior manifest anchors the frozen data, independently of regenerated assets.
frozen=json.loads((P/'results/figure_source_sha256.json').read_text())
for name,expected in frozen.items():
 assert digest(P/'results'/name)==expected,'Frozen result changed: '+name
assets=json.loads((P/'results/english_assets_manifest.json').read_text())
for group in ['source_sha256','output_sha256']:
 for name,expected in assets[group].items():
  assert digest(P/name)==expected,'English asset/source changed: '+name
from build_annotation_assets import validate as validate_annotation
annotation_manifest=json.loads((P/'results/annotation_finaldata/assets_manifest.json').read_text())
for group in ['source_sha256','output_sha256']:
 for name,expected in annotation_manifest[group].items():
  assert digest(P/name)==expected,'Annotation asset/source changed: '+name
annotation_snapshot=json.loads((P/'results/annotation_finaldata/snapshot.json').read_text())
annotation_rows=validate_annotation(annotation_snapshot)
with (P/'results/annotation_finaldata/metrics.csv').open(newline='') as f:
 annotation_csv=list(csv.DictReader(f))
assert len(annotation_csv)==len(annotation_rows)==8
for actual,expected in zip(annotation_csv,annotation_rows):
 for key,value in expected.items():
  assert actual[key]==('' if value is None else str(value)),key
with (P/'results/trials.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
assert len(rows)==480
cells=Counter((r['method'],r['object'],r['x'],r['y']) for r in rows)
assert len(cells)==240 and set(cells.values())=={2}
for a in s['summary']:
 rs=[r for r in rows if r['method']==a['method'] and (a['object']=='ALL' or (a['object']=='SEEN' and r['object']!='crayon') or r['object']==a['object'])]
 assert len(rs)==a['n'] and sum(int(r['success']) for r in rs)==a['successes']
# Recompute the overall exact tests from trial outcomes, not exported p values.
successes=Counter()
for r in rows:
 successes[r['method'],r['object'],r['x'],r['y']]+=int(r['success'])
blocks=sorted({(r['object'],r['x'],r['y']) for r in rows})
tests=[r for r in s['method_tests'] if r['object']=='ALL']
assert len(tests)==10
probabilities=[]
for test in tests:
 distribution={0:1.0}
 observed=0
 for block in blocks:
  left=successes[(test['left'],)+block]
  right=successes[(test['right'],)+block]
  total=left+right
  observed+=right-left
  local={total-2*k:math.comb(total,k)*math.comb(4-total,2-k)/6
         for k in range(max(0,total-2),min(2,total)+1)}
  updated=defaultdict(float)
  for a,pa in distribution.items():
   for b,pb in local.items():
    updated[a+b]+=pa*pb
  distribution=updated
 probability=sum(p for d,p in distribution.items() if abs(d)>=abs(observed))
 assert math.isclose(probability,test['p_two_sided'],abs_tol=1e-12),test
 probabilities.append(probability)
adjusted=0
for rank,i in enumerate(sorted(range(len(tests)),key=probabilities.__getitem__)):
 adjusted=max(adjusted,min(1,(len(tests)-rank)*probabilities[i]))
 assert math.isclose(adjusted,tests[i]['p_holm'],abs_tol=1e-12),tests[i]
active={}
def visit(f):
 if f in active:return
 text=f.read_text();active[f]=text
 for name in re.findall(r'\\(?:input|tableinput)\{([^}]+)\}',text):
  visit(P/(name if name.endswith('.tex') else name+'.tex'))
visit(P/'main.tex')
combined='\n'.join(active.values())
labels=re.findall(r'\\label\{([^}]+)\}',combined)
assert len(labels)==len(set(labels)), 'duplicate labels'
assert set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',combined))<=set(labels)
assert not re.search(r'残差|GRU|feedbackanalysis|methodfeedback',combined)
assert 'VIOLA' not in (P/'results/main_table_rows_en.tex').read_text()
# All numbered mathematics retained from the theory must match v6 exactly.
def equations(text):
 result={}
 for m in re.finditer(r'\\begin\{(equation|align)\}(.*?)\\end\{\1\}',text,re.S):
  for label in re.findall(r'\\label\{([^}]+)\}',m.group()):result[label]=m.group()
 return result
pairs=[('formulation.tex','formulation.tex'),('action_analysis.tex','action_analysis.tex'),('representation_analysis.tex','representation_analysis.tex'),('action_analysis_proofs.tex','action_analysis_proofs.tex'),('representation_feedback_proofs.tex','representation_feedback_proofs.tex')]
for name in ['formulation.tex','action_analysis.tex','representation_analysis.tex','action_analysis_proofs.tex','references.bib','build.sh']:
 assert (P/name).read_bytes()==(P.parent/'draftv6'/name).read_bytes(), name+' differs from v6'
checked=0
for new,old in pairs:
 a=equations((P/new).read_text());b=equations((P.parent/'draftv6'/old).read_text())
 for label,body in a.items():
  assert body==b[label],label
  checked+=1
 if old!='representation_feedback_proofs.tex':assert set(a)==set(b)
for name in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',combined):assert (P/name).is_file(),name
citations={k for value in re.findall(r'\\cite\{([^}]+)\}',combined) for k in value.split(',')}
bib=set(re.findall(r'@\w+\{([^,]+),',(P/'references.bib').read_text()))
assert citations<=bib, citations-bib
log=(P/'main.log').read_text()
assert 'Output written on' in log
assert not re.search(r'Overfull|Missing character|There were undefined references|Citation .* undefined|Reference .* undefined|^!',log,re.M)
print(json.dumps({'trial_count':len(rows),'method_object_position_cells':len(cells),'verified_summary_rows':len(s['summary']),'recomputed_overall_exact_tests_and_holm':len(tests),'unchanged_frozen_result_files':len(frozen),'verified_english_assets':len(assets['output_sha256']),'original_machine_sources_checked':checked_sources,'original_machine_sources_unavailable':missing_sources,'raw_video_reaudit_performed':False,'unchanged_v6_files':len(manifest)-len(volatile_v6),'volatile_v6_artifacts_excluded':volatile_v6,'unchanged_numbered_theory_equations':checked,'active_tex_files':len(active),'resolved_labels':len(labels),'resolved_citations':len(citations),'compile_no_missing_glyphs_or_refs_or_overflow':True},indent=2))
print(json.dumps({'annotation_view_model_rows':len(annotation_rows),'annotation_assets':len(annotation_manifest['output_sha256']),'annotation_aggregation':'Pooled confusion matrices and per-video flagged-frame counts verified; no new inference.'},indent=2))

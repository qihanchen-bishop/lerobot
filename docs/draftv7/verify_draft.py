"""Read-only verification of manuscript inputs, preserved source, equations and labels."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,re
P=Path(__file__).resolve().parent
ROOT=P.parents[1]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
manifest=json.loads((P/'results/draftv6_source_sha256.json').read_text())
assert all(digest(P.parent/'draftv6'/n)==v for n,v in manifest.items()),'draftv6 changed'
s=json.loads((P/'results/summary.json').read_text())
for r in s['sources']:
 assert digest(Path(r['snapshot']))==r['sha256'],r['method']+' snapshot changed'
 assert digest(Path(r['original']))==r['sha256'],r['method']+' live source changed; results need rechecking'
with (P/'results/trials.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
assert len(rows)==480
cells=Counter((r['method'],r['object'],r['x'],r['y']) for r in rows)
assert len(cells)==240 and set(cells.values())=={2}
for a in s['summary']:
 rs=[r for r in rows if r['method']==a['method'] and (a['object']=='ALL' or (a['object']=='SEEN' and r['object']!='crayon') or r['object']==a['object'])]
 assert len(rs)==a['n'] and sum(int(r['success']) for r in rs)==a['successes']
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
assert 'VIOLA' not in (P/'results/main_table_rows.tex').read_text()
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
print(json.dumps({'trial_count':len(rows),'method_object_position_cells':len(cells),'verified_summary_rows':len(s['summary']),'unchanged_v6_files':len(manifest),'unchanged_numbered_theory_equations':checked,'active_tex_files':len(active),'resolved_labels':len(labels),'resolved_citations':len(citations),'compile_no_missing_glyphs_or_refs_or_overflow':True},indent=2))

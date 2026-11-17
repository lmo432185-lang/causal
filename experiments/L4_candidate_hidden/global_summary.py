"""Evaluation view of already-frozen validation model choices.
No predictions, parameters, thresholds or saved decisions are changed.
Single-trajectory scores remain a separate preregistered diagnostic.
"""
import json,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import softmax
ROOT=Path(__file__).parent;OUT=ROOT/'results'
EDGE={'G1':{'YX','ZX'},'G2':{'YZ','ZX'},'G3':{'YX','YZ','ZX'},'G4':{'ZY','YX'}}
def effective(p):
 A=np.array(p['A']);d=p['delay'];k=p['kind'];e=set()
 if A[2,0]>=1e-3:e.add('YX')
 if A[2,1]>=1e-3:e.add('ZX')
 if k in ['G2','G3'] and A[1,0 if d==1 else 3]>=1e-3:e.add('YZ')
 if k=='G4' and A[0,1 if d==1 else 3]>=1e-3:e.add('ZY')
 return e

def main():
 assert (ROOT/'FINAL_EVALUATION_DONE.json').exists() and (ROOT/'POSITIVE_EVALUATION_DONE.json').exists();meta=json.loads((ROOT/'data/sealed_manifest.json').read_text())+json.loads((ROOT/'data/positive_sealed_manifest.json').read_text());rows=[]
 for m in meta:
  d=OUT/m['case'];a=json.loads((d/'X_only_decisions.json').read_text());p=json.loads((d/'selected_parameters.json').read_text())['models'];supp=json.loads((d/'supplementary_law_audit.json').read_text());names=['G1','G2','G3','G4'];scores=[max(v['validation_penalized'] for v in json.loads((d/f'{k}_initializations.json').read_text())) for k in names];post=softmax(scores);pred=a['selected_by_validation'];j=names.index(pred);ambiguity={pred}|{z['label'] for z in a['estimated_equivalent_alternatives'][pred]}|{k for k,b in zip(names,a['compatible_validation']) if b};guard=bool(post[j]>=.95 and len(ambiguity)==1 and not a['degenerate'][pred]);ext=bool(guard and not supp['additional_alternatives'][pred]);o=json.loads((d/'oracle_results.json').read_text());ow=max(o['fits'],key=lambda v:v['validation_score']);oe=effective(ow['parameters']);ee=effective(p[pred]);truth=m['truth'];row=dict(case=m['case'],section=m['section'],tag=m['tag'],truth=truth,seed=m['seed'],frozen_validation_prediction=pred,forced_correct=int(pred==truth),score_confidence=float(post[j]),naive_declared=bool(post[j]>=.95),primary_guard_declared=guard,supplementary_guard_declared=ext,estimated_edges=','.join(sorted(ee)),effective_graph_exact=ee==EDGE[truth],oracle_frozen_validation_prediction=ow['kind'],oracle_model_correct=ow['kind']==truth,oracle_estimated_edges=','.join(sorted(oe)),oracle_effective_graph_exact=oe==EDGE[truth],oracle_selected_delay=ow['delay'],X_only_selected_delay=p[pred]['delay'])
  rows.append(row)
 d=pd.DataFrame(rows);d.to_csv(OUT/'system_level_frozen_validation_choices.csv',index=False);d.groupby(['section','tag'])[['forced_correct','naive_declared','primary_guard_declared','supplementary_guard_declared','oracle_model_correct','oracle_effective_graph_exact','effective_graph_exact']].agg(['count','mean']).to_csv(OUT/'system_level_summary.csv');A=d[d.section=='A'];pd.crosstab(A.truth,A.frozen_validation_prediction,normalize='index').to_csv(OUT/'system_level_confusion_A.csv');(OUT/'GLOBAL_SUMMARY_NOTE.json').write_text(json.dumps(dict(time=time.time(),status='additional evaluation aggregation, not new model selection',unit='one frozen validation choice per independent training-validation dataset; 12 L4-A systems',source='selected_by_validation stored before truth reveal; no test structure labels used to pick a winner',single_trajectory_results='retain all preregistered score results; do not treat shared-fit test trajectories as independent graph discoveries'),indent=2));print(d.groupby('section')[['forced_correct','primary_guard_declared','supplementary_guard_declared','oracle_model_correct','oracle_effective_graph_exact']].agg(['sum','count','mean']).round(4).to_string());print(d[d.section=='F_positive'].to_string(index=False))
if __name__=='__main__':main()

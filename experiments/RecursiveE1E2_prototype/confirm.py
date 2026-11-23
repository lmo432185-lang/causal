"""Independent confirmation; Holm across75 prespecified selected-candidate tests."""
import json,time,hashlib
import numpy as np,pandas as pd
from scipy.stats import t as student_t
from settings import *
from core import *
from fit import features_for
from io_model import load_model

def main():
 freeze=json.loads((ROOT/'MODELS_FROZEN.json').read_text());assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in freeze['files'].items());rows=[];caseinfo={}
 comparisons=[('initial','current','initial_support'),('W1_split','W1_parent','Q1_W1_dynamics'),('witness','initial','Q2_X_witness'),('recursive','initial','Q3_vs_initial'),('recursive','long','Q3_vs_long')]
 for j in json.loads((ROOT/'public_jobs.json').read_text()):
  case=j['case'];m=load_model(OUT/case/'model.json');x=np.load(DATA/case/'confirmation_X.npy');q=feature_bundle(x,m['short'],m['long'],m['recursive_encoder'],m['B'],m['match']);f=features_for(q,m);scores={k:nll_series(m['heads'][k],v,q['Wfuture'] if k.startswith('W1_') else q['future']) for k,v in f.items()};base=affine(f['initial']);ar=q['A'][:,:,m['candidate_index']:m['candidate_index']+1]-base@m['A_base_regression'];br=q['B'][:,:,:m['rank_B']]-base@m['B_base_regression'];bw=br@m['B_axis'];pc=float(np.corrcoef(ar.ravel(),bw.ravel())[0,1]) if m['rank_B'] else 0.;fraction=float(np.var(ar)/max(np.var(q['A'][:,:,m['candidate_index']]),1e-20));caseinfo[case]=dict(partial_corr=float(abs(pc)),conditional_residual_fraction=fraction,rank_B=m['rank_B'],rank_initial=m['rank_initial'])
  for better,baseline,label in comparisons:
   diff=(scores[baseline]-scores[better]).mean(1);se=float(diff.std(ddof=1)/np.sqrt(len(diff)));stat=float(diff.mean()/max(se,1e-15));pv=float(student_t.sf(stat,len(diff)-1));rows.append(dict(case=case,comparison=label,gain_nats_per_step=float(diff.mean()),SE_trajectory=se,pvalue=pv))
 pvals=np.array([r['pvalue'] for r in rows]);order=np.argsort(pvals);adjusted=np.maximum.accumulate([(len(rows)-i)*pvals[k] for i,k in enumerate(order)]);adj=np.empty(len(rows));adj[order]=np.minimum(adjusted,1.)
 for r,a in zip(rows,adj):r['Holm_pvalue']=float(a);r['positive_confirmed']=bool(a<.05 and r['gain_nats_per_step']>=MIN_GAIN)
 tab=pd.DataFrame(rows);dec=[]
 for case,info in caseinfo.items():
  flags=tab[tab.case==case].set_index('comparison').positive_confirmed.to_dict();initial=bool(info['rank_initial']>=1 and flags['initial_support']);q1=bool(flags['Q1_W1_dynamics'] and info['conditional_residual_fraction']>=MIN_RESIDUAL_FRACTION);q2=bool(flags['Q2_X_witness'] and info['rank_B']>=2 and info['partial_corr']>=MIN_PARTIAL_CORR);q3=bool(flags['Q3_vs_initial'] and flags['Q3_vs_long']);dec.append(dict(case=case,**info,initial_state_supported=initial,Q1=q1,Q2=q2,Q3=q3,accept_new_coordinate=bool(initial and q1 and q2 and q3)))
 tab.to_csv(OUT/'confirmation_tests.csv',index=False);pd.DataFrame(dec).to_csv(OUT/'decisions.csv',index=False);files={str((OUT/f).relative_to(ROOT)):hashlib.sha256((OUT/f).read_bytes()).hexdigest() for f in ['confirmation_tests.csv','decisions.csv']};(ROOT/'DECISIONS_FROZEN.json').write_text(json.dumps(dict(time=time.time(),files=files,Holm_family_size=len(rows),test_X_and_hidden_states_not_loaded=True),indent=2));print('Confirmation complete',len(rows),'tests;',sum(v['accept_new_coordinate'] for v in dec),'accepted')
if __name__=='__main__':main()

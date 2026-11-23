"""Read-only freeze audit and deterministic prediction replay; never refit."""
import json,hashlib
import numpy as np,pandas as pd
from settings import *
from core import *
from fit import features_for
from io_model import load_model
from evaluate import score
from private_generator import one as original
from positive_generator import one as positive,SYSTEMS

def main():
 checked=0
 for name in ['REGISTERED','MODELS_FROZEN','DECISIONS_FROZEN','EVALUATION_REGISTERED','TEST_PREDICTIONS_FROZEN','POSITIVE_REGISTERED','POSITIVE_MODELS_FROZEN','POSITIVE_DECISIONS_FROZEN','POSITIVE_EVALUATION_REGISTERED','POSITIVE_TEST_PREDICTIONS_FROZEN']:
  m=json.loads((ROOT/(name+'.json')).read_text())
  for f,h in m['files'].items():
   assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,(name,f)
   checked+=1
 old=json.loads((ROOT/'REGISTERED_INITIAL.json').read_text())
 for f,h in old['files'].items():
  p=ROOT/('confirm_initial.py' if f=='confirm.py' else f)
  assert hashlib.sha256(p.read_bytes()).hexdigest()==h,('original',f)
 jobs=json.loads((ROOT/'public_jobs.json').read_text())+json.loads((ROOT/'positive_public_jobs.json').read_text())
 allmetrics=pd.read_csv(OUT/'all_trajectory_metrics.csv.gz');maxdiff=0.;examplediff=0.;prefixdiff=0.;rawdiff=0.
 for j in jobs:
  case=j['case'];idx=int(case[4:6]);seed=j['seed'];s=original(idx,seed) if idx<5 else positive(SYSTEMS[idx-5],seed)
  x=s[-100:,:,-1];assert np.array_equal(x,np.load(DATA/case/'test_X.npy'))
  m=load_model(OUT/case/'model.json');q=feature_bundle(x,m['short'],m['long'],m['recursive_encoder'],m['B'],m['match']);f=features_for(q,m);f['random']=np.concatenate([f['initial'],np.random.default_rng(seed+600000).normal(size=(*f['initial'].shape[:2],1))],-1)
  saved=np.load(OUT/case/'test_predictions.npz')
  for name,feature in f.items():
   if name.startswith('W1_'):continue
   metrics,mu,_=score(m['heads'][name],feature,q['future']);expected=allmetrics[(allmetrics.case==case)&(allmetrics.method==name)].sort_values('trajectory')
   for k,v in metrics.items():maxdiff=max(maxdiff,float(np.max(abs(v-expected[k].to_numpy()))))
   examplediff=max(examplediff,float(np.max(abs(mu[0]-saved[name+'_mean']))))
  changed=x[:1].copy();changed[:,181:]+=17.;qc=feature_bundle(changed,m['short'],m['long'],m['recursive_encoder'],m['B'],m['match']);fc=features_for(qc,m)
  for name in ['initial','recursive','witness','direct2','full_X']:
   a=predict(m['heads'][name],f[name][:1,:141]);b=predict(m['heads'][name],fc[name][:,:141]);prefixdiff=max(prefixdiff,float(np.max(abs(a-b))))
  rawdiff=max(rawdiff,float(np.max(abs(apply_raw_filter(x,m['raw_filter'])-q['A'][:,:,m['candidate_index']]))))
 assert len(allmetrics)==25200 and len(jobs)==21
 assert maxdiff<1e-10 and examplediff<1e-10 and prefixdiff<1e-10 and rawdiff<1e-10
 out=dict(freeze_hashes_checked=checked,original_registration_preserved=True,cases_replayed=21,full_test_metric_rows=25200,max_metric_replay_error=maxdiff,max_example_mean_replay_error=examplediff,max_future_perturbation_prefix_error=prefixdiff,max_raw_filter_reexpression_error=rawdiff,refitted=False,passed=True)
 (ROOT/'AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()

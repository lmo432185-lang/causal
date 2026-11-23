"""Freeze verification, deterministic data regeneration and held-out numerical replay."""
import json,hashlib,numpy as np,pandas as pd
from settings import *
from model import load,features
from core import predict,nll_series,reconstruct_raw_filter,apply_raw_filter
from generator import simulate
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 hashes=0;times={}
 for label in ['PROTOCOL_REGISTERED','CODE_REGISTERED','ANALYSIS_REGISTERED','MODELS_FROZEN','DECISIONS_FROZEN','TEST_A_FROZEN','TEST_B_FROZEN']:
  m=json.loads((ROOT/(label+'.json')).read_text());times[label]=m['time']
  for f,h in m['files'].items():assert sha(ROOT/f)==h,(label,f);hashes+=1
 assert times['PROTOCOL_REGISTERED']<times['CODE_REGISTERED']<times['MODELS_FROZEN']<times['DECISIONS_FROZEN']<times['TEST_A_FROZEN']<times['TEST_B_FROZEN']
 assert times['ANALYSIS_REGISTERED']<times['DECISIONS_FROZEN']
 manifest=json.loads((ROOT/'INPUT_MANIFEST.json').read_text());systems={m['id']:{**m,'index':i} for i,m in enumerate(json.loads((ROOT/'SYSTEMS.json').read_text()))};maxraw=0.;maxprefix=0.;maxreplay=0.;datasets=0;prefix_cases=0;replay_models=0
 tables={b:pd.read_csv(OUT/f'test_{b}_metrics.csv.gz') for b in ['A','B']}
 for r in manifest['files']:
  case=r['case'];cond,seed=case.rsplit('_',1);seed=int(seed);path=DATA/(case+'.npz');assert sha(path)==r['sha256'];d=np.load(path);x=simulate(systems[cond],seed);assert np.array_equal(x[:400],d['train']);assert np.array_equal(x[400:500],d['selection']);assert np.array_equal(x[500:600],d['confirmation']);assert np.array_equal(x[600:700],d['test_A']);assert np.array_equal(x[700:],d['test_B']);datasets+=1
  m=load(OUT/case/'model_n400.json');a=d['test_A'][:2];f,y=features(a,m);changed=a.copy();changed[:,181:]+=31.;g,_=features(changed,m)
  for method in METHODS:maxprefix=max(maxprefix,float(np.max(abs(predict(m['heads'][method],f[method][:,:141])-predict(m['heads'][method],g[method][:,:141])))))
  rf=reconstruct_raw_filter(m['short'],m['rec'],m['candidate']);raw=apply_raw_filter(a,rf);candidate=f['recursive'][:,:,-1];maxraw=max(maxraw,float(np.max(abs(raw-candidate))));prefix_cases+=1
  if seed==1011:
   for n in SIZES:
    m=load(OUT/case/f'model_n{n}.json')
    for batch in ['A','B']:
     f,y=features(d['test_'+batch],m);tab=tables[batch]
     for method in f:
      score=nll_series(m['heads'][method],f[method],y).mean(1);z=tab[(tab.case==case)&(tab.n==n)&(tab.method==method)].sort_values('trajectory').NLL.to_numpy();assert len(z)==100;maxreplay=max(maxreplay,float(np.max(abs(score-z))))
     if n==400:
      example=np.load(OUT/case/f'example_{batch}.npz');assert np.array_equal(example['X'],d['test_'+batch][0])
      for method in f:maxreplay=max(maxreplay,float(np.max(abs(predict(m['heads'][method],f[method])[0]-example[method+'_mean']))))
    replay_models+=1
 assert all(len(v)==363000 for v in tables.values());assert maxraw<1e-9 and maxprefix<1e-9 and maxreplay<1e-9
 out=dict(passed=True,freeze_hashes_verified=hashes,chronology_verified=True,datasets_regenerated=datasets,whole_data_split_equality=True,prefix_cases_checked=prefix_cases,test_models_replayed=replay_models,test_batch_rows=363000,max_prefix_error=maxprefix,max_raw_filter_error=maxraw,max_NLL_or_example_replay_error=maxreplay,refitted=False)
 (ROOT/'AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()

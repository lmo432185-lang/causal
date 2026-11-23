"""Independent confirmation X; no test arrays or truth read."""
import json,time,hashlib
import numpy as np,pandas as pd
from scipy.stats import ttest_1samp
from settings import *
from model import load,features
from core import nll_series

def holm(p):
 p=np.asarray(p);order=np.argsort(p);q=np.empty(len(p));q[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))));return q

def main():
 mark=json.loads((ROOT/'MODELS_FROZEN.json').read_text())
 for f,h in mark['files'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
 rows=[]
 for job in json.loads((ROOT/'jobs.json').read_text()):
  x=np.load(DATA/(job['case']+'.npz'))['confirmation']
  for n in SIZES:
   m=load(OUT/job['case']/f'model_n{n}.json');f,y=features(x,m);scores={k:nll_series(m['heads'][k],v,y).mean(1) for k,v in f.items()}
   for method in METHODS:
    parent='W1' if method=='recursive' else 'first_'+method.removeprefix('direct_')
    for label,delta in [('second',scores[parent]-scores[method]),('first',scores['current']-scores[parent])]:
     p=float(ttest_1samp(delta,.005,alternative='greater').pvalue);rows.append(dict(case=job['case'],seed=job['seed'],n=n,method=method,test=label,gain=float(delta.mean()),p=p))
  print('CONFIRMED',job['case'],flush=True)
 df=pd.DataFrame(rows)
 for label in ['second','first']:
  at=df.test==label;df.loc[at,'p_holm']=holm(df.loc[at,'p']);assert sum(at)==1320
 df['accepted']=(df.gain>=.005)&(df.p_holm<.05);df.to_csv(OUT/'confirmation.csv',index=False)
 p=OUT/'confirmation.csv';(ROOT/'DECISIONS_FROZEN.json').write_text(json.dumps(dict(time=time.time(),files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()},model_freeze_sha256=hashlib.sha256((ROOT/'MODELS_FROZEN.json').read_bytes()).hexdigest()),indent=2));print('DECISIONS FROZEN',flush=True)
if __name__=='__main__':main()

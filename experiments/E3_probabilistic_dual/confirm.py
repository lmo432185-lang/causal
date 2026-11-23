import json,time,hashlib
import numpy as np,pandas as pd
from scipy.stats import t as tdist,ttest_1samp
from concurrent.futures import ProcessPoolExecutor,as_completed
from settings import *
from producer import simulate
from ssm import unpack,selection_score

def holm(p):
 p=np.asarray(p);p=np.nan_to_num(p,nan=1.);ix=np.argsort(p);q=np.empty_like(p);q[ix]=np.minimum(1,np.maximum.accumulate(p[ix]*(len(p)-np.arange(len(p)))));return q

def work(j):
 x=simulate(j['index'],j['seed'],0,880)[840:];rows=[];graphs={}
 for n in SIZES:
  m=unpack(json.loads((OUT/j['case']/f'model_{n}.json').read_text()));scores={name:np.array([selection_score(x,z) for z in m[name]['models']]) for name in ['direct','recursive','witness']};graph={}
  for route in ['direct','recursive']:
   base=m['rank_'+route];allowed=[]
   for k in range(4):
    good=True
    for name in [route,'witness']:
     delta=scores[name][base]-scores[name][k];gain=float(delta.mean());se=float(delta.std(ddof=1)/np.sqrt(len(delta)));lower=gain-tdist.ppf(.95,len(delta)-1)*se
     if lower<-.005 and k!=base:good=False
    if good:allowed.append(k)
    if k:
     for name in [route,'witness']:
      delta=scores[name][k-1]-scores[name][k];p=float(ttest_1samp(delta,.005,alternative='greater').pvalue);rows.append(dict(case=j['case'],seed=j['seed'],n=n,route=route,line=name,K=k,gain=float(delta.mean()),p=p))
   graph[route]=dict(initial_rank=base,allowed=allowed)
  graphs[str(n)]=graph
 (OUT/j['case']/'graph.json').write_text(json.dumps(graphs,indent=2));print('CONFIRMED',j['case'],flush=True);return rows

def main():
 for f,h in json.loads((ROOT/'MODELS_FROZEN.json').read_text())['files'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
 rows=[];js=json.loads((ROOT/'jobs.json').read_text())
 with ProcessPoolExecutor(max_workers=2) as p:
  for f in as_completed([p.submit(work,j) for j in js]):rows.extend(f.result())
 df=pd.DataFrame(rows);df['p_holm']=holm(df.p);df['supported']=(df.gain>=.005)&(df.p_holm<.05);df.to_csv(OUT/'confirmation.csv',index=False)
 files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*/graph.json')};files['results/confirmation.csv']=hashlib.sha256((OUT/'confirmation.csv').read_bytes()).hexdigest();(ROOT/'DECISIONS_FROZEN.json').write_text(json.dumps(dict(time=time.time(),files=files),indent=2));print('DECISIONS FROZEN',len(df),flush=True)
if __name__=='__main__':main()

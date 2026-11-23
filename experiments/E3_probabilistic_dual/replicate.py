"""Separately registered follow-up: fresh seeds, primary n100, EM5 and EM25."""
import json,time,hashlib,gzip,csv
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import settings
ROOT=settings.ROOT;settings.SIZES=[100]
import producer,ssm,learn,confirm,evaluate
from ssm import unpack,pack,em,selection_score
BASE=ROOT/'replication'
JOBS=[dict(case=m['id']+'_'+str(seed),index=i,seed=seed) for i,m in enumerate(producer.SYSTEMS) for seed in range(3031,3041)]

def fitone(j):
 x=producer.simulate(j['index'],j['seed'],0,880);m=learn.fit_all(x[:400],x[400:800],x[800:840],100,j['seed']);m['seed']=j['seed'];m['case']=j['case'];m['peak_RSS_kib']=0
 for budget in ['em5','em25']:
  if budget=='em25':
   tic=time.process_time()
   for bank in ['direct','recursive','witness']:
    xx=x[400:500] if bank=='witness' else x[:100]
    for k,z in enumerate(m[bank]['models']):
     scale=z['scale'];oldhistory=z.get('EM_loglik',[]);zz=em(xx/scale,z,iters=20);zz['scale']=scale;zz['K']=k;zz['EM5_history']=oldhistory;m[bank]['models'][k]=zz
    m[bank]['selection_scores']=np.array([selection_score(x[800:840],z) for z in m[bank]['models']])
   m['rank_direct']=learn.initial_rank(m['direct'],m['witness']);m['rank_recursive']=learn.initial_rank(m['recursive'],m['witness']);m['continued_EM_cpu']=time.process_time()-tic
  d=BASE/budget/j['case'];d.mkdir(parents=True,exist_ok=True);p=d/'model_100.json';assert not p.exists();p.write_text(json.dumps(pack(m)))
 print('REPLICATION FIT',j['case'],flush=True)

def main():
 reg=json.loads((ROOT/'REPLICATION_REGISTERED.json').read_text())
 for f,h in reg['files'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
 BASE.mkdir(exist_ok=True)
 with ProcessPoolExecutor(max_workers=2) as pool:
  for f in as_completed([pool.submit(fitone,j) for j in JOBS]):f.result()
 for budget in ['em5','em25']:
  out=BASE/budget;files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*/model_100.json')};assert len(files)==170;(out/'MODELS_FROZEN.json').write_text(json.dumps(dict(time=time.time(),files=files),indent=2));confirm.OUT=out;confirm.SIZES=[100];rows=[]
  with ProcessPoolExecutor(max_workers=2) as pool:
   for f in as_completed([pool.submit(confirm.work,j) for j in JOBS]):rows.extend(f.result())
  import pandas as pd
  df=pd.DataFrame(rows);df['p_holm']=confirm.holm(df.p);df['supported']=(df.gain>=.005)&(df.p_holm<.05);df.to_csv(out/'confirmation.csv',index=False);files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*/graph.json')};files[str((out/'confirmation.csv').relative_to(ROOT))]=hashlib.sha256((out/'confirmation.csv').read_bytes()).hexdigest();(out/'DECISIONS_FROZEN.json').write_text(json.dumps(dict(time=time.time(),files=files),indent=2));evaluate.OUT=out;evaluate.SIZES=[100]
  for batch in ['A','B']:
   path=out/f'test_{batch}.csv.gz';assert not path.exists();count=0;ih={}
   with gzip.open(path,'wt') as stream:
    writer=None
    with ProcessPoolExecutor(max_workers=2) as pool:
     fs={pool.submit(evaluate.work,j,batch):j for j in JOBS}
     for f in as_completed(fs):
      rows,h=f.result();ih[fs[f]['case']]=h
      if writer is None:writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader()
      writer.writerows(rows);count+=len(rows)
   (out/f'TEST_{batch}_FROZEN.json').write_text(json.dumps(dict(time=time.time(),rows=count,input_hashes=ih,files={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()}),indent=2));print('REPLICATION TEST FROZEN',budget,batch,count,flush=True)
 print('REPLICATION ALL DONE',flush=True)
if __name__=='__main__':main()

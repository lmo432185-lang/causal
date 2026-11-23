import json,time,hashlib,csv,gzip
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
from settings import *
from producer import simulate,digest
from ssm import unpack
from predict import infer,score

def work(j,batch):
 x=simulate(j['index'],j['seed'],1 if batch=='A' else 2,40);rows=[];graph=json.loads((OUT/j['case']/'graph.json').read_text());examples={}
 for n in SIZES:
  m=unpack(json.loads((OUT/j['case']/f'model_{n}.json').read_text()));cache={}
  for group in GROUPS:
   result=infer(x,m,graph[str(n)],group,cache);metrics=score(x,result,j['seed']+n+500000)
   for i in range(len(x)):rows.append(dict(case=j['case'],seed=j['seed'],n=n,batch=batch,group=group,trajectory=i,**{k:float(v[i]) for k,v in metrics.items()}))
   if n==100 and j['seed']==3011:
    examples['X']=x[0];examples[group+'_weights']=result['weights'][0];examples[group+'_mean']=result['mean'][0];examples[group+'_cov']=result['cov'];examples[group+'_ids']=np.array([[0 if b in ['direct','recursive'] else 1,k] for b,k in result['ids']]);
    for ci,c in enumerate(result['components']):examples[f'{group}_state_mean_{ci}']=c['post']['mean'][0];examples[f'{group}_state_cov_{ci}']=c['post']['cov']
 if examples:np.savez_compressed(OUT/j['case']/f'example_{batch}.npz',**examples)
 print('TEST',batch,j['case'],flush=True);return rows,digest(x)

def main():
 for label in ['CODE_REGISTERED','MODELS_FROZEN','DECISIONS_FROZEN']:
  for f,h in json.loads((ROOT/(label+'.json')).read_text())['files'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
 js=json.loads((ROOT/'jobs.json').read_text())
 for batch in ['A','B']:
  path=OUT/f'test_{batch}.csv.gz';assert not path.exists();count=0;hashes={}
  with gzip.open(path,'wt') as stream:
   writer=None
   with ProcessPoolExecutor(max_workers=2) as pool:
    futures={pool.submit(work,j,batch):j for j in js}
    for future in as_completed(futures):
     rows,h=future.result();hashes[futures[future]['case']]=h
     if writer is None:writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader()
     writer.writerows(rows);count+=len(rows)
  files={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()};files.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob(f'*/example_{batch}.npz')});(ROOT/f'TEST_{batch}_FROZEN.json').write_text(json.dumps(dict(time=time.time(),rows=count,files=files,input_hashes=hashes,truth_not_read=True),indent=2));print('TEST FROZEN',batch,count,flush=True)
if __name__=='__main__':main()

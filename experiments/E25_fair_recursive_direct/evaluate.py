"""Fixed saved models evaluated on untouched paired test batches; no tuning."""
import json,time,hashlib,gzip,csv
import numpy as np
from scipy.special import ndtr
from scipy.stats import chi2,norm
from settings import *
from model import load,features
from core import predict,nll_series

def score(head,f,y):
 mu=predict(head,f);sd=np.sqrt(np.diag(head['cov']));r=y-mu;z=r/sd;iv=np.linalg.inv(head['cov']);mah=np.einsum('nti,ij,ntj->nt',r,iv,r);out=dict(NLL=nll_series(head,f,y).mean(1),CRPS=(sd*(z*(2*ndtr(z)-1)+2*np.exp(-z*z/2)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))).mean((1,2)),MSE=(r*r).mean((1,2)),joint_cover90=(mah<=chi2.ppf(.9,10)).mean(1),marginal_cover90=(abs(z)<=norm.ppf(.95)).mean((1,2)))
 for h in [1,2,5,10]:out[f'NLL_h{h}']=(.5*(np.log(2*np.pi*sd[h-1]**2)+z[:,:,h-1]**2)).mean(1)
 return out,mu

def main():
 for filename in ['CODE_REGISTERED.json','MODELS_FROZEN.json','DECISIONS_FROZEN.json']:
  for f,h in json.loads((ROOT/filename).read_text())['files'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
 for batch in ['A','B']:
  path=OUT/f'test_{batch}_metrics.csv.gz';assert not path.exists();count=0
  with gzip.open(path,'wt') as stream:
   writer=None
   for job in json.loads((ROOT/'jobs.json').read_text()):
    x=np.load(DATA/(job['case']+'.npz'))['test_'+batch]
    for n in SIZES:
     m=load(OUT/job['case']/f'model_n{n}.json');f,y=features(x,m);example=dict(X=x[0],time=np.arange(START,END),future=y[0]);
     for method,v in f.items():
      metrics,mu=score(m['heads'][method],v,y);example[method+'_mean']=mu[0];example[method+'_sd']=np.sqrt(np.diag(m['heads'][method]['cov']))
      for i in range(len(x)):
       row=dict(case=job['case'],seed=job['seed'],n=n,batch=batch,method=method,trajectory=i,**{k:float(z[i]) for k,z in metrics.items()})
       if writer is None:writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
       writer.writerow(row);count+=1
     if n==400 and job['seed']==1011:np.savez_compressed(OUT/job['case']/f'example_{batch}.npz',**example)
    print('TEST',batch,job['case'],flush=True)
  (ROOT/f'TEST_{batch}_FROZEN.json').write_text(json.dumps(dict(time=time.time(),rows=count,files={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()},no_model_or_rule_update=True),indent=2));print('TEST BATCH FROZEN',batch,count,flush=True)
if __name__=='__main__':main()

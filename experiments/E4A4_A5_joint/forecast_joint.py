"""Basis-independent joint future-X calibration, not physical latent-state identification."""
import json,gzip,time,numpy as np,pandas as pd
from scipy.special import logsumexp,ndtr
from scipy.stats import norm,chi2
from concurrent.futures import ProcessPoolExecutor,as_completed
from settings import *
from model import unpack
from producer import generate
from inference import mix
from predict import targets

def region(r,y):
 N,O,K,H=r['mean'].shape;z=[];ll=[]
 for k in range(K):
  L=np.linalg.cholesky(r['cov'][k]);e=y-r['mean'][:,:,k];whiten=np.einsum('oij,noj->noi',np.linalg.inv(L),e);z.append(whiten);ll.append(-.5*(whiten**2+np.log(2*np.pi)+2*np.log(np.diagonal(L,axis1=1,axis2=2))))
 z=np.stack(z,2);ll=np.stack(ll,2);lw=np.log(np.maximum(r['origin_weights'],1e-300));radius=np.zeros((N,O));logp=np.zeros((N,O))
 for h in range(H):
  lw-=logsumexp(lw,2)[:,:,None];u=np.sum(np.exp(lw)*ndtr(z[:,:,:,h]),2);radius+=norm.ppf(np.clip(u,1e-12,1-1e-12))**2;ev=logsumexp(lw+ll[:,:,:,h],2);logp+=ev;lw+=ll[:,:,:,h]
 result=dict(joint_NLL=-logp.mean(1)/H,radius=radius.mean(1))
 for lev in [.5,.8,.9,.95]:result['joint_cover'+str(int(lev*100))]=(radius<=chi2.ppf(lev,H)).mean(1)
 return result

def work(j):
 with gzip.open(OUT/(j['case']+'.json.gz'),'rt') as f:b=unpack(json.load(f))
 b['plans'].update(json.loads((OUT/'extra_plans.json').read_text())[j['case']]);rows=[]
 for batch,stage in [('A',11),('B',12)]:
  x=generate(j['index'],j['seed'],stage,NTEST);cache={};done={};y=targets(x)
  for name in ['J0','J1','J1_param','J2','A','B']:
   p=b['plans'][name];ids=p['ids'];key=(tuple(ids),tuple(p.get('prior') or []))
   if not ids:continue
   if key not in done:r,_=mix(x,b['models'],ids,p.get('prior'),cache);done[key]=region(r,y)
   for i in range(NTEST):rows.append(dict(case=j['case'],seed=j['seed'],batch=batch,method=name,trajectory=i,**{k:float(v[i]) for k,v in done[key].items()}))
 print('JOINT_X',j['case'],flush=True);return rows

def main():
 assert (ROOT/'TEST_B_FROZEN.json').exists();rows=[]
 with ProcessPoolExecutor(max_workers=3) as pool:
  for f in as_completed([pool.submit(work,j) for j in json.loads((ROOT/'jobs.json').read_text())]):rows.extend(f.result())
 pd.DataFrame(rows).to_csv(OUT/'joint_X_probability.csv.gz',index=False,compression='gzip');(ROOT/'JOINT_X_COMPLETE.json').write_text(json.dumps(dict(time=time.time(),no_hidden_truth_used=True,region='20-dimensional Rosenblatt chi-square',future_prefix_used_only_for_region_membership=True),indent=2));print('Joint X calibration complete',flush=True)
if __name__=='__main__':main()

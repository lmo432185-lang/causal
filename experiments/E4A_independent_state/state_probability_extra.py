"""Supplementary pre-test-registered full-joint score and Gaussian CRPS audit."""
import json,time,numpy as np,pandas as pd
from scipy.special import ndtr
from scipy.linalg import solve_discrete_lyapunov
from concurrent.futures import ProcessPoolExecutor,as_completed
from settings import *
from producer import generate,matrices,SYSTEMS
from model import unpack
from state_evaluate import fitted_post

def work(j):
 b=unpack(json.loads((OUT/j['case']/'model_5000.json').read_text()));meta=SYSTEMS[j['index']];fam=meta['family'];kind='factor1' if fam=='A' else 'memory2' if fam=='D' else 'factor2';F,Q,C,R=matrices(meta);oracle=dict(F=F,Q=Q,C=C,R=R,P=solve_discrete_lyapunov(F,Q),scale=1.,a=0.);rows=[]
 for batch,stage in [('A',1),('B',2)]:
  x,y=generate(j['index'],j['seed'],stage,NTEST,truth=True)
  if fam=='D':y=y[:,:,:1]
  for line in ['direct','recursive','witness','oracle']:
   m=oracle if line=='oracle' else b[line]['models'][kind]
   if m is None:continue
   mu,cov=fitted_post(x,m)
   if fam=='D':mu=mu[:,:,:1];cov=cov[:,:1,:1]
   e=y[:,200:]-mu[:,200:];v=np.maximum(np.diagonal(cov[200:],axis1=1,axis2=2),1e-12);sd=np.sqrt(v);z=e/sd;crps=np.mean(sd*(z*(2*ndtr(z)-1)+2*np.exp(-.5*z*z)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi)),axis=(0,1));joint=.5*np.mean(np.einsum('nti,tij,ntj->nt',e,np.linalg.pinv(cov[200:]),e)+np.linalg.slogdet(cov[200:])[1]+e.shape[-1]*np.log(2*np.pi))
   for k,score in enumerate(crps):rows.append(dict(case=j['case'],seed=j['seed'],condition=meta['id'],batch=batch,line=line,coordinate=k,CRPS=float(score),joint_state_NLL=float(joint)))
 print('PROBABILITY_AUDIT',j['case'],flush=True);return rows

def main():
 assert (ROOT/'TEST_B_FROZEN.json').exists();rows=[]
 with ProcessPoolExecutor(max_workers=3) as pool:
  for f in as_completed([pool.submit(work,j) for j in json.loads((ROOT/'jobs.json').read_text())]):rows.extend(f.result())
 pd.DataFrame(rows).to_csv(OUT/'state_probability_extra.csv',index=False);(ROOT/'STATE_EXTRA_COMPLETE.json').write_text(json.dumps(dict(time=time.time(),after_tests=True),indent=2))
if __name__=='__main__':main()

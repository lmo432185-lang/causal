"""Private simulation producer; learning module never imports this file."""
import json,hashlib
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from settings import ROOT,T
BASE=json.loads((ROOT/'BASE_SYSTEMS.json').read_text());SYSTEMS=[{**m,'stress':'base'} for m in BASE]
for name in ['persistence','noise','delay','correlation','nonlinear','drift']:
 SYSTEMS.append({**next(m for m in BASE if m['id']=='S3_medium'),'id':'O_'+name,'stress':name})

def simulate(index,seed,stage,count,truth=False):
 meta=SYSTEMS[index];A=np.array(meta['A'],float);Q=np.diag(meta['Q']);stress=meta['stress']
 if stress=='persistence':A[0,0]=.5
 if stress=='noise':Q[-1,-1]*=4
 if stress=='correlation':Q[0,1]=Q[1,0]=.8*np.sqrt(Q[0,0]*Q[1,1])
 if stress=='delay':
  oldA=A.copy();A=np.zeros((5,5));A[:2,:2]=oldA[:2,:2];A[2,0]=A[2,1]=1.;A[3,2]=1.;A[4,3]=1.;Q=np.diag([.25,.25,1e-12,1e-12,Q[-1,-1]])
 P=solve_discrete_lyapunov(A,Q);rng=np.random.default_rng(np.random.SeedSequence([20261123,330,index,seed,stage]));s=rng.multivariate_normal(np.zeros(len(A)),P,size=count);states=np.empty((count,T,len(A)-1));x=np.empty((count,T));L=np.linalg.cholesky(Q)
 for t in range(T):
  if t:
   new=s@A.T
   if stress=='nonlinear':new[:,-1]=3*np.tanh((s[:,0]+s[:,1])/3)
   if stress=='drift':new[:,-1]*=1-.5*np.clip((t-80)/40,0,1)
   s=new+rng.normal(size=s.shape)@L.T
  states[:,t]=s[:,:-1];x[:,t]=s[:,-1]
 return (x,states) if truth else x

def jobs():return [dict(case=m['id']+'_'+str(seed),index=i,seed=seed) for i,m in enumerate(SYSTEMS) for seed in range(3011,3031)]
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

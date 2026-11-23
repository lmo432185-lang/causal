"""Independent analytic-strength controls, registered after primary diagnostic; no primary overwrite."""
import json,numpy as np
from scipy.linalg import solve_discrete_lyapunov
from settings import *

SYSTEMS=[dict(index=5,name='one_hidden_strong',names=['W'],A=[[.97,0],[1.,0.]],Q=[.25,4.]),dict(index=6,name='two_hidden_strong',names=['Y','Z'],A=[[.97,0,0],[0,-.97,0],[1.,1.,0.]],Q=[.25,.25,4.])]
def one(meta,seed,n=N):
 A=np.array(meta['A']);Q=np.diag(meta['Q']);P0=solve_discrete_lyapunov(A,Q);rng=np.random.default_rng(np.random.SeedSequence([20611123,meta['index'],seed]));s=rng.multivariate_normal(np.zeros(len(A)),P0,size=n);v=np.empty((n,T,len(A)))
 for t in range(T):
  if t:s=s@A.T+rng.normal(size=s.shape)*np.sqrt(np.diag(Q))
  v[:,t]=s
 return v

def main():
 public=[];private=[]
 for meta in SYSTEMS:
  for seed in SEEDS:
   case=f"case{meta['index']:02}_{seed}";d=DATA/case;assert not d.exists(),'Never overwrite primary/followup data';d.mkdir();v=one(meta,seed);offset=0
   for name,n in SPLITS.items():np.save(d/f'{name}_X.npy',v[offset:offset+n,:,-1]);offset+=n
   np.savez_compressed(d/'sealed_states.npz',states=v[:,:,:-1]);public.append(dict(case=case,seed=seed));private.append(dict(case=case,seed=seed,**meta));print('generated positive',case,flush=True)
  np.save(DATA/f"probe_case{meta['index']:02}_X.npy",one(meta,7182818,n=50)[:,:,-1])
 (ROOT/'positive_public_jobs.json').write_text(json.dumps(public,indent=2));(DATA/'sealed_positive_manifest.json').write_text(json.dumps(private,indent=2))
if __name__=='__main__':main()

"""Only the simulation producer knows true states/structure; no training import."""
import json
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from settings import *
SYSTEMS=[
 dict(name='X_only_null',names=[],A=[[.7]]),
 dict(name='one_hidden',names=['W'],A=[[.85,0],[.9,.3]]),
 dict(name='cascade',names=['Y','Z'],A=[[.95,0,0],[.9,.35,0],[0,.9,.3]]),
 dict(name='parallel',names=['Y','Z'],A=[[.95,0,0],[0,-.7,0],[.8,.8,.3]]),
 dict(name='common_driver',names=['W','Y','Z'],A=[[.95,0,0,0],[.7,.55,0,0],[.6,0,-.55,0],[0,.7,.7,.3]])]

def one(index,seed,n=N):
 meta=SYSTEMS[index];A=np.array(meta['A'],float);Q=np.eye(len(A))*.25;P0=solve_discrete_lyapunov(A,Q);rng=np.random.default_rng(np.random.SeedSequence([20601123,index,seed]));s=rng.multivariate_normal(np.zeros(len(A)),P0,size=n);states=np.empty((n,T,len(A)))
 for t in range(T):
  if t:s=s@A.T+rng.normal(0,.5,s.shape)
  states[:,t]=s
 return states

def main():
 jobs=[];sealed=[]
 for i,meta in enumerate(SYSTEMS):
  for seed in SEEDS:
   case=f'case{i:02}_{seed}';d=DATA/case;d.mkdir(parents=True,exist_ok=True);s=one(i,seed);offset=0
   for name,n in SPLITS.items():np.save(d/f'{name}_X.npy',s[offset:offset+n,:,-1]);offset+=n
   np.savez_compressed(d/'sealed_states.npz',states=s[:,:,:-1]);jobs.append(dict(case=case,seed=seed));sealed.append(dict(case=case,seed=seed,index=i,**meta));print('generated',case,flush=True)
  # Separate fixed X probe for representation stability, never used to select states.
  np.save(DATA/f'probe_case{i:02}_X.npy',one(i,7182818,n=50)[:,:,-1])
 (ROOT/'public_jobs.json').write_text(json.dumps(jobs,indent=2));(DATA/'sealed_manifest.json').write_text(json.dumps(sealed,indent=2))
if __name__=='__main__':main()

import json
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from config import *
def system(cell):
 p=cell['params'];kind=cell['kind'];delay=p['delay'] if kind!='G1' else 1;n=3 if delay==1 else 4;A=np.zeros((n,n));A[0,0]=p['a'];A[1,1]=p['c'];A[2,2]=p['b']
 if kind in ['G2','G3']:A[1,0 if delay==1 else 3]=p['alpha']
 if kind=='G4':A[0,1 if delay==1 else 3]=p['alpha']
 if kind=='G4':A[2,0]=p['g1']
 else:A[2,0]=p['g1'];A[2,1]=p['g2']
 if delay==2:A[3,1 if kind=='G4' else 0]=1.
 Q=np.diag([p['qy'],p['qz'],p['qx']]+([0.] if n==4 else []));P=solve_discrete_lyapunov(A,Q);scale=np.sqrt(P[2,2]);A[2,:]/=scale;A[2,2]*=scale;Q[2,2]/=scale**2;P=solve_discrete_lyapunov(A,Q);return A,Q,P

def generate_one(cell,seed):
 A,Q,P=system(cell);ds=int(np.random.SeedSequence([970000,int(cell['id'][4:]),seed]).generate_state(1)[0]);rng=np.random.default_rng(ds);s=rng.normal(size=(N,len(A)))@np.linalg.cholesky(P).T;v=np.random.default_rng(ds+10000000).normal(0,cell['params']['obs'],(N,T));physical=np.empty((N,T,3),dtype='float32');physical[:,0]=s[:,:3]
 for t in range(1,T):s=s@A.T+rng.normal(size=s.shape)*np.sqrt(np.diag(Q));physical[:,t]=s[:,:3]
 obs=physical[:,:,2]+v;return obs.astype('float64'),physical,dict(data_seed=ds,A=A.tolist(),Q=Q.tolist(),P=P.tolist(),X_stationary_variance=float(P[2,2]))
if __name__=='__main__':
 public=[];private=[]
 for cell in CELLS:
  for seed in SEEDS:
   case=f"{cell['id']}_{seed}";dest=D/case;dest.mkdir(exist_ok=True);x,s,meta=generate_one(cell,seed);np.save(dest/'X.npy',x);np.savez_compressed(dest/'sealed_states.npz',physical=s);public.append(dict(case=case,seed=seed,X_path=str((dest/'X.npy').relative_to(R))));private.append(dict(**cell,seed=seed,case=case,generation=meta));print('generated',case,flush=True)
 (R/'public_jobs.json').write_text(json.dumps(public,indent=2));(D/'sealed_manifest.json').write_text(json.dumps(private,indent=2))

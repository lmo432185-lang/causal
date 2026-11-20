"""Simulation generator, not importable by X-only fitting worker."""
import json,hashlib
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from settings import ROOT,DATA,N,T,SEEDS
BASE=dict(a=.8,b=.7,c=.5,alpha=.5,beta=.8,obs=.25,delay=1,qy=.25,qz=.25,qx=.25)
CELLS=[dict(tag='baseline',factor='baseline',value=0,params=BASE)]
for factor,vals,basekey in [('persistence',[.5,.95],'a'),('source_gain',[.2,.8],'alpha'),('output_gain',[.2,.5],'beta'),('measurement_noise',[.05,.75],'obs'),('delay',[2,3],'delay')]:
 for v in vals:CELLS.append(dict(tag=f'{factor}_{v}',factor=factor,value=v,params={**BASE,basekey:v}))
CELLS += [dict(tag='information_rich',factor='interaction',value=1,params={**BASE,'a':.95,'alpha':.8,'beta':.8,'obs':.05}),dict(tag='information_poor',factor='interaction',value=-1,params={**BASE,'a':.5,'alpha':.2,'beta':.2,'obs':.75,'delay':3})]

def system(p):
 d=int(p['delay']);n=3+2*(d-1);A=np.zeros((n,n));A[0,0]=p['a'];A[1,1]=p['b'];A[2,2]=p['c'];A[1,0 if d==1 else 3+d-2]=p['alpha'];A[2,1 if d==1 else 3+2*(d-1)-1]=p['beta']
 if d>1:
  A[3,0]=1;A[3+d-1,1]=1
  for k in range(1,d-1):A[3+k,3+k-1]=1;A[3+d-1+k,3+d-2+k]=1
 Q=np.diag([p['qy'],p['qz'],p['qx']]+[0.]*(n-3));P=solve_discrete_lyapunov(A,Q)
 return A,Q,float(p['obs']**2),P

def one(cell,seed):
 p=cell['params'];A,Q,R,P=system(p);num=int(cell['id'][4:]);ds=int(np.random.SeedSequence([20501120,num,seed]).generate_state(1)[0]);rng=np.random.default_rng(ds);w,V=np.linalg.eigh(P);s=rng.normal(size=(N,len(A)))@(V*np.sqrt(np.maximum(w,0))).T;phys=np.empty((N,T,3));x=np.empty((N,T));std=np.sqrt(np.diag(Q))
 for t in range(T):
  if t:s=s@A.T+rng.normal(size=s.shape)*std
  phys[:,t]=s[:,:3];x[:,t]=s[:,2]+rng.normal(0,np.sqrt(R),N)
 return x,phys,dict(data_seed=ds,A=A.tolist(),Q=Q.tolist(),R=R,P0=P.tolist())

def main():
 public=[];private=[]
 for idx,c in enumerate(CELLS):
  cell={**c,'id':f'case{idx:03}'}
  for seed in SEEDS:
   case=f'{cell["id"]}_{seed}';d=DATA/case;d.mkdir(parents=True,exist_ok=True);x,s,meta=one(cell,seed);np.save(d/'X.npy',x);np.savez_compressed(d/'sealed_states.npz',states=s);public.append(dict(case=case,seed=seed,bootstrap=idx in [0,11,12]));private.append(dict(**cell,case=case,seed=seed,generation=meta));print('generated',case,flush=True)
 (ROOT/'public_jobs.json').write_text(json.dumps(public,indent=2));(DATA/'sealed_manifest.json').write_text(json.dumps(private,indent=2))
if __name__=='__main__':main()

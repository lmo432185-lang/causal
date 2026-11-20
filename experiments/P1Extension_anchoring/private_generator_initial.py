"""Separate simulation/answer generator; no imports from fitting worker."""
import json
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from settings import *
from model import plant
BASE=dict(a=.8,b=.7,c=.5,u=.5,v=.4,qx=.25,R=.25**2,delay=1)
CELLS=[dict(tag='baseline',params=BASE,valid_order=True),dict(tag='rich',params={**BASE,'a':.95,'u':.8,'R':.05**2},valid_order=True),dict(tag='delay2',params={**BASE,'delay':2},valid_order=True),dict(tag='delay3',params={**BASE,'delay':3},valid_order=True),dict(tag='weak',params={**BASE,'u':.2,'v':.1,'R':.75**2,'delay':3},valid_order=True),dict(tag='wrong_order_control',params={**BASE,'a':.55},valid_order=False)]

def theta(p):return np.array([p['a'],p['b'],p['c'],p['u'],p['v'],p['qx'],p['R'],1.,1.,1.])
def sample(P,rng,N):
 w,V=np.linalg.eigh(P);return rng.normal(size=(N,len(P)))@(V*np.sqrt(np.maximum(w,0))).T

def paired(cell,seed):
 p=cell['params'];th=theta(p);A,Q,R,P=plant(th,p['delay'],'E0');ds=int(np.random.SeedSequence([20511201,cell['index'],seed]).generate_state(1)[0]);rng=np.random.default_rng(ds);state=sample(P,rng,N);qI=np.zeros_like(Q);qI[0,0]=INPUT_STD**2;PI=solve_discrete_lyapunov(A,qI);ri=np.random.default_rng(ds+100000);ctrl=sample(PI,ri,N);I=ri.normal(0,INPUT_STD,(N,T));noise=np.empty((N,T,3));controlled=np.empty_like(noise);Xnoise=np.empty((N,T));Rnoise=np.random.default_rng(ds+200000).normal(0,np.sqrt(R),(N,T))
 for t in range(T):
  if t:state=state@A.T+rng.normal(size=state.shape)*np.sqrt(np.diag(Q));ctrl=ctrl@A.T;ctrl[:,0]+=I[:,t-1]
  noise[:,t]=state[:,:3];controlled[:,t]=(state+ctrl)[:,:3];Xnoise[:,t]=Rnoise[:,t]
 return dict(passive=(noise,noise[:,:,2]+Xnoise,np.zeros_like(I)),active=(controlled,controlled[:,:,2]+Xnoise,I)),ds

def main():
 jobs=[];sealed=[]
 for idx,cc in enumerate(CELLS):
  cell={**cc,'index':idx}
  for seed in SEEDS:
   pair,ds=paired(cell,seed)
   for policy,(s,x,I) in pair.items():
    case=f'case{idx:03}_{policy}_{seed}';d=DATA/case;d.mkdir(parents=True,exist_ok=True);np.save(d/'X.npy',x);np.save(d/'known_input.npy',I);np.savez_compressed(d/'sealed_states.npz',states=s);times=np.array([b*BLOCK+t for b in range(T//BLOCK) for t in [19,20]]);tp=np.array([b*BLOCK+20 for b in range(T//BLOCK)]);rng=np.random.default_rng(ds+300000);meas=s[:TRAIN+VALID,:,:2]+rng.normal(0,AUX_STD,(TRAIN+VALID,T,2));np.savez_compressed(d/'sparse_auxiliary.npz',Y=meas[:,times,0],Z=meas[:,times,1],Y_pair=meas[:,tp,0],Z_pair=meas[:,tp,1],times=times,pair_times=tp);(d/'E2_external_calibration.json').write_text(json.dumps(dict(c=p['c'],assumption='idealized exact independent downstream memory calibration; one known parameter only')));jobs.append(dict(case=case,seed=seed,policy=policy));sealed.append(dict(**cell,case=case,seed=seed,policy=policy,data_seed=ds))
    print('generated',case,flush=True)
 (ROOT/'public_jobs.json').write_text(json.dumps(jobs,indent=2));(DATA/'sealed_manifest.json').write_text(json.dumps(sealed,indent=2))
if __name__=='__main__':main()

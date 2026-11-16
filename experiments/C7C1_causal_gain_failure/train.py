import numpy as np
from config import *
from physics import simulate,forcing
from detectors import nominal,cusum_bank

def background(ni):
 d=simulate(N_BG,SIGMAS[ni],1.,'none',seed_for('background',ni),zero_source=True);x=d['x'];xx=x[:,:-1].ravel();y=x[:,1:].ravel();co=np.linalg.lstsq(np.c_[xx,np.ones(len(xx))],y,rcond=None)[0];return dict(q=float(co[0]),c=float(co[1]),var=float(np.mean((y-co[0]*xx-co[1])**2)))

def fit_plant(ni,ki,bg):
 seed=seed_for('training',ni,ki);parts=[simulate(N_TRAIN,SIGMAS[ni],KTEST[ki],p,seed_for('training',ni,ki,policy=pi)) for pi,p in enumerate(POLICIES)];x=np.concatenate([d['x'] for d in parts]);I=np.concatenate([d['I'] for d in parts]);q=bg['q'];a=-np.log(q)/DT;k=(1-q)/a;r=(q*x[:,:-1]+bg['c']-x[:,1:]).ravel();results=[]
 for tau in TAUFIT:
  h,_,_,_=forcing(I,tau,q);M=np.c_[np.full(len(r),k),h.ravel()];co=np.linalg.lstsq(M,r,rcond=None)[0];err=r-M@co;results.append((np.mean(err*err),tau,co))
 var,tau,co=min(results,key=lambda v:v[0]);return dict(q=q,c=bg['c'],var=float(var),tau=float(tau),f0=float(max(co[0],.05)),gain=float(max(co[1],.05)))

def fit_scales(ni,ki,policy,model):
 pi=POLICIES.index(policy);d=simulate(N_SCALE,SIGMAS[ni],KTEST[ki],policy,seed_for('scale',ni,ki,policy=pi));e,_,_=nominal(d['x'],d['I'],model);bank=cusum_bank(e,model['var']);return np.maximum(np.quantile(bank.max(1),.95,axis=0,method='higher'),1e-6).tolist()

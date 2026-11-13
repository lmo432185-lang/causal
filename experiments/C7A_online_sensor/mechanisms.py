"""Exact deterministic response kernels; class names encode hypotheses, not labels."""
import numpy as np
from config import *
C_D=1-.75/np.e

def J(rate,L,a):
 rate=np.asarray(rate);L=np.asarray(L);d=a-rate;safe=np.where(abs(d)<1e-9,1.,d);val=np.exp(-a*L)*np.expm1(d*L)/safe
 return np.where(abs(d)<1e-9,L*np.exp(-a*L),val)

def response(model,kappa,onset,duration=np.inf,base=None,steps=T):
 # Vector of parallel candidate systems, input unit pulse or persistent step.
 kappa=np.atleast_1d(np.array(kappa,float));onset=np.broadcast_to(np.asarray(onset,float),kappa.shape);duration=np.broadcast_to(np.asarray(duration,float),kappa.shape);n=len(kappa);a=.15 if base is None else -np.log(base['q'])/DT;q=np.exp(-a*DT);kk=(1-q)/a;u=np.zeros(n);w=np.zeros(n);hh=np.empty((n,steps));uu=np.empty((n,steps));yy=np.empty((n,steps));pre=kappa/C_D
 for t in range(steps):
  time=t*DT;I=((time>=onset-1e-10)&(time<onset+duration-1e-10)).astype(float);yy[:,t]=I
  if model=='B_direct':hh[:,t]=I*kk;uu[:,t]=I;continue
  uu[:,t]=u
  if model=='A_first':
   rate=1/kappa;r=np.exp(-rate*DT);hh[:,t]=I*kk+(u-I)*J(rate,DT,a);u=I+(u-I)*r
  elif model=='C_two':
   rate=2/kappa;r=np.exp(-rate*DT);d=a-rate;j0=J(rate,DT,a);j1=((d*DT-1)*np.exp(-rate*DT)+q)/(d*d);hh[:,t]=I*kk+(u-I)*j0+rate*(w-I)*j1;u=I+((u-I)+rate*(w-I)*DT)*r;w=I+(w-I)*r
  elif model=='D_variable':
   L1=np.clip(onset+pre-time,0,DT);L2=DT-L1;r1=np.exp(-L1/pre);r2=np.exp(-L2/(pre/4));h1=I*(1-np.exp(-a*L1))/a+(u-I)*J(1/pre,L1,a);u1=I+(u-I)*r1;h2=I*(1-np.exp(-a*L2))/a+(u1-I)*J(4/pre,L2,a);hh[:,t]=np.exp(-a*L2)*h1+h2;u=I+(u1-I)*r2
  else:raise ValueError(model)
 return hh,uu,yy

def library(base,steps=T):
 out={}
 for model in LABELS:
  grid=np.array([1.]) if model=='B_direct' else KGRID;k,on=np.meshgrid(grid,ONSETS,indexing='ij');h,u,y=response(model,k.ravel(),on.ravel(),base=base,steps=steps);out[model]=dict(h=h,u=u,y=y,k=k.ravel(),on=on.ravel())
 return out

def active_library(base,steps=T):
 out={}
 for model in LABELS:
  grid=np.array([1.]) if model=='B_direct' else KGRID;hs=[];us=[];pis=[]
  for pi,(amp,start,duration) in enumerate(INPUT_DESIGNS):
   h,u,y=response(model,grid,np.full(len(grid),start),duration,base,steps);hs.append(amp*h);us.append(amp*u);pis.extend([pi]*len(grid))
  out[model]=dict(h=np.concatenate(hs),u=np.concatenate(us),protocol=np.array(pis))
 return out

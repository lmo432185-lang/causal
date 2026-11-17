"""Pre-run implementation check on unrelated random synthetic arrays, not experiment test outcomes."""
import json,time
from pathlib import Path
import numpy as np
from ssm import *
root=Path(__file__).parent;results=[]
for kind,delay,oracle in [('G2',1,False),('G3',2,False),('G3',2,True),('single',1,False)]:
 p=initialize(kind,delay,918,oracle);H,R=obs_matrix(p);N=4;T=180;y=np.random.default_rng(919).normal(size=(N,T,len(H)));fast=estep(y,p);A,Q=p['A'],p['Q'];P=p['P0'].copy();n=len(A);mean=np.zeros((N,n));mf=[];PF=[];PP=[];ll=0.
 for t in range(T):
  if t:mean=mean@A.T;P=A@PF[-1]@A.T+Q
  PP.append(P.copy());S=H@P@H.T+R;K=P@H.T@np.linalg.inv(S);er=y[:,t]-mean@H.T;ll-=.5*(np.einsum('ni,ij,nj->n',er,np.linalg.inv(S),er)+np.linalg.slogdet(S)[1]+len(H)*np.log(2*np.pi)).sum();mean+=er@K.T;PF.append(P-K@H@P);mf.append(mean.copy())
 mf=np.stack(mf,1);ms=mf.copy();PS=np.array(PF)
 for t in range(T-2,-1,-1):
  J=PF[t]@A.T@np.linalg.pinv(PP[t+1],hermitian=True,rcond=1e-12);ms[:,t]=mf[:,t]+(ms[:,t+1]-mf[:,t]@A.T)@J.T;PS[t]=PF[t]+J@(PS[t+1]-PP[t+1])@J.T
 err=max(float(np.max(abs(mf-fast['filtered']))),float(np.max(abs(ms-fast['smoothed']))),float(np.max(abs(PS-fast['cov_smooth']))));ld=abs(ll-fast['loglik']);assert err<1e-8 and ld<1e-7,(kind,delay,oracle,err,ld);results.append(dict(kind=kind,delay=delay,oracle=oracle,state_cov_max_error=err,loglik_error=ld))
# Mirrored G2/G4 EM paths are equal even under independently executed EM.
y=np.random.default_rng(921).normal(size=(12,100));p=initialize('G2',1,923);q=mirror(p)
for _ in range(5):
 e=estep(y,p);f=estep(y,q);assert abs(e['loglik']-f['loglik'])<1e-7;p=mstep(e,p);q=mstep(f,q);assert np.max(abs(mirror(p)['A']-q['A']))<1e-8
(root/'DEVELOPMENT_VALIDATION.json').write_text(json.dumps(results,indent=2));print(results);print('independent mirrored EM check PASS')

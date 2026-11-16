import numpy as np
from scipy.signal import fftconvolve
from scipy.special import logsumexp
from config import *

def bins(n):return np.arange(3,int(25*(n-1)*DT/(2*np.pi))+1,3)
def periodogram(x):
 n=x.shape[1];w=np.hanning(n);return abs(np.fft.rfft(x*w[None,:],axis=1)[:,bins(n)])**2/np.sum(w*w)

def ou_basis(rate,n):
 w=np.hanning(n);aw=fftconvolve(w,w[::-1],mode='full')[n-1:]/np.sum(w*w);cov=np.exp(-rate*DT*np.arange(n))/(2*rate);return (2*np.fft.rfft(cov*aw).real-cov[0])[bins(n)]

def library(qx,n):
 rates=np.unique(np.r_[AGRID,CGRID,B]);basis={float(r):ou_basis(r,n) for r in rates};sb=basis[B];templates=[];meta=[];classes=[]
 def add(kind,a,c,beta1,gamma,eta,body):
  templates.append(np.maximum(qx*eta*sb+body,1e-14));meta.append([a,c,beta1,gamma,qx*eta]);classes.append(CLASS.index(kind))
 for eta in NOISE_FACTORS:
  add('none',1.,1.,0.,0.,eta,0)
  for a in AGRID:
   ab=(basis[a]-sb)/(B*B-a*a)
   for gain in GAINS:add('direct',a,1.,gain,0.,eta,gain*gain*ab)
   for c in CGRID:
    acb=basis[a]/((c*c-a*a)*(B*B-a*a))+basis[c]/((a*a-c*c)*(B*B-c*c))+sb/((a*a-B*B)*(c*c-B*B))
    for gamma in GAINS:add('indirect',a,c,0.,gamma,eta,gamma*gamma*acb)
    for beta1 in MIX_B:
     for gamma in MIX_G:add('mixed',a,c,beta1,gamma,eta,beta1*beta1*ab+(2*beta1*gamma*c+gamma*gamma)*acb)
 f=np.asarray(templates);return dict(inv=(1/f).T,logsum=np.log(f).sum(1),meta=np.asarray(meta),classes=np.asarray(classes),n=n)

def infer(x,bank):
 I=periodogram(x);ll=-I@bank['inv']-bank['logsum'][None,:];ev=[];best=[];profile=[]
 for k in range(4):
  ids=np.flatnonzero(bank['classes']==k);v=ll[:,ids];ev.append(logsumexp(v,axis=1)-np.log(len(ids)));idx=v.argmax(1);best.append(ids[idx]);profile.append(v[np.arange(len(x)),idx])
 ev=np.stack(ev,1);profile=np.stack(profile,1);p=np.exp(ev-logsumexp(ev,1)[:,None]);pred=p.argmax(1);idx=np.stack(best,1)[np.arange(len(x)),pred];gap=profile[np.arange(len(x)),pred]-np.max(np.where(np.arange(4)[None,:]==pred[:,None],-np.inf,profile),axis=1);declare=(pred!=0)&(p.max(1)>=CONF)&(gap>=PROFILE_GAP)
 return dict(p=p,pred=pred,declare=declare,gap=gap,theta=bank['meta'][idx],index=idx,loglik_profile=profile)

def diffusion_from_x(x):
 r=np.exp(-B*DT);den=(3-4*r+r*r)/B;return float(np.mean(np.diff(x,n=2,axis=1)**2)/den)

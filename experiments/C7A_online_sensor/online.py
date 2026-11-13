"""Streaming sufficient statistics. No truth or future observations in interface."""
import numpy as np
from scipy.special import logsumexp,log_ndtr
from config import *

def stream(x,bank,bg,amp_scale,V=None,Vvar=None,sensor='u',classify=True):
 n=len(x);steps=x.shape[1]-1;updates=steps//BLOCK;score=np.zeros((n,updates));post=np.zeros((n,updates,len(bank))) if classify else None
 states=[]
 for m,bb in bank.items():
  # Sensors arrive at current data time X_j, not at the beginning of the next interval.
  g=bb[sensor][:,1:steps+1] if V is not None else None
  h=bb['h'][:,:steps];H=np.cumsum(h*h/bg['var']+(g*g/Vvar if g is not None else 0),axis=1)[:,BLOCK-1::BLOCK]
  states.append((bb,h,g,H,np.zeros((n,len(h)))))
 for u in range(updates):
  start=u*BLOCK;end=start+BLOCK;time=end*DT;le=[];mx=np.zeros(n)
  z=-(x[:,start+1:end+1]-bg['q']*x[:,start:end]-bg['c'])
  for bb,h,g,H,b in states:
   b+=z@h[:,start:end].T/bg['var']
   if V is not None:b+=V[:,start:end]@g[:,start:end].T/Vvar
   keep=bb['on']<=time-CLOCK+1e-10
   if not keep.any():
    le.append(np.zeros(n));continue
   bv=b[:,keep];hv=H[keep,u];mx=np.maximum(mx,np.max(np.maximum(bv,0)**2/np.maximum(hv,1e-300),axis=1))
   if classify:
    aa=hv+1/amp_scale**2;logbf=np.log(2)-np.log(amp_scale)-.5*np.log(aa)[None,:]+bv*bv/(2*aa[None,:])+log_ndtr(bv/np.sqrt(aa)[None,:]);le.append(logsumexp(logbf,axis=1)-np.log(len(hv)))
  score[:,u]=mx
  if classify:
   ll=np.stack(le,1);post[:,u,:]=np.exp(ll-logsumexp(ll,axis=1)[:,None])
 return score,post

def alarm(score,threshold):
 hit=(score[:,1:]>threshold)&(score[:,:-1]>threshold);has=hit.any(1);idx=hit.argmax(1)+1
 return np.where(has,(idx+1)*CLOCK,np.nan)

def commit(score,post,threshold):
 at=alarm(score,threshold);pred=post.argmax(2);conf=post.max(2)>=ABSTAIN;time=np.arange(1,post.shape[1]+1)*CLOCK
 ok=(pred[:,1:]==pred[:,:-1])&conf[:,1:]&conf[:,:-1]&(time[None,1:]>=at[:,None]);has=ok.any(1);idx=ok.argmax(1)+1
 return at,np.where(has,(idx+1)*CLOCK,np.nan),np.where(has,pred[np.arange(len(pred)),idx],-1)

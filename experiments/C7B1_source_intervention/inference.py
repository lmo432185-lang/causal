import numpy as np
from scipy.special import logsumexp
from config import *
from physics import model_bank

def stream(x,command,bg,V=None,Vvar=None,passive=False):
 """Known registered command, no true identity/tau/onset or future data supplied."""
 n=len(x);steps=x.shape[1]-1;updates=steps//BLOCK
 if passive:
  # Exact theorem: both hypotheses induce the same passive X and V_U likelihood.
  return np.full((n,updates,2),.5),None
 h,u=model_bank(command,bg);r=-(x[:,1:]-bg['q']*x[:,:-1]-bg['c']);ll=np.zeros((n,len(KGRID)+1));energy=np.zeros(n);p=np.empty((n,updates,2));adequacy=np.empty((n,updates));channels=1+(V is not None)
 for clock in range(updates):
  start=clock*BLOCK;end=start+BLOCK;rr=r[:,None,start:end];hh=h[:,:,start:end]
  ll+=np.sum(rr*hh-.5*hh*hh,axis=2)/bg['var'];energy+=np.sum(r[:,start:end]**2,axis=1)/bg['var']
  if V is not None:
   vv=V[:,None,start:end];gg=u[:,:,start:end];ll+=np.sum(vv*gg-.5*gg*gg,axis=2)/Vvar;energy+=np.sum(V[:,start:end]**2,axis=1)/Vvar
  ev=np.stack([logsumexp(ll[:,:-1],axis=1)-np.log(len(KGRID)),ll[:,-1]],axis=1);p[:,clock,:]=np.exp(ev-logsumexp(ev,axis=1)[:,None]);df=end*channels;best_residual=energy-2*np.max(ll,axis=1);adequacy[:,clock]=(best_residual-df)/np.sqrt(2*df)
 return p,adequacy

def commit(p):
 pred=p.argmax(2);conf=p.max(2)>=CONF;ok=(pred[:,1:]==pred[:,:-1])&conf[:,1:]&conf[:,:-1];has=ok.any(1);idx=ok.argmax(1)+1
 return np.where(has,(idx+1)*CLOCK,np.nan),np.where(has,pred[np.arange(len(p)),idx],-1)

def anomaly_alarm(score,threshold):
 # This is an expected-input goodness-of-fit flag, not a source-activation alarm.
 hit=(score[:,1:]>threshold)&(score[:,:-1]>threshold);has=hit.any(1);idx=hit.argmax(1)+1
 return np.where(has,(idx+1)*CLOCK,np.nan)

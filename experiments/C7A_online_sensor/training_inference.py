import numpy as np
from scipy.special import logsumexp,log_ndtr
from config import *

def evidence(x,bank,bg,amp_scale,V=None,Vvar=None,sensor='u',details=False):
 steps=x.shape[1]-1;z=-(x[:,1:]-bg['q']*x[:,:-1]-bg['c']);out=[];meta=[]
 for model in bank:
  hh=bank[model]['h'][:,:steps];b=z@hh.T/bg['var'];H=np.sum(hh*hh,axis=1)/bg['var']
  if V is not None:
   g=bank[model][sensor][:,:steps];b+=V[:,:steps]@g.T/Vvar;H=H+np.sum(g*g,axis=1)/Vvar
  aa=H+1/amp_scale**2;logbf=np.log(2)-np.log(amp_scale)-.5*np.log(aa)[None,:]+b*b/(2*aa[None,:])+log_ndtr(b/np.sqrt(aa)[None,:]);out.append(logsumexp(logbf,axis=1)-np.log(hh.shape[0]));ix=logbf.argmax(1);meta.append(dict(index=ix,amplitude=np.maximum(0,b[np.arange(len(x)),ix]/aa[ix])))
 return (np.stack(out,1),meta) if details else np.stack(out,1)

def posterior(logev):return np.exp(logev-logsumexp(logev,axis=1)[:,None])

def active_evidence(x,bank,bg,protocol=None):
 z=-(x[:,1:]-bg['q']*x[:,:-1]-bg['c']);out=[]
 for model in bank:
  h=bank[model]['h'];ll=z@h.T/bg['var']-.5*np.sum(h*h,axis=1)[None,:]/bg['var']
  if protocol is None:out.append(logsumexp(ll,axis=1)-np.log(h.shape[0]))
  else:
   ids=bank[model]['protocol'];v=np.empty(len(x))
   for pi in range(len(INPUT_DESIGNS)):
    keep=protocol==pi
    if keep.any():v[keep]=logsumexp(ll[keep][:,ids==pi],axis=1)-np.log(np.sum(ids==pi))
   out.append(v)
 return np.stack(out,1)

def confidence(p):
 pred=p.argmax(1);mx=p.max(1);c=np.cumsum(np.sort(p,axis=1)[:,::-1],axis=1);size=np.sum(c<.95,axis=1)+1
 return pred,mx,mx<ABSTAIN,size

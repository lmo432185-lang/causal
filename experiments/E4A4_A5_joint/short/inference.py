"""Each X likelihood is counted once per model alternative; never multiplied across routes."""
import numpy as np
from scipy.special import logsumexp
from settings import *
from predict import component,scores,targets,logpdf

def mix(x,models,ids,prior=None,cache=None,pool=False,update=True):
 cache={} if cache is None else cache;prior=np.full(len(ids),-np.log(len(ids))) if prior is None else np.array(prior);parts=[]
 for k in ids:
  key=(k,update)
  if key not in cache:cache[key]=component(x,models[k],update=update)
  parts.append(cache[key])
 ll=np.stack([z['post']['ll'] for z in parts],-1)
 if pool:
  lw=np.cumsum(ll.sum(0),axis=0)[None]+prior;lw=np.broadcast_to(lw,(len(x),*lw.shape[1:]))
 else:lw=np.cumsum(ll,axis=1)+prior
 if not update:lw[:,200:]=lw[:,199:200]
 w=np.exp(lw-logsumexp(lw,axis=2)[:,:,None]);t=origins(x.shape[1]);r=dict(ids=ids,weights=w,origin_weights=w[:,t],mean=np.stack([z['mean'] for z in parts],2),cov=np.stack([z['cov'] for z in parts]),parts=parts)
 posterior=prior+ll.sum((0,1));posterior-=logsumexp(posterior);return r,posterior

def validate(x,models,ids):
 r,post=mix(x,models,ids,pool=True);return float(scores(x,r)['NLL'].mean()),post

def point_score(x,m):
 c=component(x,m);return float((-logpdf(c['mean'],c['cov'],targets(x)).mean(1)/H).mean())

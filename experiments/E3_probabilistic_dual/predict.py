"""Fixed-parameter prequential Gaussian-mixture inference. No parameter fitting."""
import numpy as np
from scipy.special import logsumexp,ndtr
from settings import *
from ssm import component,target,logpdf

def layout(m,graph,group):
 route='recursive' if group in ['C','D','F'] else 'direct';fixed=group in ['E','F','G'];ks=[m['rank_'+route]] if fixed else graph[route]['allowed'];banks=[route,'witness'];ids=[(b,k) for b in banks for k in ks];pri=[]
 for b in banks:
  val=np.array([np.mean(m[b]['selection_scores'][k])+.001*k for k in ks]);p=np.exp(-20*(val-val.min()));p/=p.sum();pri.extend(.5*p)
 return ids,np.array(pri)

def weights(ll,prior,update):
 n,T,M=ll.shape;w=np.tile(prior,(n,1));out=np.empty((n,T,M))
 for t in range(T):
  if update or t<40:
   p=.99*w+.01*prior;l=np.log(np.maximum(p,1e-300))+ll[:,t];w=np.exp(l-logsumexp(l,axis=1)[:,None])
  out[:,t]=w
 return out

def infer(x,m,graph,group,cache=None):
 cache={} if cache is None else cache;ids,prior=layout(m,graph,group);online=group not in ['A','C'];rolling=group=='G';components=[]
 for b,k in ids:
  key=(b,k,online,rolling)
  if key not in cache:cache[key]=component(x,m[b]['models'][k],update=online,rolling=rolling)
  components.append(cache[key])
 ll=np.stack([c['post']['ll'] for c in components],-1);w=np.broadcast_to(prior,(len(x),x.shape[1],len(prior))).copy() if rolling else weights(ll,prior,online);ow=w[:,ORIGINS,:];mu=np.stack([c['mean'] for c in components],2);cov=np.stack([c['cov'] for c in components],0)
 return dict(ids=ids,weights=w,origin_weights=ow,mean=mu,cov=cov,components=components)

def Aabs(a,s):
 z=a/s;return 2*s*np.exp(-.5*z*z)/np.sqrt(2*np.pi)+a*(2*ndtr(z)-1)

def ensemble(mu,cov,w,U,Z):
 N,O,M,H=mu.shape;S=U.shape[-1];idx=np.sum(U[:,:,:,None]>np.cumsum(w,axis=-1)[:,:,None,:],-1);idx=np.minimum(idx,M-1);out=np.empty((N,O,S,H))
 for j in range(M):
  L=np.linalg.cholesky(cov[j]);draw=np.einsum('nosj,oij->nosi',Z,L)+mu[:,:,j,None,:];mask=idx==j;out[mask]=draw[mask]
 return out

def score(x,result,seed):
 y=target(x);mu=result['mean'];cov=result['cov'];w=result['origin_weights'];N,O,M,H=mu.shape;out={}
 ll=np.stack([np.stack([logpdf(mu[:,:,j,:h],cov[j,:,:h,:h],y[:,:,:h]) for h in HORIZONS],-1) for j in range(M)],2);lp=logsumexp(np.log(np.maximum(w,1e-300))[:,:,:,None]+ll,axis=2)
 for i,h in enumerate(HORIZONS):out[f'NLL_h{h}']=-lp[:,:,i].mean(1)/h
 out['NLL']=out['NLL_h20'];ind=np.array(HORIZONS)-1;pm=mu[:,:,:,ind];sd=np.sqrt(np.diagonal(cov,axis1=2,axis2=3))[:,:,ind].transpose(1,0,2);obs=y[:,:,ind];s1=np.sum(w[:,:,:,None]*Aabs(obs[:,:,None,:]-pm,sd[None]),axis=2);s2=0.
 for i in range(M):
  for j in range(M):s2=s2+w[:,:,i,None]*w[:,:,j,None]*Aabs(pm[:,:,i]-pm[:,:,j],np.sqrt(sd[None,:,i]**2+sd[None,:,j]**2))
 out['CRPS']=(s1-.5*s2).mean((1,2));pit=np.sum(w[:,:,:,None]*ndtr((obs[:,:,None]-pm)/sd[None]),axis=2);out['cover90']=((pit>=.05)&(pit<=.95)).mean((1,2));mean=np.sum(w[:,:,:,None]*mu,axis=2);out['MSE']=((mean-y)**2).mean((1,2));rng=np.random.default_rng(seed);U=rng.random((N,O,16));Z=rng.normal(size=(N,O,16,H));U2=rng.random((N,O,16));Z2=rng.normal(size=(N,O,16,H));s=ensemble(mu,cov,w,U,Z);s2=ensemble(mu,cov,w,U2,Z2);out['Energy']=(np.linalg.norm(s-y[:,:,None],axis=-1).mean(2)-.5*np.linalg.norm(s-s2,axis=-1).mean(2)).mean(1)
 vario=0.;total=0
 for i in range(5):
  for j in range(i+1,5):
   a,b=ind[i],ind[j];weight=1/(b-a);v=(np.sqrt(abs(y[:,:,a]-y[:,:,b]))-np.sqrt(abs(s[:,:,:,a]-s[:,:,:,b])).mean(2))**2;vario+=weight*v;total+=weight
 out['Variogram']=(vario/total).mean(1);ks=np.array([k for b,k in result['ids']]);kw=np.stack([result['weights'][:,:,ks==k].sum(2) for k in range(4)],-1);out['expected_K']=np.sum(kw*np.arange(4),axis=2)[:,40:].mean(1);mapk=np.argmax(kw,axis=2);out['MAP_K']=np.mean(mapk[:,40:],axis=1);out['K_changes']=np.sum(np.diff(mapk[:,40:],axis=1)!=0,axis=1);out['structure_entropy']=(-np.sum(kw*np.log(np.maximum(kw,1e-300)),axis=2))[:,40:].mean(1)
 return out

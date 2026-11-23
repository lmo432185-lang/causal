"""Online Gaussian state inference and full joint predictive densities."""
import numpy as np
from scipy.special import logsumexp,ndtr
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter
from scipy.stats import norm
from settings import *
from model import acov,PARAMS
import native

def kernel(m):
 k=len(m['C']);F=m['F'];Q=m['Q'];C=m['C'];A=np.zeros((k+1,k+1));A[:k,:k]=F;A[k,:k]=C@F;A[k,k]=m.get('a',0);V=np.zeros_like(A);V[:k,:k]=Q;V[:k,k]=Q@C;V[k,:k]=C@Q;V[k,k]=C@Q@C+m['R'];powers=[np.eye(k+1)]
 for h in range(H):powers.append(powers[-1]@A)
 w,U=np.linalg.eigh((V+V.T)/2);L=U*np.sqrt(np.maximum(w,0));B=np.array([z[-1] for z in powers]);noise=B[:-1]@L;G=np.zeros((H,H*(k+1)))
 for i in range(H):
  for j in range(i+1):G[i,j*(k+1):(j+1)*(k+1)]=noise[i-j]
 return B[1:],G@G.T

def component(x,m,update=True):
 scale=m['scale'];xx=x/scale;p=native.run(xx,m,stop=100000 if update else 200);p['ll']-=np.log(scale)
 if m['kind']=='obs_ar':p['ll'][:,0]=-.5*(np.log(2*np.pi*m['R']/(1-m['a']**2))+(xx[:,0]**2)/(m['R']/(1-m['a']**2)))-np.log(scale)
 t=np.array(origins(x.shape[1]));B,N=kernel(m);k=len(m['C']);mu=np.einsum('nok,hk->noh',p['mean'][:,t],B[:,:k])+xx[:,t,None]*B[None,None,:,-1];cov=np.einsum('hi,oij,kj->ohk',B[:,:k],p['cov'][t],B[:,:k])+N;return dict(mean=mu*scale,cov=cov*scale**2,post=p)

def targets(x):
 t=np.array(origins(x.shape[1]));return np.stack([x[:,t+j] for j in range(1,H+1)],-1)
def logpdf(mu,cov,y):
 r=y-mu;return -.5*(np.einsum('noi,oij,noj->no',r,np.linalg.inv(cov),r)+np.linalg.slogdet(cov)[1]+y.shape[-1]*np.log(2*np.pi))
def nll(x,m):
 c=component(x,m);return -logpdf(c['mean'],c['cov'],targets(x)).mean(1)/H

def evidence(x,m,ntrain):
 c=component(x,m);return float(c['post']['ll'][:,200:].sum()-.5*PARAMS[m['kind']]*np.log(ntrain*x.shape[1]))

def history_fit(x,v):
 g=acov(x,200);models=[]
 for order in [40,200]:
  first=g[:order].copy();first[0]+=g[0]*1e-6;phi=solve_toeplitz(first,g[1:order+1]);R=max(float(g[0]-phi@g[1:order+1]),1e-8);res=lfilter(np.r_[1.,-phi],[1.],v,axis=1)[:,200:];score=float(np.mean(.5*(np.log(2*np.pi*R)+res*res/R)));models.append(dict(order=order,phi=phi,R=R,validation_NLL=score))
 return min(models,key=lambda m:m['validation_NLL'])

def history_forecast(x,m):
 phi=m['phi'];p=len(phi);t=np.array(origins(x.shape[1]));history=np.stack([x[:,t-j] for j in range(p)],-1);means=[]
 for h in range(H):
  mu=history@phi;means.append(mu);history=np.concatenate([mu[:,:,None],history[:,:,:-1]],-1)
 psi=np.zeros(H);psi[0]=1.
 for h in range(1,H):psi[h]=sum(phi[j-1]*psi[h-j] for j in range(1,min(h,p)+1))
 L=np.zeros((H,H))
 for i in range(H):L[i,:i+1]=psi[i::-1]
 C=m['R']*L@L.T;return np.stack(means,-1),np.broadcast_to(C,(len(t),H,H))

def infer(x,bundle,group,cache):
 update=group in ['M3','M4'];main='recursive' if group in ['M2','M4'] else 'direct';ids=[];prior=[]
 for bank in [main,'witness']:
  candidates=[k for k in KINDS if bundle[bank]['models'][k] is not None];ll=np.array([bundle[bank]['log_evidence'][k] for k in candidates]);lp=ll-logsumexp(ll)-np.log(2.)
  ids.extend((bank,k) for k in candidates);prior.extend(lp)
 parts=[]
 for bank,kind in ids:
  key=(bank,kind,update)
  if key not in cache:cache[key]=component(x,bundle[bank]['models'][kind],update)
  parts.append(cache[key])
 ll=np.stack([p['post']['ll'] for p in parts],-1);cum=np.cumsum(ll,axis=1)
 if not update:cum[:,200:]=cum[:,199:200]
 lw=cum+np.array(prior);w=np.exp(lw-logsumexp(lw,axis=2)[:,:,None]);t=origins(x.shape[1]);mu=np.stack([p['mean'] for p in parts],2);cov=np.stack([p['cov'] for p in parts]);return dict(ids=ids,weights=w,origin_weights=w[:,t],mean=mu,cov=cov,parts=parts)

def Aabs(a,s):
 z=a/s;return 2*s*np.exp(-.5*z*z)/np.sqrt(2*np.pi)+a*(2*ndtr(z)-1)

def scores(x,r):
 mu=r['mean'];cov=r['cov'];w=r['origin_weights'];y=targets(x);N,O,M,H=mu.shape;out={}
 for h in HORIZONS:
  ll=np.stack([logpdf(mu[:,:,i,:h],cov[i,:,:h,:h],y[:,:,:h]) for i in range(M)],-1);out[f'NLL_h{h}']=-logsumexp(np.log(np.maximum(w,1e-300))+ll,axis=2).mean(1)/h
 out['NLL']=out['NLL_h20'];ix=np.array(HORIZONS)-1;mm=mu[:,:,:,ix];sd=np.sqrt(np.diagonal(cov,axis1=2,axis2=3))[:,:,ix].transpose(1,0,2);obs=y[:,:,ix];s1=np.sum(w[:,:,:,None]*Aabs(obs[:,:,None]-mm,sd[None]),axis=2);s2=0.
 for i in range(M):
  for j in range(M):s2+=w[:,:,i,None]*w[:,:,j,None]*Aabs(mm[:,:,i]-mm[:,:,j],np.sqrt(sd[None,:,i]**2+sd[None,:,j]**2))
 out['CRPS']=(s1-.5*s2).mean((1,2));pit=np.sum(w[:,:,:,None]*ndtr((obs[:,:,None]-mm)/sd[None]),axis=2);out['cover90']=((pit>=.05)&(pit<=.95)).mean((1,2));mean=np.sum(w[:,:,:,None]*mu,axis=2);out['MSE']=np.mean((mean-y)**2,axis=(1,2))
 # Actual central90 mixture width by deterministic bisection at the five horizons.
 lo=mm.min(2)-8*sd.max(1)[None];hi=mm.max(2)+8*sd.max(1)[None];quant=[]
 for level in [.05,.95]:
  a=lo.copy();b=hi.copy()
  for step in range(28):
   mid=(a+b)/2;cdf=np.sum(w[:,:,:,None]*ndtr((mid[:,:,None]-mm)/sd[None]),axis=2);a=np.where(cdf<level,mid,a);b=np.where(cdf>=level,mid,b)
  quant.append((a+b)/2)
 out['width90']=(quant[1]-quant[0]).mean((1,2));return out

def history_scores(x,m):
 mu,cov=history_forecast(x,m);r=dict(mean=mu[:,:,None],cov=cov[None],origin_weights=np.ones((*mu.shape[:2],1)));return scores(x,r)

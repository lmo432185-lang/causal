"""Linear predictive directions and Gaussian forecast heads; observations only."""
import numpy as np
from scipy.special import ndtr
from scipy.stats import norm,chi2
from settings import *

def base_arrays(x,p=P):
 t=np.arange(START,END);current=x[:,t,None];past=np.stack([x[:,t-j] for j in range(1,p+1)],axis=-1);future=np.stack([x[:,t+j] for j in range(1,H+1)],axis=-1);return current,past,future

def regfit(x,y):
 x=x.reshape(-1,x.shape[-1]);y=y.reshape(-1,y.shape[-1]);return np.linalg.lstsq(x,y,rcond=1e-10)[0]
def affine(x):return np.concatenate([np.ones((*x.shape[:-1],1)),x],axis=-1)
def whitening(C):
 C=(C+C.T)/2;w,V=np.linalg.eigh(C);cut=max(float(w[-1])*1e-8,1e-10);return (V/np.sqrt(np.maximum(w,cut)))@V.T

def learn(current,past,future,k=KMAX):
 b=affine(current);g=regfit(b,past);gy=regfit(b,future);f=(past-b@g).reshape(-1,past.shape[-1]);y=(future-b@gy).reshape(-1,future.shape[-1]);W=whitening(f.T@f/len(f));Y=whitening(y.T@y/len(y));u,s,v=np.linalg.svd(W@(f.T@y/len(f))@Y,full_matrices=False);E=W@u[:,:min(k,u.shape[1])];return dict(g=g,E=E,canonical_correlations=s.tolist(),future_dim=future.shape[-1])
def apply(enc,current,past):return (past-affine(current)@enc['g'])@enc['E']

def wseries(x,enc,p=P):
 # W1_t uses only X at t and earlier; prefix before p is deliberately unavailable.
 t=np.arange(p,x.shape[1]);cur=x[:,t,None];past=np.stack([x[:,t-j] for j in range(1,p+1)],axis=-1);out=np.full_like(x,np.nan);out[:,p:]=apply(enc,cur,past)[:,:,0];return out

def recursive_arrays(w):
 t=np.arange(START,END);cur=w[:,t,None];past=np.stack([w[:,t-j] for j in range(1,L+1)],axis=-1);future=np.stack([w[:,t+j] for j in range(1,H+1)],axis=-1);assert np.isfinite(past).all();return cur,past,future

def fit_head(features,targets):
 z=features.reshape(-1,features.shape[-1]);y=targets.reshape(-1,targets.shape[-1]);mu=z.mean(0);sd=np.maximum(z.std(0),1e-8);D=affine((z-mu)/sd);coef=regfit(D,y);res=y-D@coef;C=res.T@res/len(res);C=(C+C.T)/2;w,V=np.linalg.eigh(C);C=(V*np.maximum(w,max(w[-1]*1e-8,1e-10)))@V.T
 return dict(mean=mu,sd=sd,coef=coef,cov=C,effective_design_rank=int(np.linalg.matrix_rank(D,tol=1e-8)))
def predict(head,features):return affine((features-head['mean'])/head['sd'])@head['coef']
def nll_series(head,features,target):
 m=predict(head,features);r=target-m;iv=np.linalg.inv(head['cov']);return .5*(np.einsum('nti,ij,ntj->nt',r,iv,r)+np.linalg.slogdet(head['cov'])[1]+target.shape[-1]*np.log(2*np.pi))/target.shape[-1]

def choose_rank(scores):
 # Mean trajectory score and its standard error, no IID timepoint sample size.
 means=np.array([s.mean() for s in scores]);best=int(np.argmin(means));se=float(scores[best].mean(1).std(ddof=1)/np.sqrt(len(scores[best])));eligible=np.flatnonzero(means<=means[best]+se);return int(eligible[0]),dict(means=means.tolist(),best=best,one_SE=se)

def feature_bundle(x,enc_initial,enc_long,enc_rec,enc_B,enc_match):
 current,past20,future=base_arrays(x,P);_,past40,_=base_arrays(x,FULL);ws=wseries(x,enc_initial,P);wc,wp,wf=recursive_arrays(ws);a=apply(enc_rec,wc,wp);short=wc;long=apply(enc_long,current,past40)[:,:,:1];B=apply(enc_B,current,past40);match=apply(enc_match,current,past40)
 return dict(current=current,past=past40,future=future,W1=short,W1long=long,A=a,B=B,match=match,Whistory=wp,Wfuture=wf)

def reconstruct_raw_filter(initial,recursive,j):
 e=initial['E'][:,0];w=np.r_[-initial['g'][1]@e,e];cw=float(-initial['g'][0]@e);a=recursive['E'][:,j];aa=np.r_[-recursive['g'][1]@a,a];ca=float(-recursive['g'][0]@a);return dict(coeff=np.convolve(aa,w),bias=ca+cw*aa.sum())
def apply_raw_filter(x,f):
 t=np.arange(START,END);hist=np.stack([x[:,t-j] for j in range(FULL+1)],-1);return hist@f['coeff']+f['bias']

def pack(x):
 if isinstance(x,np.ndarray):return {'__array__':x.tolist()}
 if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
 if isinstance(x,list):return [pack(v) for v in x]
 if isinstance(x,(np.floating,np.integer)):return x.item()
 return x

def unpack(x):
 if isinstance(x,dict) and set(x)=={'__array__'}:return np.array(x['__array__'],float)
 if isinstance(x,dict):return {k:unpack(v) for k,v in x.items()}
 if isinstance(x,list):return [unpack(v) for v in x]
 return x

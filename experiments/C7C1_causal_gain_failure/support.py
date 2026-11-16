"""X-only switching-AR learning and retrospective event localization.
No true labels, physical coefficients, or event timing enter fit().
"""
import os
os.environ['MPLCONFIGDIR']='/workspace/work/mpl'; os.environ['XDG_CACHE_HOME']='/workspace/work/cache'
from pathlib import Path
import json,csv,time
import numpy as np
from scipy.special import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).parent; OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
DT=.05; T=500; SEEDS=[11,22,33]; LAGS=[0,10,20,40,500]
CFG=dict(dt=DT,duration=25,train=200,validation_null=100,test_event=200,test_null=200,test_short=200,training_seeds=SEEDS,noise=[.12,.35,.7],lag_seconds=[0,.5,1,2,'full'],EM_iterations=45,states=2,threshold_null_false_alarm_target=.05,simulation_equation='x[t+1]=x[t]+dt*(1.5-.15*x[t]-1.5*y[t])+sigma*sqrt(dt)*noise',training_model='two-state switching AR: dx=beta*x+mu[state]+Gaussian residual; all parameters and transitions learned from X')
def generate(n,sigma,seed,kind='mixed'):
 rng=np.random.default_rng(seed); y=np.zeros((n,T)); on=np.full(n,-1); off=on.copy(); amp=np.zeros(n)
 for i in range(n):
  if kind=='null' or (kind=='mixed' and rng.random()<.3): continue
  on[i]=rng.integers(120,241); dur=rng.integers(4,13) if kind=='short' else rng.integers(40,101); off[i]=on[i]+dur; amp[i]=rng.uniform(.8,1.2); y[i,on[i]:off[i]]=amp[i]
 x=np.zeros((n,T+1)); x[:,0]=rng.normal(10,.3,n)
 for t in range(T): x[:,t+1]=x[:,t]+DT*(1.5-.15*x[:,t]-1.5*y[:,t])+sigma*np.sqrt(DT)*rng.normal(size=n)
 return dict(x=x,y=y,on=on,off=off,amp=amp)
def emission(x,p):
 d=np.diff(x,axis=1); residual=d[:,:,None]-p['beta']*x[:,:-1,None]-p['mu']; ll=-.5*(residual**2/p['var']+np.log(2*np.pi*p['var'])); scale=ll.max(2); return np.exp(ll-scale[:,:,None]),scale

def fb(x,p):
 e,scale=emission(x,p); n,t,_=e.shape; prior=np.empty_like(e); post=np.empty_like(e); norm=np.empty((n,t)); prior[:,0]=p['pi']
 for j in range(t):
  v=prior[:,j]*e[:,j]; norm[:,j]=v.sum(1); post[:,j]=v/norm[:,j,None]
  if j<t-1: prior[:,j+1]=post[:,j]@p['A']
 b=np.ones_like(e)
 for j in range(t-2,-1,-1):
  z=(e[:,j+1]*b[:,j+1])@p['A'].T; b[:,j]=z/z.sum(1)[:,None]
 g=post*b; g/=g.sum(2)[:,:,None]
 return e,prior,post,b,g,float((np.log(norm)+scale).sum())
def fit(x,seed):
 rng=np.random.default_rng(seed); d=np.diff(x,axis=1); xx=x[:,:-1]; init=np.linalg.lstsq(np.c_[xx.ravel(),np.ones(xx.size)],d.ravel(),rcond=None)[0]; residual=d-init[0]*xx
 p=dict(beta=float(init[0]),mu=np.quantile(residual,[.12,.75])+rng.normal(0,.002,2),var=float(np.var(residual)),A=np.array([[.96,.04],[.01,.99]]),pi=np.array([.15,.85]))
 trace=[]
 for k in range(CFG['EM_iterations']):
  e,prior,post,b,g,ll=fb(x,p); trace.append(ll)
  counts=np.zeros((2,2))
  for j in range(T-1):
   v=post[:,j,:,None]*p['A'][None,:,:]*(e[:,j+1]*b[:,j+1])[:,None,:]; v/=v.sum((1,2))[:,None,None]; counts+=v.sum(0)
  # Small symmetric pseudocount prevents exact absorbing states.
  counts+=.01; A=counts/counts.sum(1)[:,None]; pi=g[:,0].mean(0); pi=np.maximum(pi,1e-6); pi/=pi.sum()
  mass=g.sum((0,1)); sx=(g*xx[:,:,None]).sum((0,1)); sy=(g*d[:,:,None]).sum((0,1)); sxx=(xx*xx).sum(); sxy=(xx*d).sum()
  mat=np.array([[sxx,sx[0],sx[1]],[sx[0],mass[0],0],[sx[1],0,mass[1]]]); rhs=np.array([sxy,sy[0],sy[1]]); co=np.linalg.solve(mat+np.eye(3)*1e-9,rhs)
  var=float((g*(d[:,:,None]-co[0]*xx[:,:,None]-co[1:])**2).sum()/d.size)
  p=dict(beta=float(co[0]),mu=co[1:],var=max(var,1e-8),A=A,pi=pi)
 order=np.argsort(p['mu']); p['mu']=p['mu'][order]; p['A']=p['A'][order][:,order]; p['pi']=p['pi'][order]
 return p,trace

def infer(x,p):
 e,prior,post,b,g,ll=fb(x,p); n,t,_=e.shape; out={0:prior[:,:,0],500:g[:,:,0]}
 # z_t inferred using X up through t+lag; e_t is first seen at t+1.
 for lag in [10,20,40]:
  beta=np.ones_like(e)
  for depth in range(lag):
   new=np.ones_like(e); new[:,-1]=e[:,-1]
   new[:,:-1]=e[:,:-1]*(beta[:,1:]@p['A'].T)
   beta=new/np.maximum(new.sum(2)[:,:,None],1e-300)
  pp=prior*beta; pp/=pp.sum(2)[:,:,None]; out[lag]=pp[:,:,0]
 return out,(e,prior,post,b,g)

def smooth_to_endpoint(e,prior,p,endpoint):
 # Endpoint is an X sample index; observations include increments [0,endpoint).
 end=min(endpoint,T); n=e.shape[0]; beta=np.ones((n,2)); probs=np.empty((n,end))
 for j in range(end-1,-1,-1):
  beta=e[:,j]*beta; beta/=beta.sum(1)[:,None]; z=prior[:,j]*beta; z/=z.sum(1)[:,None]; probs[:,j]=z[:,0]
  beta=beta@p['A'].T; beta/=beta.sum(1)[:,None]
 return probs

def first_run(mask,length=2):
 if len(mask)<length: return None
 ix=np.flatnonzero(np.convolve(mask.astype(int),np.ones(length,dtype=int),'valid')==length)
 return int(ix[0]) if len(ix) else None

def stats(v):
 a=np.asarray(v,float); a=a[np.isfinite(a)]; n=len(a)
 if not n:return dict(n=0,mean=None,median=None,ci95=None)
 se=a.std(ddof=1)/np.sqrt(n) if n>1 else 0
 return dict(n=n,mean=float(a.mean()),median=float(np.median(a)),ci95=[float(a.mean()-1.972*se),float(a.mean()+1.972*se)])
def score(case,label,out,threshold,seed,sigma):
 rows=[]
 for lag,pp in out.items():
  for i in range(len(pp)):
   on=int(case['on'][i]); off=int(case['off'][i]); true=case['y'][i]>0; active=pp[i]>=threshold
   start=first_run(active); detected=start is not None; row=dict(sigma=sigma,seed=seed,case=label,trajectory=i,lag=lag,threshold=float(threshold),detected=int(detected),onset_error=np.nan,early_alarm=0,miss=0,brier=float(np.mean((pp[i]-true)**2)),state_mae=float(np.mean(np.abs(pp[i]-true))),onset_signed=np.nan)
   if on>=0:
    # No matching window selected using truth: first inferred sustained episode wins.
    row['miss']=int(not detected); row['early_alarm']=int(detected and start<on-10)
    if detected: row['onset_error']=abs(start-on)*DT; row['onset_signed']=(start-on)*DT
   rows.append(row)
 return rows


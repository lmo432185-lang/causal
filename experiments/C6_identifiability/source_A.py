"""Experiment A only. Frozen earlier X-only HMM, strictly forward test inference."""
import os
os.environ['MPLCONFIGDIR']='/workspace/work/mpl';os.environ['XDG_CACHE_HOME']='/workspace/work/cache'
import json,csv,time
from pathlib import Path
import numpy as np
from scipy.special import expit,logsumexp
from support import fit,generate as old_generate,first_run
ROOT=Path(__file__).parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
DT=.05;NSTEP=500;T1=10.;SIGMAS=[.12,.35,.7];TAUS=[.1,.5,1.,2.];STRENGTHS=[.5,1.,1.5];SEEDS=[11,22,33]
CFG=dict(experiment='A: causal online detection only',dt=DT,duration=25,T1=T1,tau=TAUS,noise=SIGMAS,strength=STRENGTHS,n_event_per_cell=200,n_null_test_per_noise=500,n_null_calibration_per_noise=500,n_null_AR_training_per_noise=200,model_seeds=SEEDS,model='frozen two-state switching AR HMM trained in preceding retrospective experiment',target_trajectory_false_alarm=.05,persistence_samples=2,response_equations='dX=(1.5-.15 X-1.5 U)dt+sigma dW; dU=(Y-U)/tau dt',T2_definition='first 2 consecutive significant downward X deviations from learned no-event trajectory mean; threshold calibrated on whole null trajectories',T3_definition='first 2 consecutive significant negative-regime forward HMM posterior scores; threshold calibrated on same null trajectories',integration='exact joint deterministic step for X,U with exact OU process-noise variance; no Euler approximation',censoring='first alarms before T1 counted separately; no restart at true T1; non-early non-detected cells retained as failures; delay conditional on valid first alarm, capped delay assigns 15 seconds to every early/missing trial')

def simulate(n,sigma,tau,strength,seed,event=True):
 rng=np.random.default_rng(seed);x=np.empty((n,NSTEP+1));x[:,0]=rng.normal(10,.3,n);u=np.zeros_like(x); y=np.zeros((n,NSTEP));
 if event:y[:,int(T1/DT):]=strength
 a=.15;b=1.5;f=1.5;q=np.exp(-a*DT);r=np.exp(-DT/tau);j=(r-q)/(a-1/tau);k=(1-q)/a;noise_sd=sigma*np.sqrt((1-q*q)/(2*a))
 for t in range(NSTEP):
  force=y[:,t]*k+(u[:,t]-y[:,t])*j
  x[:,t+1]=q*x[:,t]+f*k-b*force+noise_sd*rng.normal(size=n)
  u[:,t+1]=y[:,t]+(u[:,t]-y[:,t])*r
 return dict(x=x,u=u,y=y)

def forward_score(x,p):
 # Every output at index t uses only X[0:t+2], i.e. data time (t+1)*dt.
 n,t=x.shape[0],x.shape[1]-1;prior=np.broadcast_to(p['pi'],(n,2)).copy();res=np.empty((n,t))
 for j in range(t):
  dx=x[:,j+1]-x[:,j];ll=-.5*((dx[:,None]-p['beta']*x[:,j,None]-p['mu'])**2/p['var']+np.log(2*np.pi*p['var']));lp=np.log(np.maximum(prior,1e-300))+ll;lp-=logsumexp(lp,axis=1)[:,None];post=np.exp(lp);res[:,j]=lp[:,0]-lp[:,1];prior=post@p['A']
 return res

def fit_normal(x):
 xx=x[:,:-1].ravel();target=x[:,1:].ravel();co=np.linalg.lstsq(np.c_[xx,np.ones(len(xx))],target,rcond=None)[0];var=np.mean((target-(co[0]*xx+co[1]))**2);return dict(q=float(co[0]),c=float(co[1]),var=float(var))
def observed_score(x,p):
 mean=x[:,0].copy();var=0.;scores=np.zeros((len(x),x.shape[1]-1))
 for j in range(scores.shape[1]):
  mean=p['q']*mean+p['c'];var=p['q']**2*var+p['var'];scores[:,j]=(mean-x[:,j+1])/np.sqrt(var)
 return scores

def threshold(scores):
 # max of pairwise minima is exactly the score of first 2-consecutive crossing.
 maxima=np.minimum(scores[:,:-1],scores[:,1:]).max(1);return float(np.quantile(maxima,.95,method='higher'))
def alarm_times(scores,thr):
 result=np.full(len(scores),np.nan)
 for i,v in enumerate(scores):
  start=first_run(v>thr)
  if start is not None:result[i]=(start+2)*DT
 return result

def statistics(values):
 v=np.asarray(values,float);v=v[np.isfinite(v)];n=len(v)
 if not n:return dict(n=0,mean=None,sd=None,median=None,ci95=None)
 se=v.std(ddof=1)/np.sqrt(n) if n>1 else 0;return dict(n=n,mean=float(v.mean()),sd=float(v.std(ddof=1)) if n>1 else 0.,median=float(np.median(v)),ci95=[float(v.mean()-1.972*se),float(v.mean()+1.972*se)])
def fields(times):
 early=np.isfinite(times)&(times<T1);valid=np.isfinite(times)&(times>=T1);delay=np.where(valid,times-T1,np.nan)
 return early,valid,delay


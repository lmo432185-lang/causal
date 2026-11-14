import numpy as np
from config import *

def response(command,kappa,base=None,direct=False):
 """Exact integral of weighted force and endpoint U for piecewise-constant source."""
 command=np.atleast_2d(command);n,steps=command.shape;kappa=np.broadcast_to(np.asarray(kappa,float),(n,));a=.15 if base is None else -np.log(base['q'])/DT;q=np.exp(-a*DT);kk=(1-q)/a;u=np.zeros(n);h=np.empty_like(command);us=np.empty_like(command)
 rate=1/kappa;d=a-rate;j=np.exp(-a*DT)*np.expm1(d*DT)/d;r=np.exp(-rate*DT)
 for t in range(steps):
  I=command[:,t]
  if direct:h[:,t]=I*kk;us[:,t]=command[:,t+1] if t+1<steps else 0
  else:h[:,t]=I*kk+(u-I)*j;u=I+(u-I)*r;us[:,t]=u
 return h,us

def make_commands(n,seed):
 out={};start=int(START/DT);stop=int(STOP/DT);step=np.zeros((n,T));step[:,start:stop]=1.;out['I1_step_hold']=step;pulse=np.zeros_like(step);pulse[:,start:start+10]=np.sqrt(8);out['I2_pulse']=pulse
 rng=np.random.default_rng(seed+17);seq=np.zeros((n,16));seq[:,0]=np.sqrt(2)
 for i in range(n):seq[i,rng.choice(np.arange(1,16),7,replace=False)]=np.sqrt(2)
 random=np.zeros_like(step);random[:,start:stop]=np.repeat(seq,5,axis=1);out['I3_random']=random;out['blank']=np.zeros_like(step);return out

def innovations(n,sigma,seed):
 rng=np.random.default_rng(seed);noise=rng.normal(size=(n,T));eta=rng.normal(size=(n,T));x0=rng.normal(10,.3,n);amp=rng.uniform(1.125,1.875,n);return noise,eta,x0,amp

def simulate(command,sigma,kappa,truth,seed,passive=False):
 n=len(command);noise,eta,x0,amp=innovations(n,sigma,seed)
 if passive:
  # Both causal models have the same continuous U and integrated force, different endogenous Y.
  I=np.zeros((n,T));I[:,int(START/DT):]=amp[:,None];h,U=response(I,kappa);end=np.arange(1,T+1)*DT;Y=np.where(end[None,:]>=START,amp[:,None],0) if truth==0 else U.copy()
 else:
  if np.ndim(truth):
   hf,uf=response(command,kappa);hd,ud=response(command,kappa,direct=True);mask=np.asarray(truth)==1;h=np.where(mask[:,None],hd,hf);U=np.where(mask[:,None],ud,uf)
  else:h,U=response(command,kappa,direct=truth==1)
  Y=np.c_[command[:,1:],np.zeros(n)]
 q=np.exp(-.15*DT);c=1.5*(1-q)/.15;sd=sigma*np.sqrt((1-q*q)/.3);x=np.empty((n,T+1));x[:,0]=x0
 for t in range(T):x[:,t+1]=q*x[:,t]+c-h[:,t]+sd*noise[:,t]
 return dict(x=x,U=U,Y=Y,eta=eta,command=command)

def healthy(n,sigma,seed):
 return simulate(np.zeros((n,T)),sigma,np.ones(n),1,seed)['x']

def model_bank(command,bg):
 n=len(command);hh=[];gg=[]
 for tau in KGRID:
  h,u=response(command,np.full(n,tau),bg);hh.append(h);gg.append(u)
 h,u=response(command,np.ones(n),bg,direct=True);hh.append(h);gg.append(u)
 return np.stack(hh,1),np.stack(gg,1)

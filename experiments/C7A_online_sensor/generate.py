import numpy as np
from config import *
from mechanisms import response

def draw(n,sigma,kappa,seed,mode='passive',train=False):
 rng=np.random.default_rng(seed);amp=rng.uniform(1.125,1.875,n);on=np.round(rng.uniform(6,14,n)/.25)*.25 if train else np.full(n,10.);kap=np.exp(rng.uniform(np.log(.05),np.log(5),n)) if train else np.full(n,kappa);protocol=rng.integers(0,len(INPUT_DESIGNS),n);x0=rng.normal(10,.3,n);noise=rng.normal(size=(n,T));eta=rng.normal(size=(n,T));q=np.exp(-.15*DT);c=1.5*(1-q)/.15;sd=sigma*np.sqrt((1-q*q)/.3);data={}
 if mode=='active':
  design=np.array([INPUT_DESIGNS[i] for i in protocol]);amp=design[:,0];on=design[:,1];duration=design[:,2]
 else:duration=np.full(n,np.inf)
 for model in LABELS:
  h,u,y=response(model,kap,on,duration,steps=T+1);x=np.empty((n,T+1));x[:,0]=x0
  for t in range(T):x[:,t+1]=q*x[:,t]+c-amp*h[:,t]+sd*noise[:,t]
  data[model]=dict(x=x,U=amp[:,None]*u[:,1:],Y=amp[:,None]*y[:,1:],eta=eta,amp=amp,on=on,k=kap,protocol=protocol)
 return data

def normal(n,sigma,seed):
 rng=np.random.default_rng(seed);q=np.exp(-.15*DT);c=1.5*(1-q)/.15;x=np.empty((n,T+1));x[:,0]=rng.normal(10,.3,n)
 for t in range(T):x[:,t+1]=q*x[:,t]+c+sigma*np.sqrt((1-q*q)/.3)*rng.normal(size=n)
 return x

def equivalent_pair(n,sigma,kappa,seed):
 a=draw(n,sigma,kappa,seed)['A_first'];b={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in a.items()};b['Y']=a['U'].copy();return a,b

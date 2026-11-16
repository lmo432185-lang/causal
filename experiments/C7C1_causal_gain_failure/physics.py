import numpy as np
from scipy.signal import lfilter
from config import *
def forcing(command,tau,q):
 a=-np.log(q)/DT;k=(1-q)/a;L=(DT-k)/a;r=np.exp(-DT/tau);d=a-1/tau;j=(r-q)/d;j1=((d*DT-1)*r+q)/(d*d);u=lfilter([1-r],[1,-r],command,axis=1);old=np.c_[np.zeros(len(command)),u[:,:-1]];h=command*k+(old-command)*j;moment=command*L+(old-command)*j1
 return h,moment,old,u

def design(n,seed):
 rng=np.random.default_rng(np.random.SeedSequence([seed,11]));return rng.integers(0,160,n)*DT,rng.integers(40,73,n)*CLOCK,rng.uniform(4,12,n),rng.uniform(0,2*np.pi,n)

def command(n,policy,seed,steps=T):
 phase,*_=design(n,seed);age=(np.arange(steps)[None,:]*DT+phase[:,None])%PERIOD
 if policy=='none':return np.zeros((n,steps))
 if policy=='pulse':return np.sqrt(8)*(age<.5-1e-9)
 return np.sqrt(2)*((age<4-1e-9)&((np.round(age/DT).astype(int)%4)<2))

def simulate(n,sigma,tau,policy,seed,mode='null',zero_source=False):
 I=command(n,policy,seed);phase,tf,period,angle=design(n,seed);times=np.arange(T)*DT;osc=1+.05*np.sin(2*np.pi*times[None,:]/period[:,None]+angle[:,None]);progress=np.zeros((n,T));slope=np.zeros_like(progress);rho=0.
 if mode in ['abrupt','source_only']:progress=(times[None,:]>=tf[:,None]-1e-9).astype(float)
 if mode in ['gradual','partial']:
  progress=np.clip((times[None,:]-tf[:,None])/4,0,1);slope=((times[None,:]>=tf[:,None]-1e-9)&(times[None,:]<tf[:,None]+4-1e-9))/4
 if mode in ['abrupt','gradual']:rho=1.
 if mode=='partial':rho=.5
 beta=osc*(1-rho*progress);betaslope=-rho*osc*slope;base=np.zeros_like(I) if zero_source else np.ones_like(I)
 if mode=='source_only':base-=.5*progress
 S=base+I;r=np.exp(-DT/tau);u0=0. if zero_source else 1.;u=np.empty((n,T));old=np.full(n,u0);Uold=np.empty_like(u)
 for t in range(T):Uold[:,t]=old;old=S[:,t]+(old-S[:,t])*r;u[:,t]=old
 a=.15;q=np.exp(-a*DT);k=(1-q)/a;L=(DT-k)/a;d=a-1/tau;j=(r-q)/d;j1=((d*DT-1)*r+q)/(d*d);force=S*(beta*k+betaslope*L)+(Uold-S)*(beta*j+betaslope*j1)
 rng=np.random.default_rng(np.random.SeedSequence([seed,31]));noise=rng.normal(size=(n,T));x=np.empty((n,T+1));x[:,0]=rng.normal(10 if zero_source else (1.5-1)/.15,.3,n);sd=sigma*np.sqrt((1-q*q)/.3)
 for t in range(T):x[:,t+1]=q*x[:,t]+1.5*k-force[:,t]+sd*noise[:,t]
 return dict(x=x,I=I,U=u,Uold=Uold,beta=beta,beta_slope=betaslope,source=S,failure_time=tf,probe_phase=phase,noise=noise)

import numpy as np
from scipy.signal import lfilter
from config import *

def response(command,kappa,base=None,direct=False):
 command=np.atleast_2d(command);n,steps=command.shape;kappa=np.broadcast_to(np.asarray(kappa,float),(n,));a=.15 if base is None else -np.log(base['q'])/DT;q=np.exp(-a*DT);kk=(1-q)/a
 if direct:return command*kk,np.c_[command[:,1:],np.zeros(n)]
 h=np.empty_like(command);u=np.empty_like(command)
 for tau in np.unique(kappa):
  rows=kappa==tau;I=command[rows];r=np.exp(-DT/tau);us=lfilter([1-r],[1,-r],I,axis=1);old=np.c_[np.zeros(len(I)),us[:,:-1]];j=(r-q)/(a-1/tau);h[rows]=I*kk+(old-I)*j;u[rows]=us
 return h,u

def command(n,program,seed):
 if isinstance(program,str):program=PROGRAMS[IDS.index(program)]
 I=np.zeros((n,T));start=int(START/DT);stop=int(STOP/DT);tt=np.arange(stop-start)*DT;kind=program['kind'];param=program['param']
 if kind=='step':wave=np.ones((n,len(tt)))
 elif kind=='pulse':wave=np.broadcast_to((tt<param-1e-10).astype(float),(n,len(tt))).copy()
 elif kind=='square':wave=np.broadcast_to(((np.arange(len(tt))%round(param/DT))<round(param/DT)/2).astype(float),(n,len(tt))).copy()
 elif kind=='chirp':
  phase=2*np.pi*(.25*tt+.5*(param-.25)/(STOP-START)*tt*tt);wave=np.broadcast_to(.5*(1+np.sin(phase)),(n,len(tt))).copy()
 elif kind=='prbs':
  blocks=round((STOP-START)/param);seq=np.zeros((n,blocks));seq[:,0]=1;rng=np.random.default_rng(np.random.SeedSequence([seed,17,round(param*1000)]))
  for i in range(n):seq[i,rng.choice(np.arange(1,blocks),blocks//2-1,replace=False)]=1
  wave=np.repeat(seq,round(param/DT),axis=1)
 else:raise ValueError(kind)
 wave*=np.sqrt(program['energy']/(np.sum(wave*wave,axis=1)*DT))[:,None];I[:,start:stop]=wave;return I

def innovations(n,sigma,seed):
 rng=np.random.default_rng(seed);return rng.normal(size=(n,T)),rng.normal(10,.3,n)

def simulate(I,sigma,kappa,truth,seed):
 n=len(I);noise,x0=innovations(n,sigma,seed)
 if np.ndim(truth):
  ha,ua=response(I,kappa);hb,ub=response(I,np.ones(n),direct=True);mask=np.asarray(truth)==1;h=np.where(mask[:,None],hb,ha);u=np.where(mask[:,None],ub,ua)
 else:h,u=response(I,kappa,direct=truth==1)
 q=np.exp(-.15*DT);c=1.5*(1-q)/.15;sd=sigma*np.sqrt((1-q*q)/.3);x=np.empty((n,T+1));x[:,0]=x0
 for t in range(T):x[:,t+1]=q*x[:,t]+c-h[:,t]+sd*noise[:,t]
 return dict(x=x,U=u,command=I)

def healthy(n,sigma,seed):return simulate(np.zeros((n,T)),sigma,np.ones(n),1,seed)['x']
def model_bank(command,bg):
 hh=[];gg=[]
 for tau in KGRID:
  h,u=response(command,np.full(len(command),tau),bg);hh.append(h);gg.append(u)
 h,u=response(command,np.ones(len(command)),bg,direct=True);hh.append(h);gg.append(u);return np.stack(hh,1),np.stack(gg,1)

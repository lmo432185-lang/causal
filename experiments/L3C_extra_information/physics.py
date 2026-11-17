import json
from functools import lru_cache
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from config import T

def matrices(p):return np.array([[p['a'],0.,0.],[p['alpha'],p['c'],0.],[p['beta1'],p['beta'],.5]]),np.diag([p['qy'],p['qz'],p['qx']])
def waveform():
 u=np.zeros(T);t=np.arange(50,450);u[t]=.5*np.where(((t-50)//16)%2==0,1.,-1.);return u

def target(info):return 0 if info=='C2_Y' else 1 if info=='C2_Z' else 2 if info=='C2_X_negative' else None

def observation(info):
 return np.array([[0.,0.,1.],[0.,1.,0.]]) if info=='C1_XZ' else np.array([[0.,0.,1.]])

def simulate(p,seed,sigma,info,n):
 A,Q=matrices(p);P=solve_discrete_lyapunov(A,Q);rng=np.random.default_rng(seed);s=rng.normal(size=(n,3))@np.linalg.cholesky(P).T;innov=rng.normal(size=(n,T-1,3))*np.sqrt(np.diag(Q));v=np.random.default_rng(seed+7000000).normal(size=(n,T,2))*sigma;states=np.empty((n,T,3));states[:,0]=s;u=waveform();dest=target(info)
 for t in range(T-1):
  s=s@A.T+innov[:,t]
  if dest is not None:s[:,dest]+=u[t]
  states[:,t+1]=s
 obs=np.stack([states[:,:,2],states[:,:,1]],2)+v
 return obs[:,:,:2] if info=='C1_XZ' else obs[:,:,:1],states

@lru_cache(maxsize=None)
def schedule(key,sigma,joint):
 p=json.loads(key);A,Q=matrices(p);H=observation('C1_XZ' if joint else 'C0_X');P=solve_discrete_lyapunov(A,Q);V=np.eye(len(H))*sigma*sigma;K=[];inv=[];logdet=[]
 for _ in range(T):
  S=H@P@H.T+V;Si=np.linalg.inv(S);k=P@H.T@Si;K.append(k);inv.append(Si);logdet.append(np.linalg.slogdet(S)[1]);P=P-k@H@P;P=A@((P+P.T)/2)@A.T+Q
 return A,H,np.array(K),np.array(inv),np.array(logdet)

def log_likelihood(obs,p,sigma,info,prefixes):
 A,H,K,inv,ld=schedule(json.dumps(p,sort_keys=True),sigma,info=='C1_XZ');m=np.zeros((len(obs),3));ll=np.zeros(len(obs));ans={};u=waveform();dest=target(info);const=len(H)*np.log(2*np.pi)
 for t in range(T):
  err=obs[:,t]-m@H.T;ll-=.5*(np.einsum('ni,ij,nj->n',err,inv[t],err)+ld[t]+const);m+=err@K[t].T
  if t+1 in prefixes:ans[t+1]=ll.copy()
  m=m@A.T
  if dest is not None:m[:,dest]+=u[t]
 return ans

def kernel(p,info,sigma,maxlag=101):
 A,Q=matrices(p);P=solve_discrete_lyapunov(A,Q);H=observation(info);out=np.array([H@np.linalg.matrix_power(A,k)@P@H.T for k in range(maxlag)]);out[0]+=np.eye(len(H))*sigma*sigma;return out

def controlled_mean(p,info):
 A,Q=matrices(p);H=observation(info);m=np.zeros(3);out=[];u=waveform();dest=target(info)
 for t in range(T):
  out.append(H@m);m=A@m
  if dest is not None:m[dest]+=u[t]
 return np.array(out)

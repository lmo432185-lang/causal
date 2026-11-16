import numpy as np
from scipy.linalg import expm,solve_continuous_lyapunov
from scipy.signal import lfilter
from config import *

def system(kind,a,c=1.,alpha=1.,beta=1.,beta1=0.,sigma=.35,qz=0.,qy=1.,b=B):
 if kind in ['none','direct']:
  A=np.array([[-a,0.],[beta if kind=='direct' else 0.,-b]]);D=np.diag([qy,sigma*sigma])
 else:A=np.array([[-a,0.,0.],[alpha,-c,0.],[beta1 if kind=='mixed' else 0.,beta,-b]]);D=np.diag([qy,qz,sigma*sigma])
 dim=len(A);E=expm(np.block([[A,D],[np.zeros_like(A),-A.T]])*DT);F=E[:dim,:dim];Q=E[:dim,dim:]@F.T;Q=(Q+Q.T)/2;P=solve_continuous_lyapunov(A,-D);return A,F,Q,(P+P.T)/2

def simulate(params,n,seed,rho=0.,stationary=False,steps=STEPS):
 A,F,Q,P=system(**params);dim=len(A);rng=np.random.default_rng(seed);S=P.copy() if stationary else np.diag(np.diag(P))
 if not stationary:S[0,-1]=S[-1,0]=rho*np.sqrt(P[0,0]*P[-1,-1])
 z0=rng.normal(size=(n,dim))@np.linalg.cholesky(S+np.eye(dim)*1e-13).T;eps=rng.normal(size=(n,steps,dim))@np.linalg.cholesky(Q+np.eye(dim)*1e-15).T;ends=[];previous=[]
 for j in range(dim):
  inp=eps[:,:,j].copy()
  for k in range(j):inp+=F[j,k]*previous[k]
  val=lfilter([1.],[1.,-F[j,j]],inp,axis=1,zi=(F[j,j]*z0[:,j])[:,None])[0];ends.append(val);previous.append(np.c_[z0[:,j],val[:,:-1]])
 full=np.stack([np.c_[z0[:,j],ends[j]] for j in range(dim)],axis=2);X=full[:,:,-1];Y=full[:,:,0];Z=full[:,:,1] if dim==3 else np.zeros_like(Y);W=params.get('beta',1.)*Z if dim==3 else np.zeros_like(Y)
 return dict(x=X,Y=Y,Z=Z,W=W,states=full,P=P)

def random_pool(n,sigma,seed):
 rng=np.random.default_rng(seed);out=[]
 for i in range(n):
  kind=CLASS[i%4];a=float(rng.choice(AS));c=float(rng.choice(CS));beta=float(np.exp(rng.uniform(np.log(.15),np.log(2.5))));beta1=float(np.exp(rng.uniform(np.log(.3),np.log(1.8))));params=dict(kind=kind,a=a,c=c,alpha=1.,beta=beta,beta1=beta1,sigma=sigma)
  d=simulate(params,1,int(rng.integers(2**31)),stationary=True);out.append({k:d[k][0,NBURN:] for k in ['x','Y','Z','W']})
 return {key:np.stack([d[key] for d in out]) for key in out[0]}

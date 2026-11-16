import numpy as np
from scipy.linalg import solve_discrete_lyapunov

def plant(kind='indirect',a=.8,c=.7,alpha=.5,beta=.8,beta1=0.,qy=.25,qz=.25,qx=.25,obs=0.):
 if kind=='direct':A=np.array([[a,0],[beta,.5]]);Q=np.diag([qy,qx])
 else:A=np.array([[a,0,0],[alpha,c,0],[beta1,beta,.5]]);Q=np.diag([qy,qz,qx])
 return A,Q,float(obs**2)
def generate(params,seed,n=1000,T=500,stationary=False):
 A,Q,R=plant(**params);rng=np.random.default_rng(seed);p=solve_discrete_lyapunov(A,Q);s=rng.normal(size=(n,len(A)))@np.linalg.cholesky(p+np.eye(len(A))*1e-14).T if stationary else np.zeros((n,len(A)));out=[];L=np.diag(np.sqrt(np.diag(Q)))
 for t in range(T if stationary else T+200):
  s=s@A.T+rng.normal(size=s.shape)@L
  if stationary or t>=200:out.append(s.copy())
 states=np.stack(out,1).astype('float32');physical=states[:,:,-1];observed=(physical+np.random.default_rng(seed+1000000).normal(0,np.sqrt(R),physical.shape)).astype('float32');return observed,states

def covariance_schedule(A,Q,R,T):
 p=np.zeros_like(A)
 for _ in range(201):p=A@p@A.T+Q
 prior=[];pf=[];kk=[]
 for _ in range(T):
  prior.append(p.copy());k=p[:,-1]/(p[-1,-1]+R);kk.append(k);f=p-np.outer(k,p[-1]);pf.append((f+f.T)/2);p=A@pf[-1]@A.T+Q
 prior=np.array(prior);pf=np.array(pf);ps=pf.copy();J=[]
 for t in range(T-1):J.append(np.linalg.solve(prior[t+1],A@pf[t]).T)
 for t in range(T-2,-1,-1):ps[t]=pf[t]+J[t]@(ps[t+1]-prior[t+1])@J[t].T;ps[t]=(ps[t]+ps[t].T)/2
 return prior,pf,ps,np.array(kk),np.array(J)
def kalman(x,A,Q,R):
 n,T=x.shape;prior,pf,ps,K,J=covariance_schedule(A,Q,R,T);m=np.zeros((n,len(A)));mf=np.zeros((n,T,len(A)))
 for t in range(T):
  if t:m=m@A.T
  m=m+(x[:,t]-m[:,-1])[:,None]*K[t];mf[:,t]=m
 ms=mf.copy()
 for t in range(T-2,-1,-1):ms[:,t]=mf[:,t]+(ms[:,t+1]-mf[:,t]@A.T)@J[t].T
 return mf,ms,pf,ps

def windows(x,h=1):return np.lib.stride_tricks.sliding_window_view(x,20,axis=1)[:,:-(h or 0)].copy().reshape(-1,20)
def apply(x,m):return np.c_[np.ones(len(x)),(x-np.asarray(m['mean'],dtype=x.dtype))/np.asarray(m['sd'],dtype=x.dtype)]@np.asarray(m['coef'])
def ridge(x,y,z):
 mu=x.mean(0);sd=np.maximum(x.std(0),1e-7);xx=np.c_[np.ones(len(x)),(x-mu)/sd];coef=np.linalg.solve(xx.T@xx+np.diag([0]+[.001]*x.shape[1]),xx.T@y);m=dict(mean=mu.tolist(),sd=sd.tolist(),coef=coef.tolist());return apply(z,m),m

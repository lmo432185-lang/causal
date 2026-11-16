"""Known-model Gaussian conditional means. No hidden truth enters this module."""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
A=np.array([[.8,0,0],[.5,.7,0],[0,.8,.5]],dtype=float);Q=np.eye(3)*.25;H=np.array([0.,0.,1.]);R=0.
def initial_covariance():
 # L1 starts at zero and first retained state is after 201 transitions.
 p=np.zeros((3,3))
 for _ in range(201):p=A@p@A.T+Q
 return p

def covariance_schedule(T):
 prior=np.empty((T,3,3));pf=np.empty_like(prior);K=np.empty((T,3));p=initial_covariance()
 for t in range(T):
  prior[t]=p;k=p[:,2]/p[2,2];K[t]=k;f=p-np.outer(k,p[2]);pf[t]=(f+f.T)/2;p=A@pf[t]@A.T+Q
 ps=pf.copy();J=np.empty((max(T-1,0),3,3))
 for t in range(T-2,-1,-1):
  j=np.linalg.solve(prior[t+1],A@pf[t]).T;J[t]=j;ps[t]=pf[t]+j@(ps[t+1]-prior[t+1])@j.T;ps[t]=(ps[t]+ps[t].T)/2
 return prior,pf,ps,K,J

def estimate(x,smooth=True):
 x=np.asarray(x,dtype=float);n,T=x.shape;prior,pf,ps,K,J=covariance_schedule(T);m=np.zeros((n,3));mf=np.empty((n,T,3))
 for t in range(T):
  if t:m=m@A.T
  m=m+(x[:,t]-m[:,2])[:,None]*K[t];mf[:,t]=m
 ms=mf.copy()
 if smooth:
  for t in range(T-2,-1,-1):ms[:,t]=mf[:,t]+(ms[:,t+1]-mf[:,t]@A.T)@J[t].T
 return mf,ms,pf,ps

def covariance_joint(T):
 # Independent block-Gaussian construction for verification.
 P=initial_covariance();c=np.empty((T,3,T,3));powers=[np.linalg.matrix_power(A,k) for k in range(T)]
 for s in range(T):
  if s:P=A@P@A.T+Q
  for t in range(s,T):c[t,:,s,:]=powers[t-s]@P;c[s,:,t,:]=c[t,:,s,:].T
 return c

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import ss2tf,lfilter
from config import *
from physics import system

def fitted_system(pred,theta):
 a,c,beta1,gamma,qx=theta
 if pred in [0,1]:return system('none' if pred==0 else 'direct',a,beta=beta1,sigma=np.sqrt(qx))
 return system('indirect' if pred==2 else 'mixed',a,c,alpha=gamma,beta=1.,beta1=beta1,sigma=np.sqrt(qx))

def recurrence_filter(A,B,C,x):
 # z_t=A z_(t-1)+B x_t; output C z_t. Standard SS uses a one-step state lag.
 D=C@B;Cstd=C@A;outputs=np.zeros((x.shape[0],C.shape[0]))
 for j in range(B.shape[1]):
  nums,den=ss2tf(A,B,Cstd,D,input=j)
  for i,num in enumerate(nums):outputs[:,i]+=lfilter(np.real_if_close(num),np.real_if_close(den),x[:,j])
 return np.real_if_close(outputs)

def filter_smooth(x,pred,theta,smooth=True):
 A,F,Q,P=fitted_system(pred,theta);d=len(A);H=np.zeros((1,d));H[0,-1]=1.;Pp=solve_discrete_are(F.T,H.T,Q,np.zeros((1,1)));K=Pp@H.T/(H@Pp@H.T).item();Af=(np.eye(d)-K@H)@F;mf=recurrence_filter(Af,K,np.eye(d),x[:,None])
 if not smooth:return mf,None,(F,Pp,K)
 Pf=(np.eye(d)-K@H)@Pp;J=Pf@F.T@np.linalg.inv(Pp);v=(np.eye(d)-J@F)@mf.T;v=v.T[::-1].copy();v[0]=mf[-1];ms=recurrence_filter(J,np.eye(d),np.eye(d),v)[::-1]
 return mf,ms,(F,Pp,K)

def loop_reference(x,pred,theta):
 A,F,Q,P=fitted_system(pred,theta);d=len(A);H=np.zeros((1,d));H[0,-1]=1;Pp=solve_discrete_are(F.T,H.T,Q,np.zeros((1,1)));K=Pp@H.T/(H@Pp@H.T).item();mf=np.zeros((len(x),d));m=np.zeros(d)
 for t in range(len(x)):m=F@m;m=m+K[:,0]*(x[t]-m[-1]);mf[t]=m
 Pf=(np.eye(d)-K@H)@Pp;J=Pf@F.T@np.linalg.inv(Pp);ms=mf.copy()
 for t in range(len(x)-2,-1,-1):ms[t]=mf[t]+J@(ms[t+1]-F@mf[t])
 return mf,ms

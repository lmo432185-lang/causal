import json
import numpy as np,pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from config import O,SEEDS
from core import plant,generate
PAIRS=[('direct_vs_noisy_indirect',dict(kind='direct',a=.8,beta=.8,qy=.5,qx=.25),dict(kind='indirect',a=.8,c=.4,alpha=np.sqrt(.68),beta=.8,qy=.25,qz=.25,qx=.25)),('mixed_positive_cancellation_vs_direct',dict(kind='mixed',a=.3,c=.8,alpha=.5,beta=.5,beta1=.5,qy=.25,qz=0.,qx=.25),dict(kind='direct',a=.8,beta=.5,qy=.25,qx=.25)),('source_middle_clock_swap',dict(kind='indirect',a=.8,c=.4,alpha=.5,beta=.8,qy=.25,qz=0.,qx=.25),dict(kind='indirect',a=.4,c=.8,alpha=.5,beta=.8,qy=.25,qz=0.,qx=.25))]
def covariance(params):
 A,Q,R=plant(**params);P=solve_discrete_lyapunov(A,Q);return np.array([(np.linalg.matrix_power(A,k)@P)[-1,-1]+(R if k==0 else 0) for k in range(101)])
def spectrum(params,w):
 A,Q,R=plant(**params);H=np.zeros(len(A));H[-1]=1;out=[]
 for v in w:
  inv=np.linalg.inv(np.eye(len(A))-A*np.exp(-1j*v));out.append(float(np.real(H@inv@Q@inv.conj().T@H)+R))
 return np.array(out)
def run():
 proofs=[];rows=[]
 for pi,(name,p0,p1) in enumerate(PAIRS):
  cov0=covariance(p0);cov1=covariance(p1);rel=float(np.max(abs(cov0-cov1))/np.max(abs(cov0)));s0=spectrum(p0,np.linspace(0,np.pi,501));s1=spectrum(p1,np.linspace(0,np.pi,501));serr=float(np.max(abs(s0-s1))/np.max(abs(s0)));assert rel<1e-10 and serr<1e-10;proofs.append(dict(pair=name,H0=p0,H1=p1,covariance_max_relative_error=rel,spectrum_max_relative_error=serr,matched_stationary_Gaussian_output_laws=True,equal_prior_exact_posterior=[.5,.5],decision='refuse unique structure; known equivalent alternatives, not a learned ambiguity detector'))
  for seed in SEEDS:
   for label,p in enumerate([p0,p1]):
    x,_=generate(p,seed+50000+1000*pi+100*label,n=200,stationary=True)
    for i in range(200):rows.append(dict(pair=name,seed=seed,trajectory=i,label=label,X_variance=float(x[i].var()),lag1=float(np.cov(x[i,:-1],x[i,1:])[0,1]),posterior_H0=.5,posterior_H1=.5,refused=True))
 (O/'equivalence_proofs.json').write_text(json.dumps(proofs,indent=2));pd.DataFrame(rows).to_csv(O/'equivalence_metrics.csv.gz',index=False)

import json
import numpy as np,pandas as pd
from scipy.linalg import expm
from config import *
from physics import system,simulate
from spectral import infer,library,diffusion_from_x

def pairs(sigma):
 common=dict(sigma=sigma)
 return [
 ('direct_vs_noisy_indirect',dict(kind='direct',a=.25,beta=.3,**common),dict(kind='indirect',a=.25,c=np.sqrt(.25**2+1/.09),alpha=1.,beta=1.,qz=.09,**common)),
 ('mixed_vs_noisy_indirect',dict(kind='mixed',a=.25,c=2.,alpha=1.,beta=1.,beta1=.5,**common),dict(kind='indirect',a=.25,c=2.,alpha=np.sqrt(4-.25*.25**2),beta=1.,qz=.25,**common)),
 ('source_transfer_pole_swap',dict(kind='indirect',a=.25,c=2.,alpha=1.,beta=1.,**common),dict(kind='indirect',a=2.,c=.25,alpha=1.,beta=1.,**common)),
 ('source_scale_beta_ambiguity',dict(kind='direct',a=1.,beta=1.,qy=1.,**common),dict(kind='direct',a=1.,beta=.5,qy=4.,**common)),
 ('mixed_clean_pole_cancellation',dict(kind='mixed',a=2.,c=.5,alpha=1.,beta=.75,beta1=.5,**common),dict(kind='direct',a=.5,beta=.5,**common))]

def main():
 proofs=[];rows=[]
 for ni,sigma in enumerate(SIGMAS):
  for pi,(name,p0,p1) in enumerate(pairs(sigma)):
   A0,F0,Q0,P0=system(**p0);A1,F1,Q1,P1=system(**p1);lags=np.linspace(0,30,601);c0=np.array([(expm(A0*t)@P0)[-1,-1] for t in lags]);c1=np.array([(expm(A1*t)@P1)[-1,-1] for t in lags]);err=float(np.max(abs(c0-c1))/max(abs(c0[0]),1e-12));assert err<1e-8
   proofs.append(dict(pair=name,noise=sigma,relative_covariance_error=err,H0=p0,H1=p1,matched_output_laws=True))
   for co in COHORTS:
    for truth,params in enumerate([p0,p1]):
     seed=int(np.random.SeedSequence([SEEDS['equivalence'],ni,pi,co,truth]).generate_state(1)[0]);d=simulate(params,N,seed,stationary=True)
     # Equal Gaussian covariance kernels imply equal likelihood for every X, not just close samples.
     for i in range(N):rows.append(dict(pair=name,noise=sigma,cohort=co,trajectory=i,truth=truth,posterior_H0=.5,posterior_H1=.5,refused=1,forced_prediction=0,correct=int(truth==0)))
 (OUT/'equivalence_proofs.json').write_text(json.dumps(proofs,indent=2));pd.DataFrame(rows).to_csv(OUT/'equivalence_metrics.csv.gz',index=False);print('D exact covariance maximum error',max(p['relative_covariance_error'] for p in proofs))
if __name__=='__main__':main()

import json
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from model import *
from settings import ROOT

def main():
 th=np.array([.73,.51,.29,.43,.32,.27,.08,1.2,.8,.9]);errs={}
 for kind in ['E0','E1','E2','E3','E4Y','E4Z','E4YZ']:
  C,D=covariance(th,2,kind);Z=approx_derivative(lambda t:covariance(t,2,kind,False).ravel(),th,method='3-point');err=float(np.max(abs(Z-D.reshape(10,-1).T)));assert err<2e-7,(kind,err);errs[kind]=err
 # Observation matrix layouts must match permitted data designs for every method.
 for kind in ['E0','E1','E2','E3','E4Y','E4Z','E4YZ']:
  z=design(np.zeros((2,500)),kind,dict(Y=np.zeros((2,12 if kind=='E4YZ' else 24)),Z=np.zeros((2,12 if kind=='E4YZ' else 24))),np.zeros((2,500)));assert z.shape[1]==covariance(th,2,kind,False).shape[0],kind
 # Same candidate posterior prefixes must be exactly causal, including observed known input.
 x=np.random.default_rng(90701).normal(size=(3,80));I=np.random.default_rng(90702).normal(size=x.shape);a=infer(x,th,3,'E3',I);b=infer(x[:,:50],th,3,'E3',I[:,:50]);prefix=float(np.max(abs(a['mean'][:,:50]-b['mean'])));assert prefix<1e-12
 # Oracle model covariance of different pure delays is identical for X alone.
 covs=[covariance(th,d,'E0',False) for d in [1,2,3]];differ=float(max(np.max(abs(c-covs[0])) for c in covs));assert differ<1e-10
 result=dict(status='PASS',unrelated_toy_parameters_only=True,covariance_analytic_derivative_max_errors=errs,causal_prefix_error=prefix,X_only_delay_covariance_difference=differ,native_backend=available);(ROOT/'DEVELOPMENT_VALIDATION.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()

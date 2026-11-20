"""Numerical checks on unrelated toy parameters, without experiment data or latent answers."""
import json
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.linalg import toeplitz
from model import *
from settings import ROOT

def main():
 theta=np.array([.63,.41,.29,.37,.46,.31,.09]);lags=80;S=spectrum(theta);gam=np.fft.ifft(S).real[:lags];covs=[]
 for d in [1,2,3]:
  A,Q,R,P=plant(theta,d);g=np.array([(np.linalg.matrix_power(A,k)@P)[2,2] for k in range(lags)]);g[0]+=R;covs.append(g)
 err=float(max(np.max(abs(g-gam)) for g in covs));assert err<1e-11
 St=toeplitz(gam[:BLOCK]);v,der=objective(theta,St);num=approx_derivative(lambda z:np.array([objective(z,St)[0]]),theta,method='3-point').ravel();grad=float(np.max(abs(der-num)));assert grad<1e-7
 rng=np.random.default_rng(90117);x=rng.normal(size=(5,90));par=plant(theta,3);a=infer(x,par,True);b=infer(x[:,:60],par);prefix=float(np.max(abs(a['mean'][:,:60]-b['mean'])));assert prefix<1e-12
 A,Q,R,P=par;H,K,PP,PF,S,J,PS=filter_cov(A,Q,R,P,90,True);mf=np.zeros((5,90,len(A)));m=np.zeros((5,len(A)))
 for t in range(90):
  if t:m=m@A.T
  m+=(x[:,t]-m[:,2])[:,None]*K[t,:,0];mf[:,t]=m
 ms=mf.copy()
 for t in range(88,-1,-1):ms[:,t]=mf[:,t]+(ms[:,t+1]-mf[:,t]@A.T)@J[t].T
 ferr=float(np.max(abs(mf-a['mean'])));serr=float(np.max(abs(ms-a['smoothed'])));assert ferr<1e-11 and serr<1e-11
 # Delay-independent X likelihood must not be used to select true hidden timing.
 vals=[infer(x,plant(theta,d))['trajectory_ll'] for d in [1,2,3]];llerr=float(max(np.max(abs(v-vals[0])) for v in vals));assert llerr<1e-9
 result=dict(status='PASS',FFT_vs_Lyapunov_covariance_max_error=err,analytic_gradient_error=grad,causal_prefix_error=prefix,native_filter_error=ferr,native_smoother_error=serr,equivalent_delays_LL_error=llerr,native_backend=available,scope='unrelated deterministic toy parameters only');(ROOT/'DEVELOPMENT_VALIDATION.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()

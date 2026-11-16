import json
import numpy as np
from scipy.linalg import expm
from config import *
from physics import *
from spectral import *
from states import *

def main():
 params=dict(kind='mixed',a=.25,c=2.,alpha=.7,beta=1.5,beta1=.5,sigma=.35);d=simulate(params,3,990001,stationary=True);A,F,Q,P=system(**params);theta=np.array([.25,2.,.5,1.05,.35**2]);mf,ms,_=filter_smooth(d['x'][0],3,theta);m0,s0=loop_reference(d['x'][0],3,theta);assert np.max(abs(mf-m0))<1e-7 and np.max(abs(ms-s0))<1e-7;assert np.max(abs(mf[:,-1]-d['x'][0]))<1e-7
 cut=500;mc,_,_=filter_smooth(d['x'][0,:cut],3,theta,False);assert np.max(abs(mc-mf[:cut]))<1e-8
 # Sampled spectrum and covariance agree with independent covariance summation.
 n=NOBS;k=bins(n);w=np.hanning(n);from scipy.signal import fftconvolve
 aw=fftconvolve(w,w[::-1],mode='full')[n-1:]/np.sum(w*w);cov=np.array([(expm(A*(j*DT))@P)[-1,-1] for j in range(n)]);exact=(2*np.fft.rfft(cov*aw).real-cov[0])[k];sb=ou_basis(B,n);sa=ou_basis(.25,n);sc=ou_basis(2.,n);ab=(sa-sb)/(B*B-.25**2);acb=sa/((4-.25**2)*(B*B-.25**2))+sc/((.25**2-4)*(B*B-4))+sb/((.25**2-B*B)*(4-B*B));formula=.35**2*sb+.5**2*ab+(2*.5*1.05*2+1.05**2)*acb;err=float(np.max(abs(formula-exact))/np.max(exact));assert err<1e-8
 out=dict(steady_filter_matches_Kalman_loop=float(np.max(abs(mf-m0))),RTS_matches_loop=float(np.max(abs(ms-s0))),filter_no_future_error=float(np.max(abs(mc-mf[:cut]))),observed_X_reconstructed_exactly=True,tapered_spectrum_vs_covariance_relative_error=err)
 (OUT/'validation.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()

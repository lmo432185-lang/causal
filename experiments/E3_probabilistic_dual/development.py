import json,time,numpy as np
from settings import ROOT
from ssm import *

def main():
 rng=np.random.default_rng(25011);F=np.array([[.7,.1],[0,.4]]);Q=np.diag([.2,.1]);C=np.array([1.,.5]);m=complete(dict(F=F,Q=Q,C=C,a=.2,R=.3));x=rng.normal(size=(1,8));p=native.run(x,m,smooth=True);T=8;k=2;S=np.zeros((T*k,T*k))
 for i in range(T):
  for j in range(T):S[i*k:(i+1)*k,j*k:(j+1)*k]=np.linalg.matrix_power(F,i-j)@m['P'] if i>=j else m['P']@np.linalg.matrix_power(F,j-i).T
 B=np.kron(np.eye(T),C[None]);r=x[0]-np.r_[0,.2*x[0,:-1]];mu=S@B.T@np.linalg.solve(B@S@B.T+.3*np.eye(T),r);V=S-S@B.T@np.linalg.solve(B@S@B.T+.3*np.eye(T),B@S);err=float(np.max(abs(mu.reshape(T,k)-p['smean'][0])));cerr=max(np.max(abs(V[i*k:(i+1)*k,i*k:(i+1)*k]-p['scov'][i])) for i in range(T));assert err<1e-9 and cerr<1e-9
 draws=native.sample(m,p,19,draws=10000).reshape(10000,T*k);sample_error=float(np.max(abs(np.cov(draws.T)-V)));assert sample_error<.02
 xx=rng.normal(size=(20,160));v=rng.normal(size=(10,160));tic=time.time();models=[];old=None
 for K in range(4):
  init=initial(xx,K) if K<2 else recursive(xx,old,K,100+K)[0];old=em(xx,init);models.append(old);scores=selection_score(v,old);assert np.isfinite(scores).all()
 base=component(v,old);changed=v.copy();changed[:,81:]+=33.;after=component(changed,old);idx=[i for i,t in enumerate(ORIGINS) if t<=80];prefix=float(np.max(abs(base['mean'][:,idx]-after['mean'][:,idx])));assert prefix<1e-9
 out=dict(passed=True,unrelated_development_noise=True,analytic_smoother_mean_error=err,analytic_smoother_cov_error=float(cerr),posterior_path_sample_cov_error=sample_error,forecast_prefix_error=prefix,seconds=time.time()-tic)
 (ROOT/'DEVELOPMENT.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()

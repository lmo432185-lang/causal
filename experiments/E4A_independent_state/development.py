"""Independent development noise; validate issued densities, covariance and aliases."""
import json,time,numpy as np
from scipy.linalg import solve_discrete_lyapunov
from settings import *
from model import fit,finish
from predict import component,logpdf,targets,infer,scores,history_fit,history_scores
from learn import fit_all
from producer import arma_alias

def main():
 rng=np.random.default_rng(8839);x=rng.normal(size=(6,500));v=rng.normal(size=(3,500));tic=time.time();m=fit_all(x,x[::-1],v,9411);cache={};err=0.;density=0
 for group in ['M1','M2','M3','M4']:
  r=infer(v,m,group,cache);s=scores(v,r);assert all(np.isfinite(a).all() for a in s.values());assert np.max(abs(r['weights'].sum(2)-1))<1e-9
 changed=v.copy();changed[:,301:]+=30.;r0=infer(v,m,'M4',{});r1=infer(changed,m,'M4',{});ix=[i for i,t in enumerate(origins(500)) if t<=300];err=float(np.max(abs(r0['mean'][:,ix]-r1['mean'][:,ix])));assert err<1e-9
 for z in m['direct']['models'].values():
  c=component(v,z);ll=logpdf(c['mean'][:,:,:1],c['cov'][:,:1,:1],targets(v)[:,:,:1]);density=max(density,float(np.max(abs(ll-c['post']['ll'][:,np.array(origins(500))+1]))))
 assert density<1e-9;history_scores(v,history_fit(x,v))
 z=finish(dict(kind='factor2',F=np.diag([.82,-.43]),Q=np.diag([.2,.3]),C=np.ones(2),a=0.,R=.4));z['scale']=1.;a=arma_alias(z);c0=component(v,z);c1=component(v,a);equiv=max(np.max(abs(c0['mean']-c1['mean'])),np.max(abs(c0['cov']-c1['cov'])),np.max(abs(c0['post']['ll']-c1['post']['ll'])));assert equiv<1e-9
 (ROOT/'DEVELOPMENT.json').write_text(json.dumps(dict(passed=True,unrelated_noise=True,seconds=time.time()-tic,max_prefix_error=err,max_issued_density_error=density,exact_ARMA_alias_error=float(equiv)),indent=2));print('DEVELOPMENT PASSED',time.time()-tic)
if __name__=='__main__':main()

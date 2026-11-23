"""Independent synthetic development, never formal experiment data."""
import numpy as np,json,time
from scipy.linalg import solve_discrete_lyapunov
from settings import *
import native
from extension import finish,proposals,fit_candidate
from predict import component,targets,logpdf

def reference(x,m):
 P=m['P'].copy();means=[];cov=[];mu=np.zeros(len(P));F,Q,C=m['F'],m['Q'],m['C']
 for t,y in enumerate(x):
  if t:mu=F@mu;P=F@P@F.T+Q
  v=P@C;S=C@v+m['R'];mu+=v/S*(y-(m['a']*x[t-1] if t else 0)-C@mu);P-=np.outer(v,v)/S;means.append(mu.copy());cov.append(P.copy())
 return np.array(means),np.array(cov)

def main():
 rng=np.random.default_rng(7842);x=rng.normal(size=(5,500));checks=[]
 for d in [1,2,3,4]:
  A=rng.normal(size=(d,d));A*=.7/max(abs(np.linalg.eigvals(A)));B=rng.normal(size=(d,d));m=finish(dict(F=A,Q=B@B.T+.05*np.eye(d),C=rng.normal(size=d),R=.6,a=.2,kind='extension',scale=1.));p=native.run(x,m);ref,cov=reference(x[0],m);err=max(np.max(abs(ref-p['mean'][0])),np.max(abs(cov-p['cov'])));assert err<1e-8
  c=component(x,m);issued=logpdf(c['mean'][:,:,:1],c['cov'][:,:1,:1],targets(x)[:,:,:1]);de=np.max(abs(issued-c['post']['ll'][:,np.array(origins(500))+1]));assert de<1e-8;checks.append(dict(d=d,filter_error=float(err),density_error=float(de)))
 base=finish(dict(F=np.array([[.65]]),Q=np.array([[.25]]),C=np.ones(1),R=.4,a=0.,kind='factor1',scale=1.));pr=proposals(x,base,333);tic=time.time()
 for route in ['direct','path','residual','joint']:
  for mode in (['dense'] if route=='joint' else ['obs','latent','both','dense'] if route=='direct' else ['latent','both']):
   m,diag=fit_candidate(x,base,mode,route,pr,341)
   if m is not None:
    assert max(abs(np.linalg.eigvals(m['F'])))<1;c=component(x,m);changed=x.copy();changed[:,301:]+=8.;d=component(changed,m);ix=np.array(origins(500))<=300;assert np.max(abs(c['mean'][:,ix]-d['mean'][:,ix]))<1e-9
 (ROOT/'DEVELOPMENT.json').write_text(json.dumps(dict(time=time.time(),all_passed=True,unrelated_data=True,checks=checks,seconds=time.time()-tic),indent=2));print('Development passed',checks)
if __name__=='__main__':main()

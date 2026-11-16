import json,hashlib
import numpy as np
from scipy.integrate import solve_ivp
from config import *
from physics import *
from detectors import scores,nominal
from train import background,fit_plant,fit_scales

def main():
 seed=990010;n=3;d=simulate(n,.35,.5,'square',seed,'gradual');base=simulate(n,.35,.5,'square',seed,'null');assert np.array_equal(d['I'],base['I']) and np.array_equal(d['U'],base['U']);pre=np.arange(T)[None,:]*DT<d['failure_time'][:,None];assert np.all(d['beta'][pre]==base['beta'][pre])
 for p in ['pulse','square']:assert np.allclose(np.sum(command(50,p,seed)**2,axis=1)*DT,16)
 # Independent ODE integration within each source/normalbeta hold interval.
 q=np.exp(-.15*DT);k=(1-q)/.15;sd=.35*np.sqrt((1-q*q)/.3);err=0.
 for i in range(n):
  state=np.array([d['x'][i,0],1.])
  for t in range(T):
   S=d['source'][i,t];b=d['beta'][i,t];slope=d['beta_slope'][i,t]
   sol=solve_ivp(lambda ss,st:[1.5-.15*st[0]-(b+slope*ss)*st[1],(S-st[1])/.5],[0,DT],state,rtol=2e-10,atol=1e-12).y[:,-1];sol[0]+=sd*d['noise'][i,t];err=max(err,float(abs(sol[0]-d['x'][i,t+1])),float(abs(sol[1]-d['U'][i,t])));state=sol
 assert err<1e-7
 a=simulate(n,.35,.5,'pulse',seed,'abrupt');after=np.arange(T)[None,:]*DT>=a['failure_time'][:,None];res=a['x'][:,1:]-q*a['x'][:,:-1]-1.5*k-sd*a['noise'];assert np.max(abs(res[after]))<1e-12
 bg=background(1);model=fit_plant(1,1,bg);scales=fit_scales(1,1,'square',model);full,meta=scores(d['x'],d['I'],model,scales);prefix=355;cut,metac=scores(d['x'][:,:prefix+1],d['I'][:,:prefix],model,scales);errors={key:float(np.max(abs(full[key][:,:prefix//BLOCK]-cut[key]))) for key in full};assert max(errors.values())<1e-7
 # No intervention: a scalar gain loss is observationally equivalent to a forcing offset.
 dd=simulate(n,.35,.5,'none',seed,'abrupt');ss,_=scores(dd['x'],dd['I'],model,scales);assert np.max(ss['GainGLR']-ss['GLR'])<1e-6;assert np.all(ss['GainSpecific']==0)
 provenance=json.loads((ROOT/'legacy_provenance.json').read_text());legacy={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in provenance['sha256'].items()};assert all(legacy.values())
 out=dict(exact_beta_U_integration_versus_independent_ODE_error=err,full_failure_innovations_independent_of_U=True,inputs_and_U_unchanged_by_beta_failure=True,pre_failure_paths_identical=True,input_energy16_each_active=True,no_future_max_errors=errors,passive_gain_not_unique_against_offset=True,frozen_legacy_files_unchanged=legacy)
 (OUT/'validation.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()

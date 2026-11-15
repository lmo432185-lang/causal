import json
import numpy as np
from scipy.special import logsumexp
from config import *
from physics import *
from inference import stream,commit

def main():
 energies={};peaks={};kernel=0.;rng=np.random.default_rng(990010);bg=dict(q=np.exp(-.15*DT),c=1.5*(1-np.exp(-.15*DT))/.15,var=.35**2*(1-np.exp(-.3*DT))/.3)
 for program in PROGRAMS:
  I=command(4,program,990011);ee=np.sum(I*I,axis=1)*DT;peak=float(I.max());assert np.allclose(ee,program['energy']);assert peak<=(CAP if program['primary'] else HIGH_CAP)+1e-9;assert np.all(I[:,:200]==0) and np.all(I[:,280:]==0);energies[program['id']]=ee.tolist();peaks[program['id']]=peak
  tau=.05;h,u=response(I,np.full(4,tau));old=np.zeros(4);q=bg['q'];r=np.exp(-DT/tau);k=(1-q)/.15;j=(r-q)/(.15-1/tau);hh=[];uu=[]
  for t in range(T):hh.append(I[:,t]*k+(old-I[:,t])*j);old=I[:,t]+(old-I[:,t])*r;uu.append(old.copy())
  kernel=max(kernel,float(np.max(abs(h-np.stack(hh,1)))),float(np.max(abs(u-np.stack(uu,1)))))
 assert kernel<1e-11
 I=command(4,REFERENCE,990012);d=simulate(I,.35,np.ones(4)*.1,0,990013);p,z=stream(d['x'],I,bg);end=245;pp,zz=stream(d['x'][:,:end+1],I,bg);prefix_error=float(max(np.max(abs(p[:,:end//BLOCK]-pp)),np.max(abs(z[:,:end//BLOCK]-zz))));assert prefix_error<1e-9
 # Prefix-independent joint likelihood calculation.
 H,_=model_bank(I,bg);rr=-(d['x'][:,1:]-bg['q']*d['x'][:,:-1]-bg['c']);ll=-.5*np.sum((rr[:,None,:]-H)**2,axis=2)/bg['var'];ev=np.stack([logsumexp(ll[:,:-1],axis=1)-np.log(len(KGRID)),ll[:,-1]],axis=1);pb=np.exp(ev-logsumexp(ev,1)[:,None]);err=float(np.max(abs(pb-p[:,-1])));assert err<1e-9
 blank=np.zeros((4,T));x=healthy(4,.35,990014);p,_=stream(x,blank,bg);assert np.allclose(p,.5) and np.all(commit(p)[1]==-1)
 out=dict(energy_and_peak_feasible=True,programs=len(PROGRAMS),primary=len(PRIMARY),peak_by_program=peaks,accelerated_filter_vs_independent_recurrence_error=kernel,no_future_prefix_error=prefix_error,independent_likelihood_error=err,blank_all_refuse=True,high_peak_diagnostic_excluded_from_primary=True)
 (OUT/'validation.json').write_text(json.dumps(out,indent=2));print({k:v for k,v in out.items() if k!='peak_by_program'})
if __name__=='__main__':main()

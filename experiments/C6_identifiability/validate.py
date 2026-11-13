import json
import numpy as np
from scipy.integrate import solve_ivp,quad
from scipy.special import log_ndtr
from mechanisms import response
from generate import equivalent_pair,draw
from inference import evidence,active_evidence
from config import *

def main():
 errors=[]
 for m in LABELS:
  for kap in [.1,.5,2.]:
   for duration in [np.inf,.5]:
    h,u,y=response(m,[kap],[1.],duration,steps=100);q=np.exp(-.15*DT);x=np.zeros(101)
    for t in range(100):x[t+1]=q*x[t]-h[0,t]
    # Independently integrate continuous dynamics split at discontinuities.
    state=np.zeros(3);xs=[0.]
    for t in range(100):
     start=t*DT;stop=start+DT;I=float(start>=1-1e-9 and start<1+duration-1e-9)
     def fun(tm,s):
      xx,uu,ww=s
      if m=='B_direct':return [-.15*xx-I,0,0]
      if m=='A_first':return [-.15*xx-uu,(I-uu)/kap,0]
      if m=='C_two':return [-.15*xx-uu,2*(ww-uu)/kap,2*(I-ww)/kap]
      pre=kap/(1-.75/np.e);tau=pre if tm<1+pre else pre/4
      return [-.15*xx-uu,(I-uu)/tau,0]
     state=solve_ivp(fun,[start,stop],state,rtol=2e-10,atol=1e-12).y[:,-1];xs.append(state[0])
    errors.append(np.max(np.abs(x-xs)))
 assert max(errors)<2e-7,max(errors)
 # Exact representation equivalence: eliminate U by recurrence/convolution.
 hh,uu,yy=response('A_first',[.5],[1.],.5,steps=100);r=np.exp(-DT/.5);ue=np.zeros(100)
 for t in range(1,100):ue[t]=r*ue[t-1]+(1-r)*yy[0,t-1]
 assert np.max(abs(ue-uu[0]))<1e-12
 # Passive causal-equivalence pair has bit-identical X/U, different source.
 a,b=equivalent_pair(3,.35,.5,99999)
 assert np.array_equal(a['x'],b['x']) and np.array_equal(a['U'],b['U']) and not np.array_equal(a['Y'],b['Y'])
 # Half-normal integration, independently numerical quadrature.
 bg=dict(q=np.exp(-.15*DT),c=0.,var=.2);rng=np.random.default_rng(998);x=rng.normal(size=(1,6));h=rng.normal(size=(1,5))*.1;s=2.
 val=evidence(x,{'test':{'h':h}},bg,s)[0,0];z=-(x[:,1:]-bg['q']*x[:,:-1]);bb=(z@h.T/.2).item();H=float(np.sum(h*h)/.2)
 integral=quad(lambda aa:np.sqrt(2/np.pi)/s*np.exp(-aa*aa/(2*s*s)+bb*aa-H*aa*aa/2),0,np.inf)[0]
 assert abs(val-np.log(integral))<1e-8
 # Same source intervention changes genuinely different causal laws.
 aa=draw(3,.35,.5,99998,mode='active');assert not np.allclose(aa['A_first']['x'],aa['B_direct']['x'])
 result=dict(max_exact_kernel_vs_independent_ODE_error=max(errors),representation_equivalence=True,causal_passive_X_U_equivalence=True,source_intervention_separates_causal_pair=True,proper_amplitude_integral_error=abs(val-np.log(integral)))
 (OUT/'validation.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
if __name__=='__main__':main()

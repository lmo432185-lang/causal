import json,itertools,numpy as np
from run import filtering,smooth,finite_lag,kl,OUT
p=dict(beta=-.01,mu=np.array([-.05,.02]),var=.01,A=np.array([[.8,.2],[.1,.9]]),pi=np.array([.3,.7]))
x=np.array([[1,1.1,1.03,1.06,.94,.99],[2,1.9,2.1,2.05,1.9,1.85]])
e,prior,post=filtering(x,p);ends=np.array([5,3]);sm=smooth(e,prior,post,p['A'],ends)
for i,end in enumerate(ends):
 paths=list(itertools.product([0,1],repeat=int(end)));weights=[]
 for states in paths:
  w=p['pi'][states[0]]*e[i,0,states[0]]
  for j in range(1,end):w*=p['A'][states[j-1],states[j]]*e[i,j,states[j]]
  weights.append(w)
 weights=np.array(weights);weights/=weights.sum();exact=np.array([sum(w for w,states in zip(weights,paths) if states[j]==0) for j in range(end)])
 np.testing.assert_allclose(sm[i,:end],exact,atol=1e-12)
 assert np.isnan(sm[i,end:]).all()
# Endpoint truncation really excludes future X values.
changed=x.copy();changed[1,4:]+=9
ec,pc,qc=filtering(changed,p);other=smooth(ec,pc,qc,p['A'],ends);np.testing.assert_allclose(sm[1,:3],other[1,:3],atol=1e-12)
# Lagged posterior independently compared to endpoint smoothing for each target.
targets=np.array([[0,1,3],[0,2,4]])
for lag in [0,1,2,10]:
 a=finite_lag(e,prior,p['A'],targets,lag)
 for i in range(2):
  for j in range(3):
   t=targets[i,j]
   if lag==0:expected=prior[i,t,0]
   else:expected=smooth(e[i:i+1],prior[i:i+1],post[i:i+1],p['A'],np.array([min(5,t+lag)]))[0,t]
   np.testing.assert_allclose(a[i,j],expected,atol=1e-12)
np.testing.assert_allclose(kl(np.array([.2,.5,.9]),np.array([.2,.5,.9])),0,atol=1e-12)
assert np.all(kl(np.array([.2,.8]),np.array([.7,.1]))>=0)
checks=dict(variable_endpoint_smoothing_vs_exact_enumeration='passed',future_after_endpoint_excluded='passed',finite_lag_vs_independent_endpoint_smoothing='passed',KL_zero_and_nonnegative='passed')
(OUT/'validation.json').write_text(json.dumps(checks,indent=2));print(checks)
from run import evaluate
from source_A import simulate
case=simulate(2,.35,1,1,123);ar,info,ex=evaluate(case,p,0.,np.array([11.4,np.nan]),dict(sigma=.35,tau=1.,strength=1.,seed=11))
assert len(ar)==12 and len(info)==2
assert all(r['estimate_available']==0 and r['onset_error_capped']==15 for r in ar if r['trajectory']==1)
checks['end_to_end_scoring_and_missing_alarm']='passed'
(OUT/'validation.json').write_text(json.dumps(checks,indent=2));print('End-to-end scoring passed')

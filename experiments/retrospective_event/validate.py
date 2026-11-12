import itertools,json
import numpy as np
from run import fb,infer,smooth_to_endpoint,OUT
p=dict(beta=-.01,mu=np.array([-.05,.02]),var=.01,A=np.array([[.8,.2],[.1,.9]]),pi=np.array([.3,.7]))
x=np.array([[1,1.1,1.03,1.06,.94,.99]])
e,prior,post,b,g,ll=fb(x,p)
# Enumerate every hidden state path; compare exact smoothing probabilities.
weights=[]; paths=list(itertools.product([0,1],repeat=5))
for states in paths:
 w=p['pi'][states[0]]*e[0,0,states[0]]
 for j in range(1,5):w*=p['A'][states[j-1],states[j]]*e[0,j,states[j]]
 weights.append(w)
weights=np.array(weights); weights/=weights.sum(); exact=np.array([sum(w for w,states in zip(weights,paths) if states[j]==0) for j in range(5)])
np.testing.assert_allclose(g[0,:,0],exact,atol=1e-12)
np.testing.assert_allclose(smooth_to_endpoint(e,prior,p,5)[0],exact,atol=1e-12)
# Fixed lag 10 must equal full smoothing when the sequence is only 5 steps.
out,_=infer(x,p);np.testing.assert_allclose(out[10],out[500],atol=1e-12)
# Causal estimates up through t unchanged if observations after t are modified.
y=x.copy(); y[:,3:]+=3
po,_=infer(y,p);np.testing.assert_allclose(out[0][:,:3],po[0][:,:3],atol=1e-12)
# Fixed-lag inference at t may use exactly, and no more than, t+lag samples.
rng=np.random.default_rng(1); long=np.cumsum(rng.normal(0,.1,(1,60)),axis=1); out,_=infer(long,p); alt=long.copy();alt[:,31:]+=7; other,_=infer(alt,p)
np.testing.assert_allclose(out[10][:,:21],other[10][:,:21],atol=1e-12)
checks=dict(exact_HMM_enumeration='passed',snapshot_smoothing='passed',finite_lag_full_equivalence='passed',causal_future_exclusion='passed',fixed_lag_future_boundary='passed')
(OUT/'validation.json').write_text(json.dumps(checks,indent=2));print(checks)

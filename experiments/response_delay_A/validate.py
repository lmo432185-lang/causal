import json,numpy as np
from scipy.linalg import expm
from run import simulate,forward_score,observed_score,fit_normal,threshold,alarm_times,DT,OUT
from support import fb
p=np.load(OUT/'model_1_11.npz');p={k:p[k] for k in ['beta','mu','var','A','pi']}
a=simulate(4,.35,.5,1,99,True);b=simulate(4,.35,.5,1,99,False)
np.testing.assert_array_equal(a['x'][:,:201],b['x'][:,:201]);np.testing.assert_array_equal(a['u'][:,:201],0)
# U reaches 1-exp(-1) exactly tau seconds after the step.
for tau in [.1,.5,1,2]:
 c=simulate(2,0,tau,1,100);np.testing.assert_allclose(c['u'][:,200+round(tau/DT)],1-np.exp(-1),atol=1e-13)
 # Independent matrix exponential for one deterministic augmented step.
 A=np.array([[-.15,-1.5,1.5],[0,-1/tau,1/tau],[0,0,0.]])
 state=np.array([c['x'][0,200],0,1.]);expected=expm(A*DT)@state
 np.testing.assert_allclose(c['x'][0,201],expected[0],atol=1e-12);np.testing.assert_allclose(c['u'][0,201],expected[1],atol=1e-12)
s=forward_score(a['x'],p);changed=a['x'].copy();changed[:,251:]+=99;np.testing.assert_allclose(s[:,:250],forward_score(changed,p)[:,:250],atol=1e-12)
# Causal implementation agrees with the earlier forward part, not smoothing.
post=fb(a['x'],p)[2];np.testing.assert_allclose(s,np.log(post[:,:,0])-np.log(post[:,:,1]),atol=1e-9)
normal=fit_normal(b['x']);np.testing.assert_allclose(observed_score(a['x'],normal)[:,:250],observed_score(changed,normal)[:,:250],atol=1e-12)
# Alarm clock is the second endpoint; never the first backdated crossing.
test=np.array([[0,3,4,0.]]);np.testing.assert_allclose(alarm_times(test,2),3*DT)
# Exact OU noise variance, checked by Monte Carlo variance of residuals.
sigma=.35;null=simulate(10000,sigma,1,0,789,False);q=np.exp(-.15*DT);target=sigma**2*(1-q*q)/(.3);res=null['x'][:,1]-q*null['x'][:,0]-1.5*(1-q)/.15
assert abs(res.var()/target-1)<.05
checks=dict(common_prechange_observations='passed',first_order_response_timescale='passed',independent_matrix_exponential='passed',HMM_future_exclusion='passed',previous_forward_equivalence='passed',T2_future_exclusion='passed',alarm_endpoint_indexing='passed',OU_noise_variance='passed')
(OUT/'validation.json').write_text(json.dumps(checks,indent=2));print(checks)

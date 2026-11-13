import json,numpy as np
from run import templates,bank,infer,reconstruct,forecast,OUT,TAUGRID
from source_A import simulate,fit_normal
base=dict(q=np.exp(-.15*.05),c=1.5*(1-np.exp(-.15*.05))/.15,var=.006)
# Exact deterministic dynamics, fitted without providing true onset or tau.
case=simulate(1,0,.5,1,99);b=bank(case['x'],base);est=infer(b,0)
assert abs(est['on'][0]-10)<1e-12
assert abs(est['tau'][0]-.5)<1e-12
np.testing.assert_allclose(est['amp'],1.5,atol=1e-9)
y,u=reconstruct(est);np.testing.assert_allclose(u[0]/1.5,case['u'][0],atol=1e-10)
# FFT bank versus direct dot product and analytic residual regression.
r=-(case['x'][:,1:]-base['q']*case['x'][:,:-1]-base['c']);w=templates(500,base)
for k,j in [(3,12),(20,190),(35,300)]:
 start=b['starts'][j];h=w[k,:500-start];dot=np.dot(r[0,start:],h);norm=h@h
 np.testing.assert_allclose(b['score'][0,k,j],max(dot,0)**2/(norm*base['var']),rtol=1e-8,atol=1e-8)
# Prefix inference and out-of-sample forecasts use no future observations.
prefix=case['x'][:,:301];bp=bank(prefix,base);ep=infer(bp,0);pred=forecast(ep,prefix[:,-1],base)
np.testing.assert_allclose(pred[0],case['x'][0,[301,310,340,400]],atol=1e-8)
# Shape-identifiability counterexample: different tau/source trajectories can
# generate exactly the same U, so no X-only estimator can distinguish them.
t=np.linspace(0,15,301);tau0=1.;tau1=2.;U=1-np.exp(-t/tau0);dU=np.exp(-t/tau0)/tau0;Y0=U+tau0*dU;Y1=U+tau1*dU
np.testing.assert_allclose((Y0-U)/tau0,(Y1-U)/tau1,atol=1e-14)
# Source/gain scale symmetry.
np.testing.assert_allclose(1.5*U,.75*(2*U),atol=1e-14)
checks=dict(noiseless_joint_onset_tau_amplitude='passed',exact_transfer_state_reconstruction='passed',FFT_likelihood_vs_direct='passed',prefix_only_forecast_matches_deterministic_truth='passed',different_source_tau_same_transfer='passed',gain_source_scale_ambiguity='passed')
(OUT/'validation.json').write_text(json.dumps(checks,indent=2));print(checks)
from run import old_forecast
# Pipeline prefix isolation: changing any X after cut cannot change forecast inputs.
case2=simulate(1,.35,1,1,123);changed=case2['x'].copy();changed[:,301:]+=100
b1=bank(case2['x'][:,:301],base);b2=bank(changed[:,:301],base)
np.testing.assert_array_equal(b1['score'],b2['score'])
e1=infer(b1,0);e2=infer(b2,0)
np.testing.assert_array_equal(forecast(e1,case2['x'][:,300],base),forecast(e2,changed[:,300],base))
checks['prediction_future_exclusion']='passed'
(OUT/'validation.json').write_text(json.dumps(checks,indent=2));print('Prediction future exclusion passed')

import json,time,numpy as np
from settings import ROOT
from bayes import sample,diagnostics,CENTER,make_model
# With no observations the target is exactly the specified truncated Gaussian prior.
m=make_model(CENTER,1.);r=sample(np.empty((1,0)),m,96111);mean=r['chain'].mean((0,1));rh,ess=diagnostics(np.zeros((2,100,5)));assert all(not np.isfinite(v) for v in rh) and all(v==0 for v in ess);assert np.max(abs(mean-CENTER))<1.2
(ROOT/'MCMC_DEVELOPMENT.json').write_text(json.dumps(dict(time=time.time(),no_formal_data=True,prior_target_mean_error=(mean-CENTER).tolist(),Rhat=r['rhat'],ESS=r['ESS'],frozen_chain_diagnostic_correct=True,passed=True),indent=2));print('Sampler prior-target development passed',mean-CENTER)

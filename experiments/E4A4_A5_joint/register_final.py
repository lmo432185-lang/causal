from pathlib import Path
import json,time,hashlib
from settings import ROOT
assert not (ROOT/'TEST_A_FROZEN.json').exists()
assert json.loads((ROOT/'MCMC_DEVELOPMENT.json').read_text())['passed']
assert json.loads((ROOT/'FORECAST_JOINT_DEVELOPMENT.json').read_text())['passed']
groups={
'PREDICTION_REGISTERED':['confirm.py','evaluate.py','calibration_setup.py','probability.py','probability_development.py','extra_plans.py','preflight.py','PRETEST_METRIC_AMENDMENT.md','BAYES_PROTOCOL.md','SHORT_PROTOCOL.md','SCHEMA_ERRATUM.md'],
'POSTTEST_REGISTERED':['state_audit.py','forecast_joint.py','forecast_joint_development.py','audit.py','analysis.py','sensitivity_audit.py','parameter_audit.py','interpretation.py','POSTTEST_PLAN.md'],
'BAYES_REGISTERED':['bayes.py','bayes_fit.py','bayes_likelihood.cpp','bayes_development.py','mcmc_development.py','BAYES_PROTOCOL.md']}
for label,files in groups.items():(ROOT/(label+'.json')).write_text(json.dumps(dict(time=time.time(),before_fresh_confirmation_and_tests=True,files={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}),indent=2))
(ROOT/'REGISTRATION_COMPLETE.json').write_text(json.dumps(dict(time=time.time(),all_models_and_statistics_registered_before_test=True,primary_length=5000,secondary_train_prefix=1000,primary_comparisons=6,secondary_comparisons=4,mechanism_comparisons=3,capacity_comparisons=4),indent=2));print('Final pre-test registration sealed')

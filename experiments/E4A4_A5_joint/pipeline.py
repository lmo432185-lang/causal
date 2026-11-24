import subprocess,time,json
from settings import ROOT
while not ((ROOT/'CANDIDATES_FROZEN.json').exists() and (ROOT/'REGISTRATION_COMPLETE.json').exists()):time.sleep(5)
STAGES=[('preflight.py','PREFLIGHT_COMPLETE.json'),('calibration_setup.py','CALIBRATION_SETUP_FROZEN.json'),('bayes_fit.py','BAYES_FROZEN.json'),('extra_plans.py','EXTRA_PLANS_FROZEN.json'),('confirm.py','FINAL_MODELS_FROZEN.json'),('interpretation.py','INTERPRETATION_COMPLETE.json'),('evaluate.py','TEST_B_FROZEN.json'),('forecast_joint.py','JOINT_X_COMPLETE.json'),('state_audit.py','TRUTH_AUDIT_COMPLETE.json'),('sensitivity_audit.py','SENSITIVITY_COMPLETE.json'),('parameter_audit.py','PARAMETER_AUDIT_COMPLETE.json'),('audit.py','AUDIT_COMPLETE.json'),('analysis.py','ANALYSIS_COMPLETE.json')]
for script,marker in STAGES:
 if (ROOT/marker).exists():continue
 print('START',script,time.time(),flush=True)
 with (ROOT/(script[:-3]+'.log')).open('w') as log:subprocess.run(['/workspace/causal-venv/bin/python',str(ROOT/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('DONE',script,time.time(),flush=True)
(ROOT/'PIPELINE_COMPLETE.json').write_text(json.dumps(dict(time=time.time()),indent=2))

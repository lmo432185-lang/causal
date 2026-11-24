import json,time,subprocess
from settings import ROOT
# Primary long fitting is completed first; paired sensitivity then shares spare CPU.
while not (ROOT.parent/'CANDIDATES_FROZEN.json').exists():time.sleep(5)
for script,marker in [('run_fit.py','CANDIDATES_FROZEN.json'),('preflight.py','PREFLIGHT_COMPLETE.json'),('extra_plans.py','EXTRA_PLANS_FROZEN.json'),('confirm.py','FINAL_MODELS_FROZEN.json'),('evaluate.py','TEST_B_FROZEN.json'),('short_audit.py','AUDIT_COMPLETE.json'),('short_analysis.py','ANALYSIS_COMPLETE.json')]:
 if (ROOT/marker).exists():continue
 print('START',script,time.time(),flush=True)
 with (ROOT/(script[:-3]+'.log')).open('w') as log:subprocess.run(['/workspace/causal-venv/bin/python',str(ROOT/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('DONE',script,time.time(),flush=True)
(ROOT/'PIPELINE_COMPLETE.json').write_text(json.dumps(dict(time=time.time()),indent=2))

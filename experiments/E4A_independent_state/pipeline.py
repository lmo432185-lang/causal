"""Resume-safe sequential final experiment. No retuning on confirmation/test outputs."""
import subprocess,time,json
from settings import ROOT
python='/workspace/causal-venv/bin/python'
while not (ROOT/'MODELS_FROZEN.json').exists():time.sleep(5)
for script,marker in [('confirm.py','DECISIONS_FROZEN.json'),('evaluate.py','TEST_B_FROZEN.json'),('state_evaluate.py','TRUTH_AUDIT_COMPLETE.json'),('audit.py','AUDIT_COMPLETE.json'),('analysis.py','ANALYSIS_COMPLETE.json')]:
 if (ROOT/marker).exists():continue
 print('START',script,time.time(),flush=True)
 with (ROOT/(script[:-3]+'.log')).open('w') as log:subprocess.run([python,str(ROOT/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('DONE',script,time.time(),flush=True)
(ROOT/'PIPELINE_COMPLETE.json').write_text(json.dumps(dict(time=time.time(),all_stages_complete=True),indent=2))

import subprocess,time,json
from settings import ROOT
while not (ROOT/'REPORT_RENDERED.json').exists():
 if (ROOT/'render_when_ready.log').exists() and 'Traceback' in (ROOT/'render_when_ready.log').read_text():raise RuntimeError('Inspect rendering failure first')
 time.sleep(5)
for script,marker in [('rotation_audit.py','ROTATION_AUDIT_COMPLETE.json'),('state_probability_extra.py','STATE_EXTRA_COMPLETE.json')]:
 if (ROOT/marker).exists():continue
 with (ROOT/(script[:-3]+'.log')).open('w') as log:subprocess.run(['/workspace/causal-venv/bin/python',str(ROOT/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
print('EXTRA AUDITS COMPLETE',flush=True)

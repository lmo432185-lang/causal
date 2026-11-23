import subprocess,time,json
from settings import ROOT
while not (ROOT/'PIPELINE_COMPLETE.json').exists():
 if (ROOT/'pipeline.log').exists() and 'Traceback' in (ROOT/'pipeline.log').read_text():raise RuntimeError('Numerical pipeline failed; inspect logs before rendering')
 time.sleep(5)
for script in ['interpret.py','figures.py','report.py']:
 print('START',script,time.time(),flush=True)
 with (ROOT/(script[:-3]+'.log')).open('w') as log:subprocess.run(['/workspace/causal-venv/bin/python',str(ROOT/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('DONE',script,time.time(),flush=True)
(ROOT/'REPORT_RENDERED.json').write_text(json.dumps(dict(time=time.time(),figures=13),indent=2))

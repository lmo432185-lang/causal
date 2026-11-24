"""Wait for immutable numerical outputs, then produce descriptive artifacts."""
import time,subprocess,json
from settings import ROOT
while not all((ROOT/p).exists() for p in ['PIPELINE_COMPLETE.json','short/ANALYSIS_A_COMPLETE.json','short/AUDIT_A_COMPLETE.json','small_validation/COMPLETE.json']):time.sleep(5)
for script in ['diagnostic_summary.py','length_effects.py','memory_signal_audit.py','figures.py','report.py']:
 print('START',script,flush=True)
 subprocess.run(['/workspace/causal-venv/bin/python',str(ROOT/script)],cwd=ROOT,check=True)
 print('DONE',script,flush=True)
(ROOT/'OUTPUTS_COMPLETE.json').write_text(json.dumps(dict(time=time.time(),figures=15,report_formats=['md','html','pdf']),indent=2))

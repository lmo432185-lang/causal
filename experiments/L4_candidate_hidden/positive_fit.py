# This process imports no private positive generator or manifest.
import json,time,hashlib
from settings import ROOT,OUT
from fit_jobs import fit_job
from extended_audit import delayed_counterparts
from ssm import restore
if __name__=='__main__':
 jobs=json.loads((ROOT/'positive_public_jobs.json').read_text());files={}
 for job in jobs:
  fit_job(job);d=OUT/job['case'];m=json.loads((d/'selected_parameters.json').read_text())['models'];dec=json.loads((d/'X_only_decisions.json').read_text());alt={k:delayed_counterparts(restore(v)) for k,v in m.items() if k!='single'};guard=[bool(dec['guarded_declared'][i] and not alt[k]) for i,k in enumerate(dec['prediction'])];f=d/'supplementary_law_audit.json';f.write_text(json.dumps(dict(additional_alternatives=alt,supplementary_guarded_declared=guard,primary_decisions_unchanged=True),indent=2))
  for name in ['selected_parameters.json','X_only_decisions.json','LOCAL_X_FREEZE.json','supplementary_law_audit.json']:
   f=d/name;files[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
 (ROOT/'POSITIVE_ALL_X_FROZEN.json').write_text(json.dumps(dict(time=time.time(),files=files,truth_not_opened=True),indent=2));print('Positive control X-only frozen',flush=True)

import json,time,hashlib
from settings import ROOT
from evaluate import evaluate_job
if __name__=='__main__':
 while not (ROOT/'POSITIVE_ALL_X_FROZEN.json').exists():time.sleep(5)
 f=json.loads((ROOT/'POSITIVE_ALL_X_FROZEN.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert all(sha(ROOT/k)==v for k,v in f['files'].items());jobs=json.loads((ROOT/'data/positive_sealed_manifest.json').read_text())
 for j in jobs:evaluate_job(j)
 assert all(sha(ROOT/k)==v for k,v in f['files'].items());(ROOT/'POSITIVE_EVALUATION_DONE.json').write_text(json.dumps(dict(time=time.time(),cases=len(jobs),X_only_hashes_unchanged=True),indent=2));print('Positive evaluation complete',flush=True)

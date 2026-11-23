import json,time,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
from settings import *
from workflow import fit_job

def main():
 for f,h in json.loads((ROOT/'CODE_REGISTERED.json').read_text())['files'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
 jobs=json.loads((ROOT/'jobs.json').read_text())
 with ProcessPoolExecutor(max_workers=3) as pool:
  for f in as_completed([pool.submit(fit_job,j) for j in jobs]):f.result()
 files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.json.gz')};assert len(files)==480;(ROOT/'CANDIDATES_FROZEN.json').write_text(json.dumps(dict(time=time.time(),files=files),indent=2));print('480 CANDIDATES FROZEN',flush=True)
if __name__=='__main__':main()

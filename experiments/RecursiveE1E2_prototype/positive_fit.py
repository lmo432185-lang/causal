import json,time,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from settings import ROOT,OUT
from fit import fit_case

def main():
 jobs=json.loads((ROOT/'positive_public_jobs.json').read_text())
 with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool:
  for task in as_completed([pool.submit(fit_case,j) for j in jobs]):task.result()
 files={}
 for j in jobs:
  for f in ['model.json','LOCAL_MODEL_FREEZE.json']:
   p=OUT/j['case']/f;files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 (ROOT/'POSITIVE_MODELS_FROZEN.json').write_text(json.dumps(dict(time=time.time(),files=files,model_count=6),indent=2))
if __name__=='__main__':main()

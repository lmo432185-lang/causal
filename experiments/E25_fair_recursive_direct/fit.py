"""Read only training and selection X. Freeze all models before confirmation."""
import json,time,hashlib,resource
from concurrent.futures import ProcessPoolExecutor,as_completed
from settings import *
from model import train,save
import numpy as np

def fit_case(job):
 data=np.load(DATA/(job['case']+'.npz'));x=data['train'];v=data['selection'];d=OUT/job['case'];d.mkdir(exist_ok=True,parents=True);meta=[]
 for n in SIZES:
  p=d/f'model_n{n}.json';assert not p.exists();m=train(x[:n],v);m['case']=job['case'];m['seed']=job['seed'];m['peak_RSS_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;save(p,m);meta.append(dict(n=n,cpu=m['total_cpu'],rss=m['peak_RSS_kib']))
 print('FITTED',job['case'],meta,flush=True)

def main():
 reg=json.loads((ROOT/'CODE_REGISTERED.json').read_text())
 for f,h in reg['files'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
 jobs=json.loads((ROOT/'jobs.json').read_text())
 with ProcessPoolExecutor(max_workers=2) as pool:
  for future in as_completed([pool.submit(fit_case,j) for j in jobs]):future.result()
 files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*/model_n*.json'))};assert len(files)==330
 (ROOT/'MODELS_FROZEN.json').write_text(json.dumps(dict(time=time.time(),files=files),indent=2));print('ALL MODELS FROZEN',len(files),flush=True)
if __name__=='__main__':main()

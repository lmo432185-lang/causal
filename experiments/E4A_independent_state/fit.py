"""Producer/learner boundary: true states never passed to fitting functions."""
import json,time,hashlib,resource
from concurrent.futures import ProcessPoolExecutor,as_completed
from settings import *
from producer import jobs,generate,digest
from learn import fit_all
from model import pack

def work(j):
 x=generate(j['index'],j['seed'],0,70);xr=x[:20];xd=x[20:40];v=x[40:50];d=OUT/j['case'];d.mkdir(parents=True,exist_ok=True);h={name:digest(a) for name,a in [('train_R',xr),('train_D',xd),('selection',v),('confirmation_A',x[50:60]),('confirmation_B',x[60:])]};timing=[]
 for length in LENGTHS:
  p=d/f'model_{length}.json'
  if p.exists():
   m=json.loads(p.read_text());assert m['case']==j['case'] and m['length']==length;continue
  m=fit_all(xr[:,:length],xd[:,:length],v[:,:length],j['seed']);m['case']=j['case'];m['seed']=j['seed'];m['length']=length;m['peak_RSS_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;p.write_text(json.dumps(pack(m)));timing.append([length,round(m['cpu'],2)])
 (d/'INPUT_HASHES.json').write_text(json.dumps(h,indent=2));print('FITTED',j['case'],timing,flush=True)

def main():
 for f,h in json.loads((ROOT/'CODE_REGISTERED.json').read_text())['files'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
 js=jobs();(ROOT/'jobs.json').write_text(json.dumps(js,indent=2))
 with ProcessPoolExecutor(max_workers=3) as pool:
  for f in as_completed([pool.submit(work,j) for j in js]):f.result()
 files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*/model_*.json')};assert len(files)==800;(ROOT/'MODELS_FROZEN.json').write_text(json.dumps(dict(time=time.time(),files=files),indent=2));print('MODELS FROZEN',len(files),flush=True)
if __name__=='__main__':main()

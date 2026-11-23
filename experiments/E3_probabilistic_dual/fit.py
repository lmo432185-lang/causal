"""Orchestration only: private producer sends X arrays to the X-only learning API."""
import json,time,hashlib,resource
from concurrent.futures import ProcessPoolExecutor,as_completed
from settings import *
from producer import jobs,simulate,digest
from learn import fit_all
from ssm import pack

def work(job):
 case=job['case'];d=OUT/case;d.mkdir(exist_ok=True,parents=True);x=simulate(job['index'],job['seed'],0,880);xr=x[:400];xd=x[400:800];v=x[800:840];hashes={k:digest(a) for k,a in [('train_R',xr),('train_D',xd),('selection',v),('confirmation',x[840:])]};summary=[]
 for n in SIZES:
  p=d/f'model_{n}.json';assert not p.exists();m=fit_all(xr,xd,v,n,job['seed']);m['seed']=job['seed'];m['case']=case;m['peak_RSS_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;p.write_text(json.dumps(pack(m)));summary.append(round(m['cpu'],3))
 (d/'INPUT_HASHES.json').write_text(json.dumps(hashes,indent=2));print('FITTED',case,summary,flush=True)

def main():
 reg=json.loads((ROOT/'CODE_REGISTERED.json').read_text())
 for f,h in reg['files'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
 js=jobs();(ROOT/'jobs.json').write_text(json.dumps(js,indent=2))
 with ProcessPoolExecutor(max_workers=2) as p:
  for f in as_completed([p.submit(work,j) for j in js]):f.result()
 files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*/model_*.json'))};assert len(files)==1700
 (ROOT/'MODELS_FROZEN.json').write_text(json.dumps(dict(time=time.time(),files=files),indent=2));print('MODELS FROZEN',len(files),flush=True)
if __name__=='__main__':main()

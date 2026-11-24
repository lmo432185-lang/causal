"""Pre-test continuous-parameter posterior diagnostic for four NEW C conditions."""
import gzip,json,time,hashlib,numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
from settings import *
from model import pack,unpack
from producer import generate,digest
from bayes import sample,make_model

def work(j):
 path=OUT/'bayes'/(j['case']+'.json.gz');path.parent.mkdir(exist_ok=True)
 if path.exists():return
 with gzip.open(OUT/(j['case']+'.json.gz'),'rt') as f:b=unpack(json.load(f))
 raw=generate(j['index'],j['seed'],0,70);assert digest(raw[:20])==b['training_hashes']['R'];m=b['bases']['R']['canonical_point'];scale=m['scale'];x=np.ascontiguousarray(np.concatenate([raw[:20],raw[40:50]])/scale);tic=time.process_time();result=sample(x,m,j['seed']+78621);chain=result['chain'];chosen=np.array([374,1124,1874,2624]);models=[make_model(z,scale) for c in chain for z in c[chosen]];models16=[make_model(z,scale) for c in chain for z in c[np.arange(187,3000,375)[:8]]];result.update(case=j['case'],seed=j['seed'],scale=scale,models8=models,models16=models16,point_model=make_model(result['mode'],scale),cpu=time.process_time()-tic,data='R20 + independent selection10, counted once; no test/hidden truth',draws=3000,warmup=1000)
 tmp=path.with_suffix('.tmp')
 with gzip.open(tmp,'wt') as f:json.dump(pack(result),f)
 tmp.replace(path);print('BAYES',j['case'],round(result['cpu'],2),max(result['rhat']),min(result['ESS']),flush=True)

def main():
 assert not (ROOT/'TEST_A_FROZEN.json').exists()
 for f,h in json.loads((ROOT/'BAYES_REGISTERED.json').read_text())['files'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
 js=[j for j in json.loads((ROOT/'jobs.json').read_text()) if j['index']>=20]
 with ProcessPoolExecutor(max_workers=3) as pool:
  for f in as_completed([pool.submit(work,j) for j in js]):f.result()
 files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'bayes').glob('*.json.gz')};assert len(files)==80;(ROOT/'BAYES_FROZEN.json').write_text(json.dumps(dict(time=time.time(),files=files,diagnostic_only=True,tests_not_opened=True),indent=2));print('80 continuous posterior fits frozen',flush=True)
if __name__=='__main__':main()

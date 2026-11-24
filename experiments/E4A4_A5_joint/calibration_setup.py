"""X-only calibration parameter prior setup. No supervised state calibration."""
import json,gzip,time,hashlib,numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
from scipy.special import logsumexp
from settings import *
from model import unpack,pack
from producer import generate
import native

def work(j):
 with gzip.open(OUT/(j['case']+'.json.gz'),'rt') as f:b=unpack(json.load(f))
 x=generate(j['index'],j['seed'],0,70)[40:50];r={}
 for bank in ['R','D']:
  ms=b['bases'][bank]['canonical_bootstrap'];ll=np.array([native.run(x/m['scale'],m)['ll'].sum()-x.size*np.log(m['scale']) for m in ms]);r[bank]=(ll-logsumexp(ll)).tolist()
 return j['case'],r

def main():
 assert (ROOT/'CANDIDATES_FROZEN.json').exists();rows={}
 with ProcessPoolExecutor(max_workers=3) as pool:
  for f in as_completed([pool.submit(work,j) for j in json.loads((ROOT/'jobs.json').read_text())]):k,v=f.result();rows[k]=v
 p=OUT/'canonical_priors.json';p.write_text(json.dumps(rows));(ROOT/'CALIBRATION_SETUP_FROZEN.json').write_text(json.dumps(dict(time=time.time(),uses_hidden_truth=False,files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()}),indent=2));print('480 X-only canonical parameter priors frozen',flush=True)
if __name__=='__main__':main()

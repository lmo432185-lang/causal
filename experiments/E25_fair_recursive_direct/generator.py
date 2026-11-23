"""Simulation producer; never imported by fit/confirmation/test programs."""
import json,hashlib,time
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from settings import ROOT,DATA,SEEDS,T

def simulate(meta,seed):
 A=np.array(meta['A']);q=np.array(meta['Q']);rng=np.random.default_rng(np.random.SeedSequence([20261123,250,seed,int(meta['index'])]));s=rng.multivariate_normal(np.zeros(len(q)),solve_discrete_lyapunov(A,np.diag(q)),size=800);x=np.empty((800,T))
 for t in range(T):
  if t:s=s@A.T+rng.normal(size=s.shape)*np.sqrt(q)
  x[:,t]=s[:,-1]
 return x

def main():
 reg=json.loads((ROOT/'CODE_REGISTERED.json').read_text())
 for f,h in reg['files'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
 DATA.mkdir(exist_ok=True);jobs=[];manifest=[]
 for i,m in enumerate(json.loads((ROOT/'SYSTEMS.json').read_text())):
  for seed in SEEDS:
   case=m['id']+'_'+str(seed);p=DATA/(case+'.npz');assert not p.exists();x=simulate({**m,'index':i},seed)
   np.savez_compressed(p,train=x[:400],selection=x[400:500],confirmation=x[500:600],test_A=x[600:700],test_B=x[700:800]);jobs.append(dict(case=case,seed=seed));manifest.append(dict(case=case,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),shape=list(x.shape)))
 (ROOT/'jobs.json').write_text(json.dumps(jobs,indent=2));(ROOT/'INPUT_MANIFEST.json').write_text(json.dumps(dict(time=time.time(),files=manifest),indent=2));print('generated',len(jobs),'X-only datasets',flush=True)
if __name__=='__main__':main()

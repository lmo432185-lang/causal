"""Restore deterministic X-only input archives without changing frozen records."""
import json,hashlib,numpy as np
from settings import ROOT,DATA
from generator import simulate

def main():
 items={m['id']:{**m,'index':i} for i,m in enumerate(json.loads((ROOT/'SYSTEMS.json').read_text()))};manifest=json.loads((ROOT/'INPUT_MANIFEST.json').read_text());DATA.mkdir(exist_ok=True)
 for r in manifest['files']:
  case=r['case'];p=DATA/(case+'.npz')
  if not p.exists():
   cond,seed=case.rsplit('_',1);x=simulate(items[cond],int(seed));np.savez_compressed(p,train=x[:400],selection=x[400:500],confirmation=x[500:600],test_A=x[600:700],test_B=x[700:800])
  assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],case
 print('110 input archives verified/restored; frozen records unchanged')
if __name__=='__main__':main()

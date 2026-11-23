import json,time,hashlib,numpy as np,pandas as pd
from settings import *
from producer import SYSTEMS,simulate,digest
from ssm import unpack
from predict import infer,score

def main():
 checked=0;err=0.;inputs=0
 for fn in ROOT.glob('*REGISTERED.json'):
  m=json.loads(fn.read_text())
  for f,h in m.get('files',{}).items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,(fn.name,f);checked+=1
 for budget in ['em5','em25']:
  out=ROOT/'replication'/budget;times={}
  for name in ['MODELS_FROZEN','DECISIONS_FROZEN','TEST_A_FROZEN','TEST_B_FROZEN']:
   reg=json.loads((out/(name+'.json')).read_text());times[name]=reg['time']
   for f,h in reg['files'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h;checked+=1
  assert times['MODELS_FROZEN']<times['DECISIONS_FROZEN']<times['TEST_A_FROZEN']<times['TEST_B_FROZEN'];tables={b:pd.read_csv(out/f'test_{b}.csv.gz') for b in ['A','B']}
  for i,system in enumerate(SYSTEMS):
   for seed in range(3031,3041):
    case=system['id']+'_'+str(seed)
    for batch in ['A','B']:
     x=simulate(i,seed,1 if batch=='A' else 2,40);reg=json.loads((out/f'TEST_{batch}_FROZEN.json').read_text());assert digest(x)==reg['input_hashes'][case];inputs+=1
     if seed!=3031:continue
     m=unpack(json.loads((out/case/'model_100.json').read_text()));graph=json.loads((out/case/'graph.json').read_text())['100'];cache={};tab=tables[batch]
     for group in GROUPS:
      r=infer(x,m,graph,group,cache);scores=score(x,r,seed+100+500000);expected=tab[(tab.case==case)&(tab.group==group)].sort_values('trajectory')
      for key,z in scores.items():err=max(err,float(np.max(abs(z-expected[key].to_numpy()))))
  assert all(len(t)==47600 for t in tables.values())
 assert err<1e-8;out=dict(passed=True,registered_and_replica_freeze_hashes_checked=checked,replica_test_inputs_regenerated=inputs,max_replica_metric_replay_error=err,refitted=False);(ROOT/'REPLICATION_AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()

"""Post-freeze dimension-proxy audit, distinct from confirmed novelty counts."""
import json,gzip,csv,time
import numpy as np
from settings import *
from producer import simulate,SYSTEMS
from ssm import unpack
from state_evaluate import posterior

def main():
 assert (ROOT/'TEST_B_FROZEN.json').exists();count=0
 with gzip.open(OUT/'operational_states.csv.gz','wt') as stream:
  writer=None
  for j in json.loads((ROOT/'jobs.json').read_text()):
   trueK=len(SYSTEMS[j['index']]['A'])-1+(2 if SYSTEMS[j['index']]['stress']=='delay' else 0);m=unpack(json.loads((OUT/j['case']/'model_100.json').read_text()));graph=json.loads((OUT/j['case']/'graph.json').read_text())['100']
   for batch in ['A','B']:
    x=simulate(j['index'],j['seed'],1 if batch=='A' else 2,40);cache={}
    for group in ['B','D','E','F']:
     ids,parts,w=posterior(x,m,graph,group,cache);ks=np.array([k for b,k in ids]);mass=np.stack([w[:,:,ks==k].sum(2) for k in range(4)],-1)[:,40:];mapk=mass.argmax(2);pmax=mass.max(2)
     for i in range(40):
      row=dict(case=j['case'],seed=j['seed'],batch=batch,group=group,trajectory=i,true_dimension_proxy=trueK,expected_K=float(np.sum(mass[i]*np.arange(4),axis=1).mean()),MAP_abs_error=float(np.mean(abs(mapk[i]-trueK))),overextended_fraction=float(np.mean(mapk[i]>trueK)),confident_overextended_fraction=float(np.mean((mapk[i]>trueK)&(pmax[i]>=.8))),underextended_fraction=float(np.mean(mapk[i]<trueK)),uncertain_fraction=float(np.mean(pmax[i]<.8)))
      for k in range(4):row[f'mass_K{k}']=float(mass[i,:,k].mean())
      if writer is None:writer=csv.DictWriter(stream,fieldnames=row);writer.writeheader()
      writer.writerow(row);count+=1
   print('DIMENSION AUDIT',j['case'],flush=True)
 (ROOT/'OPERATIONAL_AUDIT_DONE.json').write_text(json.dumps(dict(time=time.time(),rows=count,dimensions_are_predictive_proxies_not_identified_physical_counts=True),indent=2))
if __name__=='__main__':main()

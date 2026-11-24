"""Post-test C5 descriptive baseline, using the already frozen single-memory parents.

No new model fit, threshold, hypothesis test, or modification of registered methods.
Physical U/V scores are deliberately not fabricated for memory coordinates.
"""
import json,gzip,time,numpy as np,pandas as pd
from settings import *
from producer import generate,SYSTEMS,digest,matrices
from model import unpack
from inference import mix
from probability import TIMES,marginal
def main():
 assert (ROOT/'TRUTH_AUDIT_COMPLETE.json').exists()
 rows=[]
 for j in json.loads((ROOT/'jobs.json').read_text()):
  if SYSTEMS[j['index']]['id'] not in ['CL','CM','CH','CE'] or j['seed'] not in [4011,4012,4013]:continue
  with gzip.open(OUT/(j['case']+'.json.gz'),'rt') as f:b=unpack(json.load(f))
  ids=['R_none','D_none'];pr=[-np.log(2)]*2
  for batch,stage in [('A',11),('B',12)]:
   x,y=generate(j['index'],j['seed'],stage,NTEST,truth=True);assert digest(x)==json.loads((ROOT/f'TEST_{batch}_FROZEN.json').read_text())['input_hashes'][j['case']];x=x[:5].copy();y=y[:5].copy()
   r,_=mix(x,b['models'],ids,pr,{});mu=[];var=[]
   for key,part in zip(ids,r['parts']):
    m=b['models'][key];mu.append(np.einsum('i,nti->nt',m['C'],part['post']['mean'][:,TIMES])*m['scale']);var.append(np.einsum('i,tij,j->t',m['C'],part['post']['cov'][TIMES],m['C'])*m['scale']**2)
   mu=np.stack(mu,2);v=np.broadcast_to(np.stack(var,1)[None],mu.shape);w=r['weights'][:,TIMES];target=(y@matrices(SYSTEMS[j['index']])[2])[:,TIMES];sc,_,_=marginal(mu.reshape(-1,2),v.reshape(-1,2),w.reshape(-1,2),target.ravel())
   for i in range(len(x)):rows.append(dict(case=j['case'],seed=j['seed'],batch=batch,method='C5_memory',trajectory=i,physical_UV_applicable=False,**{k:float(z.reshape(len(x),len(TIMES))[i].mean()) for k,z in sc.items()}))
 pd.DataFrame(rows).to_csv(OUT/'memory_signal_probability.csv.gz',index=False);(ROOT/'MEMORY_SIGNAL_AUDIT_COMPLETE.json').write_text(json.dumps(dict(time=time.time(),descriptive_posttest=True,model_source='frozen R_none/D_none single-memory parents',prior='equal, online test likelihood once',truth_used_for_fitting=False,physical_UV='not applicable',primary_results_modified=False),indent=2))
if __name__=='__main__':main()

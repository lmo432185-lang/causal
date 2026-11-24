import json,gzip,time,hashlib,numpy as np,pandas as pd
from settings import *
from model import unpack

def main():
 assert (ROOT/'CANDIDATES_FROZEN.json').exists();rows=[];particles=[]
 for file in OUT.glob('*.json.gz'):
  with gzip.open(file,'rt') as f:b=unpack(json.load(f))
  for key,m in b['models'].items():
   d=len(m['C']);radius=float(max(abs(np.linalg.eigvals(m['F'])))) if d else abs(m['a']);qmin=float(np.linalg.eigvalsh(m['Q']).min()) if d else 0.;pmin=float(np.linalg.eigvalsh(m['P']).min()) if d else 0.;assert radius<1 and qmin>=-1e-8 and pmin>=-1e-8 and m['R']>0,(b['case'],key,radius,qmin,pmin)
   rows.append(dict(case=b['case'],key=key,dimension=d,radius=radius,q_min=qmin,p_min=pmin,R=m['R'],iterations=m.get('iterations',0),converged=m.get('converged',True)))
  for name in ['A','B','E']:
   count=len(b['plans'][name]['ids']);particles.append(dict(case=b['case'],method=name,particles=count,nominal=16,complete=count==16,parameter_prior='uniform finite bootstrap atoms before selection'))
 pd.DataFrame(rows).to_csv(OUT/'preflight_models.csv.gz',index=False,compression='gzip');pd.DataFrame(particles).to_csv(OUT/'particle_counts.csv',index=False);(ROOT/'PREFLIGHT_COMPLETE.json').write_text(json.dumps(dict(time=time.time(),models=len(rows),all_stable_psd=True,complete_particle_plans=sum(x['complete'] for x in particles),total_particle_plans=len(particles)),indent=2));print('Preflight complete',len(rows),flush=True)
if __name__=='__main__':main()

"""Group-specific permitted observations only. Full test states are never opened."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json,time,hashlib
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from settings import *
from model import fit,design,block_ll,infer
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2))

def worker(job):
 start=time.time();case=job['case'];d=OUT/case;d.mkdir(parents=True,exist_ok=True)
 if (d/'LOCAL_X_FREEZE.json').exists():return case
 x=np.load(DATA/case/'X.npy');out={};results={};allfiles=[]
 for method in METHODS:
  # E3 with zero input is exactly the E0 information condition, not a spurious additional estimator.
  effective='E0' if method=='E3' and job['policy']=='passive' else method
  I=np.load(DATA/case/'known_input.npy') if effective=='E3' else None
  aux=None
  if method.startswith('E4'):
   z=np.load(DATA/case/'sparse_auxiliary.npz');keys=['Y_pair','Z_pair'] if method=='E4YZ' else ['Y','Z'];aux={s:z[k] for s,k in zip(['Y','Z'],keys)}
  anchor=json.loads((DATA/case/'E2_external_calibration.json').read_text())['c'] if method=='E2' else None
  def sliced_aux(sl):return None if aux is None else {k:v[sl] for k,v in aux.items()}
  if method=='E3' and job['policy']=='passive':
   results[method]={**results['E0'],'information_equivalent_to':'E0','zero_input_not_informative':True};best=results[method]['selected'];theta=np.array(best['theta']);lag=best['delay']
  else:
   Z=design(x[:TRAIN],effective,sliced_aux(slice(0,TRAIN)),None if I is None else I[:TRAIN]);S=Z.T@Z/len(Z);V=design(x[TRAIN:TRAIN+VALID],effective,sliced_aux(slice(TRAIN,TRAIN+VALID)),None if I is None else I[TRAIN:TRAIN+VALID]);records=[]
   schedule=[(1,k) for k in range(INITS*len(DELAYS))] if method in ['E0','E1','E2'] else [(lag,k) for lag in DELAYS for k in range(INITS)]
   for lag,k in schedule:
    theta,rec=fit(S,effective,lag,job['seed']+177000+101*k+1009*lag,anchor,MAXITER);val=block_ll(V,theta,lag,effective).reshape(VALID,-1).sum(1);records.append(dict(delay=lag,initialization=k,theta=theta.tolist(),validation_LL=float(val.sum()),validation_trajectory_LL=val.tolist(),fit=rec));print(case,method,lag,k,'nit',rec['nit'],'ok',rec['success'],flush=True)
   idx=int(np.argmax([r['validation_LL'] for r in records]));best=records[idx];theta=np.array(best['theta']);lag=best['delay'];results[method]=dict(selected=best,candidates=records,selection='independent permitted-observation validation Gaussian block likelihood; same parameter count across delay hypotheses',information=method,effective_model=effective)
  # In every group the test auxiliary states remain hidden. Only E3 has registered input values.
  p=infer(x[-TEST:],theta,lag,effective,None if I is None else I[-TEST:]);out[method+'_mean']=p['mean'];out[method+'_cov']=p['cov'];out[method+'_forecast']=p['forecast'];out[method+'_forecast_var']=p['forecast_var'];out[method+'_test_LL']=p['trajectory_ll']
  if method in ['E0','E1','E2'] or (method=='E3' and job['policy']=='passive'):
   results[method]['selected']['delay']=1;results[method]['canonical_delay_rule']='X-only likelihood is exactly delay invariant; output d1, retain all3';lag=1
   for alt in [2,3]:
    a=infer(x[-TEST:],theta,alt,effective);out[method+f'_d{alt}_mean']=a['mean'];out[method+f'_d{alt}_cov']=a['cov']
  save(d/f'{method}_models.json',results[method]);allfiles.append(f'{method}_models.json')
  print(case,method,'DONE',flush=True)
 np.savez_compressed(d/'permitted_predictions.npz',**out);allfiles.append('permitted_predictions.npz');save(d/'LOCAL_X_FREEZE.json',dict(time=time.time(),case=case,test_auxiliary_states_seen=False,files={f:sha(d/f) for f in allfiles}));print('CASE DONE',case,round(time.time()-start,1),'s',flush=True);return case

def main():
 jobs=json.loads((ROOT/'public_jobs.json').read_text());done=[]
 with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool:
  for f in as_completed([pool.submit(worker,j) for j in jobs]):done.append(f.result());print('PROGRESS',len(done),'/',len(jobs),flush=True)
 files={}
 for j in jobs:
  d=OUT/j['case'];m=json.loads((d/'LOCAL_X_FREEZE.json').read_text())
  for f in list(m['files'])+['LOCAL_X_FREEZE.json']:files[str((d/f).relative_to(ROOT))]=sha(d/f)
 save(ROOT/'ALL_PERMITTED_MODELS_FROZEN.json',dict(time=time.time(),cases=len(done),files=files,full_test_states_not_opened=True));print('ALL MODELS FROZEN',flush=True)
if __name__=='__main__':main()

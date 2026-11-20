"""X-only worker: deliberately does not import private_generator or load any sealed file."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json,time,hashlib
import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import lfilter
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from settings import *
from model import fit,plant,infer,block_statistics,serialize
from encoder import encode,linear_head,apply

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2))

def ar_fit(train,valid,test):
 g=np.array([(train[:,k:]*train[:,:train.shape[1]-k]).mean() for k in range(41)]);records=[];preds=[]
 for p in [1,5,20,40]:
  coef=np.linalg.solve(toeplitz(g[:p])+np.eye(p)*1e-6*g[0],g[1:p+1]);tr=lfilter(np.r_[0,coef],[1],train,axis=1);var=float(np.mean((train[:,50:]-tr[:,50:])**2));va=lfilter(np.r_[0,coef],[1],valid,axis=1);ll=float(-.5*np.mean((valid[:,50:]-va[:,50:])**2/var+np.log(2*np.pi*var)));records.append(dict(order=p,coef=coef.tolist(),variance=var,validation_LL_per_step=ll));preds.append(lfilter(np.r_[0,coef],[1],test,axis=1))
 k=int(np.argmax([v['validation_LL_per_step'] for v in records]));return records,k,preds[k]

def worker(job):
 begin=time.time();case=job['case'];seed=job['seed'];dest=OUT/case;dest.mkdir(parents=True,exist_ok=True)
 if (dest/'LOCAL_X_FREEZE.json').exists():return case
 x=np.load(DATA/case/'X.npy');mu=float(x[:TRAIN].mean());xc=x-mu;tr=xc[:TRAIN];va=xc[TRAIN:TRAIN+VALID];te=xc[-TEST:];S=block_statistics(tr);fits=[];models=[]
 for start in range(INITS):
  theta,rec=fit(S,seed+11000+101*start);par=plant(theta,1);v=infer(va,par);rec.update(start=start,theta=theta.tolist(),validation_LL=float(v['trajectory_ll'].sum()),validation_trajectory_LL=v['trajectory_ll'].tolist());fits.append(rec);models.append(theta);print(case,'start',start,'nit',rec['nit'],'ok',rec['success'],flush=True)
 k=int(np.argmax([v['validation_LL'] for v in fits]));theta=models[k];params=plant(theta,1);base=infer(te,params);out={};allpars={};lls={}
 for delay in DELAYS:
  p=plant(theta,delay);v=infer(te,p);out[f'B_d{delay}_mean']=v['mean'][:,:,:2];out[f'B_d{delay}_cov']=v['cov'][:,:2,:2];allpars[str(delay)]=serialize(p);lls[str(delay)]=v['trajectory_ll'].tolist()
 out['B_forecast']=base['forecast']+mu;out['B_forecast_var']=base['forecast_var'];out['B_d1_diag_mean']=out['B_d1_mean'];out['B_d1_diag_cov']=np.array([np.diag(np.diag(p)) for p in out['B_d1_cov']]);A,Q,R,P=params;gain=P[:2,2]/(P[2,2]+R);C=P[:2,:2]-np.outer(P[:2,2],P[2,:2])/(P[2,2]+R);out['B_current_mean']=te[:,:,None]*gain;out['B_current_cov']=C
 # Original L1 is frozen. New probabilistic readout learns only from X-only SSM pseudo-states and their covariance.
 f=encode(x,seed);trainpost=infer(tr,params);h=linear_head(f[:TRAIN],trainpost['mean'][:,19:,:2],trainpost['cov'][19:,:2,:2]);out['B_L1_mean']=apply(f[-TEST:],h);out['B_L1_cov']=np.array(h['covariance']);save(dest/'L1_Xonly_head.json',h)
 records,aridx,arpre=ar_fit(tr,va,te);out['A_AR_forecast']=arpre+mu;out['A_AR_forecast_var']=np.array(records[aridx]['variance']);save(dest/'AR_models.json',dict(models=records,selected=aridx,physical_YZ='not identified; predictive lag state has no physical Y/Z posterior'))
 boots=[]
 if job['bootstrap']:
  rng=np.random.default_rng(seed+170000);stats=np.stack([block_statistics(v[None]) for v in tr]);mm=[];cc=[]
  for i in range(BOOT):
   ids=rng.integers(0,TRAIN,TRAIN);bt,rec=fit(stats[ids].mean(0),seed+i,maxiter=BOOT_MAXITER,initial=theta);p=plant(bt,1);v=infer(te,p);mm.append(v['mean'][:,:,:2]);cc.append(v['cov'][:,:2,:2]);boots.append(dict(replicate=i,theta=bt.tolist(),fit=rec,training_trajectory_resampling_sha256=hashlib.sha256(ids.tobytes()).hexdigest()))
  mm=np.array(mm);avg=mm.mean(0);delta=mm-avg;cov=np.mean(cc,axis=0)[None]+np.einsum('bnti,bntj->ntij',delta,delta)/BOOT;out['B_boot_mean']=avg;out['B_boot_cov']=cov
  print(case,'bootstrap complete',flush=True)
 save(dest/'bootstrap.json',dict(replicates=boots,label='trajectory-bootstrap moment-matched Gaussian sensitivity, conditional on delay1; not Bayesian posterior'))
 save(dest/'parameters.json',dict(case=case,X_training_mean=mu,theta=theta.tolist(),selected_initialization=k,delay_hypotheses=allpars,delay_trajectory_LL=lls,structure='assumed positive linear Gaussian chain; unit latent innovation gauge; unknown delay',fit_method='stationary Gaussian block composite likelihood BLOCK40, validation exact sequence log-likelihood',fits=fits));np.savez_compressed(dest/'X_only_predictions.npz',**out);files=['parameters.json','AR_models.json','L1_Xonly_head.json','bootstrap.json','X_only_predictions.npz'];save(dest/'LOCAL_X_FREEZE.json',dict(time=time.time(),case=case,no_truth_loaded=True,files={f:sha(dest/f) for f in files}));print(case,'DONE',round(time.time()-begin,2),'s',flush=True);return case

def main():
 jobs=json.loads((ROOT/'public_jobs.json').read_text());done=[]
 with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool:
  for task in as_completed([pool.submit(worker,j) for j in jobs]):done.append(task.result());print('PROGRESS',len(done),'/',len(jobs),flush=True)
 manifest={}
 for j in jobs:
  d=OUT/j['case'];mark=json.loads((d/'LOCAL_X_FREEZE.json').read_text())
  for f in list(mark['files'])+['LOCAL_X_FREEZE.json']:manifest[str((d/f).relative_to(ROOT))]=sha(d/f)
 save(ROOT/'ALL_X_ONLY_FROZEN.json',dict(time=time.time(),cases=len(done),files=manifest,truth_loaded=False));print('ALL X-ONLY OUTPUTS FROZEN',flush=True)
if __name__=='__main__':main()

"""Reconstruct public X + sealed answers and exact frozen predictions without refitting.
Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 to reproduce original arithmetic.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
from settings import ROOT,DATA,OUT,TRAIN,TEST,DELAYS
from private_generator import one
from model import plant,infer
from encoder import encode,apply

def predictions(x,case):
 d=OUT/case;p=json.loads((d/'parameters.json').read_text());theta=np.array(p['theta']);mu=p['X_training_mean'];te=(x-mu)[-TEST:];out={};base=infer(te,plant(theta,1))
 for delay in DELAYS:
  v=infer(te,plant(theta,delay));out[f'B_d{delay}_mean']=v['mean'][:,:,:2];out[f'B_d{delay}_cov']=v['cov'][:,:2,:2]
 out['B_forecast']=base['forecast']+mu;out['B_forecast_var']=base['forecast_var'];out['B_d1_diag_mean']=out['B_d1_mean'];out['B_d1_diag_cov']=np.array([np.diag(np.diag(v)) for v in out['B_d1_cov']]);A,Q,R,P=plant(theta,1);gain=P[:2,2]/(P[2,2]+R);out['B_current_mean']=te[:,:,None]*gain;out['B_current_cov']=P[:2,:2]-np.outer(P[:2,2],P[2,:2])/(P[2,2]+R)
 seed=int(case.rsplit('_',1)[1]);f=encode(x,seed)[-TEST:];h=json.loads((d/'L1_Xonly_head.json').read_text());out['B_L1_mean']=apply(f,h);out['B_L1_cov']=np.array(h['covariance']);ar=json.loads((d/'AR_models.json').read_text());m=ar['models'][ar['selected']];out['A_AR_forecast']=lfilter(np.r_[0,m['coef']],[1],te,axis=1)+mu;out['A_AR_forecast_var']=np.array(m['variance']);boot=json.loads((d/'bootstrap.json').read_text())['replicates']
 if boot:
  mm=[];cc=[]
  for b in boot:
   v=infer(te,plant(b['theta'],1));mm.append(v['mean'][:,:,:2]);cc.append(v['cov'][:,:2,:2])
  mm=np.array(mm);avg=mm.mean(0);delta=mm-avg;out['B_boot_mean']=avg;out['B_boot_cov']=np.mean(cc,axis=0)[None]+np.einsum('bnti,bntj->ntij',delta,delta)/len(mm)
 return out

def replay(case,target):
 if target.exists():raise ValueError('Choose a fresh output directory to preserve existing results')
 meta=next(v for v in json.loads((DATA/'sealed_manifest.json').read_text()) if v['case']==case);x,s,g=one(meta,meta['seed']);target.mkdir(parents=True);np.save(target/'X.npy',x);np.savez_compressed(target/'sealed_states.npz',states=s);out=predictions(x,case);np.savez_compressed(target/'X_only_predictions.npz',**out);wanted=json.loads((OUT/case/'LOCAL_X_FREEZE.json').read_text())['files']['X_only_predictions.npz'];actual=hashlib.sha256((target/'X_only_predictions.npz').read_bytes()).hexdigest();ok=actual==wanted
 (target/'REPLAY_VERIFICATION.json').write_text(json.dumps(dict(case=case,prediction_file_bitwise_matches_freeze=ok,actual_sha256=actual,expected_sha256=wanted,refitted=False),indent=2));assert ok,'Replay differs: check package versions, BLAS threads, architecture';print(case,'bitwise frozen prediction replay PASS',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--out',required=True);a=p.parse_args();replay(a.case,Path(a.out))

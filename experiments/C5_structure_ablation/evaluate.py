import os
os.environ['MPLCONFIGDIR']='/workspace/work/mpl';os.environ['XDG_CACHE_HOME']='/workspace/work/cache'
import csv,gzip,json,hashlib,time,numpy as np
from config import *
from generate import normal,event
from detectors import scores,alarms,predict
from calibrate import load

def writecsv(path,rows):
 with gzip.open(path,'wt',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 assert json.loads((OUT/'PROTOCOL.json').read_text())==PROTOCOL;(OUT/'TEST_START.json').write_text(json.dumps({'protocol_sha256':hashlib.sha256((OUT/'PROTOCOL.json').read_bytes()).hexdigest(),'protocol_markdown_sha256':hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest(),'models_frozen_before_test':True},indent=2));cal=json.loads((OUT/'calibration.json').read_text());rows=[];null=[];examples={};begin=time.time()
 for gi,sigma in enumerate(SIGMAS):
  base,fixed,old=load(gi);cc=cal[str(sigma)];th=cc['thresholds'];x=normal(N_NULL,sigma,SEEDS['null_test'][gi]);sc=scores(x,base,fixed,old['scales']);preds={o:predict(x,sc,base,o,old['drift_window']) for o in [12,15]}
  for model in MODELS:
   times,idx=alarms(sc[model],th[model],2)
   for i,tt in enumerate(times):
    r=dict(sigma=sigma,model=model,trajectory=i,alarm=int(np.isfinite(tt)),alarm_time=tt)
    for o in [12,15]:r[f'forecast_mse_at{o}']=float((preds[o][model][i]-x[i,round((o+2)/DT)])**2)
    null.append(r)
  print('null',sigma,'done',round(time.time()-begin,1),flush=True);del x,sc,preds
  for cohort in COHORTS:
   for ti,tau in enumerate(TAUS):
    for strength in STRENGTHS:
     data=event(gi,tau,strength,cohort);x=data['x'];sc=scores(x,base,fixed,old['scales']);tf=alarms(sc['FAIL'],cc['failure_threshold'],2)[0];tx=alarms(sc['X'],cc['X_threshold'],2)[0];preds={o:predict(x,sc,base,o,old['drift_window']) for o in [12,15]}
     assert np.array_equal(sc['M3'],sc['Ablation'])
     for model in MODELS:
      tt,idx=alarms(sc[model],th[model],2)
      for i,t in enumerate(tt):
       valid=np.isfinite(t) and t>=10;vf=np.isfinite(tf[i]) and tf[i]>=10;vl=valid and vf;param=model!='CUSUM';on=sc['on_'+model][i,idx[i]] if valid and param else np.nan;shape=sc['shape_'+model][i,idx[i]] if valid and param else np.nan;lead=tf[i]-t if vl else np.nan
       r=dict(sigma=sigma,tau=tau,strength=strength,cohort=cohort,trajectory=i,model=model,alarm=t,failure=tf[i],X_alarm=tx[i],valid_alarm=int(valid),early_alarm=int(np.isfinite(t) and t<10),missing_alarm=int(not np.isfinite(t)),delay=t-10 if valid else np.nan,capped_delay=t-10 if valid else 15.,lead_failure=lead,failure_observed=int(vf),onset_estimate=on,onset_error=abs(on-10) if np.isfinite(on) else np.nan,shape_estimate=shape,tau_error=abs(shape-tau) if valid and model in ['M3','M2','Ablation'] else np.nan,generic_mean_lag_error=abs(shape-tau) if valid and model=='GLR_Ramp' else np.nan)
       for m in MARGINS:r[f'coverage_{m}']=int(vl and lead>=m-1e-12)
       for o in [12,15]:r[f'forecast_mse_at{o}']=float((preds[o][model][i]-x[i,round((o+2)/DT)])**2)
       rows.append(r)
     if cohort==0 and strength==1:
      key=f'n{gi}_t{ti}_';examples[key+'x']=x[0]
      for model in MODELS:examples[key+model]=sc[model][0];examples[key+model+'_threshold']=th[model]
   print('events',sigma,'cohort',cohort,'done',round(time.time()-begin,1),flush=True)
 writecsv(OUT/'per_trajectory.csv.gz',rows);writecsv(OUT/'null_trajectories.csv.gz',null);np.savez_compressed(OUT/'examples.npz',**examples)
if __name__=='__main__':main()

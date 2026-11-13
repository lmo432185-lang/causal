import os
os.environ['MPLCONFIGDIR']='/workspace/work/mpl';os.environ['XDG_CACHE_HOME']='/workspace/work/cache'
import json,csv,gzip,time,numpy as np
from config import *
from generate import null_data,event_data
from detectors import all_scores,alarms,forecast_from_scores,local_forecast
from calibrate import load

def writecsv(path,rows):
 with gzip.open(path,'wt',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 saved_protocol=json.loads((OUT/'PROTOCOL.json').read_text());assert saved_protocol==PROTOCOL
 import hashlib
 (OUT/'TEST_START.json').write_text(json.dumps({'protocol_sha256':hashlib.sha256((OUT/'PROTOCOL.json').read_bytes()).hexdigest(),'event_test_started_after_final_calibration':True,'seed_map':SEEDS},indent=2))
 c0=json.loads((OUT/'calibration_C0.json').read_text());cal=json.loads((OUT/'calibration.json').read_text());rows=[];null=[];examples={};start=time.time()
 for gi,sigma in enumerate(SIGMAS):
  base,fixed,p=load(gi,c0);cc=cal[str(sigma)];scales=cc['scales'];th=cc['thresholds'];window=cc['drift_window'];x=null_data(N_NULL,sigma,SEEDS['null_test'][gi]);sc=all_scores(x,base,p,fixed,scales)
  nullpred={}
  for origin in [12,15]:
   ff=forecast_from_scores(x,sc,base,p,TAUS[0],origin);ff['CUSUM']=local_forecast(x,base,origin,window);h=round(2/DT);qh=base['q']**h;ff['X']=qh*x[:,round(origin/DT)]+base['c']*(1-qh)/(1-base['q']);ff['FAIL']=np.full(len(x),np.nan)
   for tau0 in TAUS:ff[f'oracle_{tau0}']=forecast_from_scores(x,sc,base,p,tau0,origin)['M2_oracle']
   nullpred[origin]=ff
  for name in ['M1','M2','M3','CUSUM','X','FAIL']+ORACLES:
   tt,idx=alarms(sc[name],th[name],2)
   for i,v in enumerate(tt):
    row=dict(sigma=sigma,score_name=name,trajectory=i,alarm=int(np.isfinite(v)),alarm_time=v)
    for origin in [12,15]:row[f'forecast_mse_at{origin}']=float((nullpred[origin][name][i]-x[i,round((origin+2)/DT)])**2)
    null.append(row)
  print('null noise',sigma,'done',round(time.time()-start,1),flush=True);del x,sc
  for cohort in COHORTS:
   for ti,tau in enumerate(TAUS):
    for si,strength in enumerate(STRENGTHS):
     data=event_data(sigma,tau,strength,gi,cohort);x=data['x'];sc=all_scores(x,base,p,fixed,scales);tf=alarms(sc['FAIL'],th['FAIL'],2)[0];tx=alarms(sc['X'],th['X'],2)[0];estimates={}
     for origin in [12,15]:
      fp=forecast_from_scores(x,sc,base,p,tau,origin);fp['CUSUM']=local_forecast(x,base,origin,window);h=round(2/DT);qh=base['q']**h;fp['X']=qh*x[:,round(origin/DT)]+base['c']*(1-qh)/(1-base['q']);estimates[origin]=fp
     for method in METHODS:
      key=f'oracle_{tau}' if method=='M2_oracle' else method;alarm,idx=alarms(sc[key],th[key],2);valid=np.isfinite(alarm)&(alarm>=10);vf=np.isfinite(tf)&(tf>=10);vx=np.isfinite(tx)&(tx>=10)
      for i in range(N_EVENT):
       isstruct=method in ['M2','M2_oracle','M3'];on=sc[f'on_{key}'][i,idx[i]] if valid[i] and isstruct else np.nan;te=sc[f'tau_{key}'][i,idx[i]] if valid[i] and isstruct else np.nan
       row=dict(sigma=sigma,tau=tau,strength=strength,cohort=cohort,trajectory=i,method=method,alarm=alarm[i],failure=tf[i],X_alarm=tx[i],valid_alarm=int(valid[i]),early_alarm=int(np.isfinite(alarm[i]) and alarm[i]<10),missing_alarm=int(not np.isfinite(alarm[i])),delay=alarm[i]-10 if valid[i] else np.nan,capped_delay=alarm[i]-10 if valid[i] else 15.,lead_failure=tf[i]-alarm[i] if valid[i] and vf[i] else np.nan,lead_X=tx[i]-alarm[i] if valid[i] and vx[i] else np.nan,coverage_failure=int(valid[i] and vf[i] and alarm[i]<tf[i]),coverage_X=int(valid[i] and vx[i] and alarm[i]<tx[i]),failure_observed=int(vf[i]),onset_estimate=on,onset_error=abs(on-10) if np.isfinite(on) else np.nan,tau_estimate=te,tau_error=abs(te-tau) if np.isfinite(te) and method!='M2_oracle' else np.nan,oracle_parameter_given=int(method=='M2_oracle'))
       for origin in [12,15]:row[f'forecast_mse_at{origin}']=float((estimates[origin][method][i]-x[i,round((origin+2)/DT)])**2)
       rows.append(row)
     if cohort==0 and strength==1:
      prefix=f'n{gi}_t{ti}_';examples[prefix+'x']=x[0]
      for name in ['M1','M2','M3','CUSUM','X','FAIL',f'oracle_{tau}']:examples[prefix+name]=sc[name][0];examples[prefix+name+'_threshold']=th[name]
   print('events noise',sigma,'cohort',cohort,'done',round(time.time()-start,1),flush=True)
 writecsv(OUT/'per_trajectory.csv.gz',rows);writecsv(OUT/'null_trajectories.csv.gz',null);np.savez_compressed(OUT/'examples.npz',**examples)
if __name__=='__main__':main()

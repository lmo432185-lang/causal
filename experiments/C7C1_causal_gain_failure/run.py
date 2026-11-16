import json,hashlib,time,multiprocessing
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from config import *
from physics import simulate
from train import background,fit_plant,fit_scales
from detectors import scores,maxima,calibrate,alarm

def log(x):print(datetime.now(timezone.utc).isoformat(),x,flush=True)
def dump(p,x):p.write_text(json.dumps(x,indent=2))

def normal_cell(task):
 ni,ki,pi,model=task;policy=POLICIES[pi];scales=fit_scales(ni,ki,policy,model);stored={};th={};audit={}
 for split,nn in [('calibration',N_CAL),('null',N_NULL)]:
  chunks={m:[] for m in METHODS+['GainSpecific']}
  for batch in range(nn//100):
   # Shared noise/phase/normal-beta across policies; no hidden-failure labels used.
   seed=seed_for(split,ni,ki,batch=batch);d=simulate(100,SIGMAS[ni],KTEST[ki],policy,seed);sc,_=scores(d['x'],d['I'],model,scales)
   for method in chunks:chunks[method].append(maxima(sc[method]))
  vv={m:np.concatenate(v) for m,v in chunks.items()}
  if split=='calibration':
   for budget in BUDGETS:
    th[str(budget)]={m:calibrate(v,budget) for m,v in vv.items()};th[str(budget)]['GainSpecific']=max(th[str(budget)]['GainSpecific'],th[str(budget)]['GainGLR'])
   for m,v in vv.items():stored['cal_'+m]=v
  else:
   for m,v in vv.items():stored['null_'+m]=v
   for budget in BUDGETS:
    audit[str(budget)]={m:dict(n=nn,count=int(np.sum(v>th[str(budget)][m])),fpr=float(np.mean(v>th[str(budget)][m]))) for m,v in vv.items()}
 np.savez_compressed(OUT/f'normal_n{ni}_k{ki}_p{pi}.npz',**stored)
 return f'n{ni}_k{ki}_p{pi}',dict(model=model,scales=scales,thresholds=th,audit=audit)

def prepare():
 backgrounds=[background(ni) for ni in range(3)];models={};tasks=[]
 for ni in range(3):
  for ki in range(4):
   model=fit_plant(ni,ki,backgrounds[ni]);models[f'n{ni}_k{ki}']=model
   for pi in range(3):tasks.append((ni,ki,pi,model))
 settings={};log('Healthy plant learning complete; normal calibration begins')
 with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
  for name,ss in pool.map(normal_cell,tasks):
   settings[name]=ss;dump(OUT/'settings.json',settings);log(f'Normal calibration/audit complete {name}')
 dump(OUT/'backgrounds.json',backgrounds);return settings

def test_cell(task):
 ni,ki,co,settings=task;seed=seed_for('test',ni,ki,co=co);rows=[];archives={}
 for pi,policy in enumerate(POLICIES):
  ss=settings[f'n{ni}_k{ki}_p{pi}'];model=ss['model']
  for mode in PRIMARY_MODES+SECONDARY_MODES:
   d=simulate(N,SIGMAS[ni],KTEST[ki],policy,seed,mode);sc,meta=scores(d['x'],d['I'],model,ss['scales']);tf=d['failure_time'];rho=1. if mode in PRIMARY_MODES else .5 if mode=='partial' else 0.
   for budget in BUDGETS:
    thresholds=ss['thresholds'][str(budget)];times={};indexes={}
    for m in sc:times[m],indexes[m]=alarm(sc[m],thresholds[m])
    reference=times['PredictionError'];refvalid=np.isfinite(reference)&(reference>=tf)
    for m,at in times.items():
     valid=np.isfinite(at)&(at>=tf);early=np.isfinite(at)&(at<tf);lead=np.where(valid&refvalid,reference-at,np.nan)
     for i in range(N):
      idx=indexes[m][i];is_gain=m in ['GainGLR','GainSpecific'];beta_est=meta['beta_estimate'][i,idx] if is_gain and idx>=0 else np.nan
      row=dict(mode=mode,noise=SIGMAS[ni],plant=ki,tau_truth=KTEST[ki],cohort=co,trajectory=i,policy=policy,method=m,budget=budget,T_beta=tf[i],T_alarm=at[i],T_prediction=reference[i],early_alarm=int(early[i]),recall=int(valid[i]),no_alarm=int(not np.isfinite(at[i])),delay=float(at[i]-tf[i]) if valid[i] else np.nan,capped_delay=float(at[i]-tf[i]) if valid[i] else END-tf[i],lead=lead[i],reference_valid=int(refvalid[i]),post_failure_beta_estimate=beta_est,input_energy=float(np.sum(d['I'][i]**2)*DT),input_phase=d['probe_phase'][i],claimed_gain_onset=meta['onset'][i,idx] if is_gain and idx>=0 else np.nan,claimed_gain_duration=meta['duration'][i,idx] if is_gain and idx>=0 else np.nan)
      for margin in [.5,1.,2.]:row[f'cover_{margin}']=int(valid[i] and refvalid[i] and reference[i]-at[i]>=margin)
      rows.append(row)
   # Keep all scores for primary scenarios to replay any false-alarm operating point.
   if mode in PRIMARY_MODES:
    for m,v in sc.items():archives[f'{mode}_{policy}_{m}']=v.astype(np.float32)
    archives[f'{mode}_{policy}_beta_estimate']=meta['beta_estimate'].astype(np.float32)
    for key in ['x','I','U','beta','failure_time','probe_phase']:archives[f'{mode}_{policy}_example_'+key]=d[key][:10]
 np.savez_compressed(OUT/f'test_scores_n{ni}_k{ki}_c{co}.npz',**archives);return rows

def test(settings):
 files=['config.py','physics.py','detectors.py','train.py','run.py','PROTOCOL.json','PROTOCOL_REGISTERED.json','legacy_stream.py','source_C0.py','source_A.py','support.py','results/settings.json'];dump(OUT/'TEST_START.json',dict(time=datetime.now(timezone.utc).isoformat(),sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}));tasks=[(ni,ki,co,settings) for ni in range(3) for ki in range(4) for co in COHORTS];allrows=[]
 with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
  for task,result in zip(tasks,pool.map(test_cell,tasks)):
   allrows.extend(result);log(f'Fault tests complete n{task[0]} k{task[1]} c{task[2]}')
 pd.DataFrame(allrows).to_csv(OUT/'trajectory_metrics.csv.gz',index=False)

if __name__=='__main__':
 begin=time.time();settings=prepare();test(settings);log(f'ALL COMPLETE {time.time()-begin:.1f}s')

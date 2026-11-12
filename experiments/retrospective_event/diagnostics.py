import csv,json,numpy as np
from run import OUT,CFG,SEEDS,generate,infer,first_run,stats,DT
rows=list(csv.DictReader((OUT/'per_trajectory.csv').open())); snaps=list(csv.DictReader((OUT/'snapshots.csv').open())); checks=json.loads((OUT/'checks.json').read_text()); result={}
for sigma in CFG['noise']:
 vals=[]; successful=[]
 for j in range(200):
  dif=[]
  for seed in SEEDS:
   r=next(r for r in snaps if float(r['sigma'])==sigma and int(r['seed'])==seed and int(r['trajectory'])==j and float(r['additional_seconds'])==2)
   if int(r['estimate_available']) and np.isfinite(float(r['alert_delay'])):dif.append(abs(float(r['alert_delay']))-float(r['backdated_error']))
  vals.append(np.mean(dif) if dif else np.nan); successful.append(len(dif)/3)
 result[str(sigma)]=dict(paired_alert_absolute_delay_minus_backdated_error=stats(vals),common_availability=stats(successful))
# External physical transport delay challenge, not present in training. The estimator
# sees its effect, but has no way to know the unobserved pre-effect action time.
sigma=.35; original=generate(200,sigma,7001,'event'); effective=np.zeros_like(original['y']); delay=10;effective[:,delay:]=original['y'][:,:-delay];rng=np.random.default_rng(7002)
x=np.zeros_like(original['x']);x[:,0]=rng.normal(10,.3,200)
for t in range(500):x[:,t+1]=x[:,t]+DT*(1.5-.15*x[:,t]-1.5*effective[:,t])+sigma*np.sqrt(DT)*rng.normal(size=200)
physical=[]
for seed in SEEDS:
 z=np.load(OUT/f'model_1_{seed}.npz');p={k:z[k] for k in ['beta','mu','var','A','pi']};o,_=infer(x,p);thr=checks[f'{sigma}_{seed}']['threshold']
 for j in range(200):
  start=first_run(o[500][j]>=thr)
  physical.append(dict(seed=seed,trajectory=j,detected=int(start is not None),true_action_error=np.nan if start is None else (start-original['on'][j])*DT,effect_onset_error=np.nan if start is None else (start-original['on'][j]-delay)*DT))
result['physical_delay_0.5_seconds']={}
for key in ['detected','true_action_error','effect_onset_error']:
 vals=[]
 for j in range(200):
  a=[r[key] for r in physical if r['trajectory']==j];vals.append(np.nanmean(a) if np.isfinite(a).any() else np.nan)
 result['physical_delay_0.5_seconds'][key]=stats(vals)
# Identical observed process, two distinct causes: impossible to tell with X alone.
# Brake Y and slope force of the same strength produce exactly the same acceleration.
base=generate(200,.35,8001,'event');x_brake=base['x'];x_slope=base['x'].copy(); p=np.load(OUT/'model_1_11.npz');p={k:p[k] for k in ['beta','mu','var','A','pi']};a,_=infer(x_brake,p);b,_=infer(x_slope,p)
result['cause_ambiguity']={'max_X_difference':float(np.max(abs(x_brake-x_slope))),'max_posterior_difference':float(np.max(abs(a[500]-b[500]))),'interpretation':'paired brake and uphill mechanisms generate identical X under same net force and noise; X-only cause identity is unidentifiable'}
(OUT/'diagnostics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

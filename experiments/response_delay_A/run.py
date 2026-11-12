"""Experiment A only. Frozen earlier X-only HMM, strictly forward test inference."""
import os
os.environ['MPLCONFIGDIR']='/workspace/work/mpl';os.environ['XDG_CACHE_HOME']='/workspace/work/cache'
import json,csv,time
from pathlib import Path
import numpy as np
from scipy.special import expit,logsumexp
from support import fit,generate as old_generate,first_run
ROOT=Path(__file__).parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
DT=.05;NSTEP=500;T1=10.;SIGMAS=[.12,.35,.7];TAUS=[.1,.5,1.,2.];STRENGTHS=[.5,1.,1.5];SEEDS=[11,22,33]
CFG=dict(experiment='A: causal online detection only',dt=DT,duration=25,T1=T1,tau=TAUS,noise=SIGMAS,strength=STRENGTHS,n_event_per_cell=200,n_null_test_per_noise=500,n_null_calibration_per_noise=500,n_null_AR_training_per_noise=200,model_seeds=SEEDS,model='frozen two-state switching AR HMM trained in preceding retrospective experiment',target_trajectory_false_alarm=.05,persistence_samples=2,response_equations='dX=(1.5-.15 X-1.5 U)dt+sigma dW; dU=(Y-U)/tau dt',T2_definition='first 2 consecutive significant downward X deviations from learned no-event trajectory mean; threshold calibrated on whole null trajectories',T3_definition='first 2 consecutive significant negative-regime forward HMM posterior scores; threshold calibrated on same null trajectories',integration='exact joint deterministic step for X,U with exact OU process-noise variance; no Euler approximation',censoring='first alarms before T1 counted separately; no restart at true T1; non-early non-detected cells retained as failures; delay conditional on valid first alarm, capped delay assigns 15 seconds to every early/missing trial')

def simulate(n,sigma,tau,strength,seed,event=True):
 rng=np.random.default_rng(seed);x=np.empty((n,NSTEP+1));x[:,0]=rng.normal(10,.3,n);u=np.zeros_like(x); y=np.zeros((n,NSTEP));
 if event:y[:,int(T1/DT):]=strength
 a=.15;b=1.5;f=1.5;q=np.exp(-a*DT);r=np.exp(-DT/tau);j=(r-q)/(a-1/tau);k=(1-q)/a;noise_sd=sigma*np.sqrt((1-q*q)/(2*a))
 for t in range(NSTEP):
  force=y[:,t]*k+(u[:,t]-y[:,t])*j
  x[:,t+1]=q*x[:,t]+f*k-b*force+noise_sd*rng.normal(size=n)
  u[:,t+1]=y[:,t]+(u[:,t]-y[:,t])*r
 return dict(x=x,u=u,y=y)

def forward_score(x,p):
 # Every output at index t uses only X[0:t+2], i.e. data time (t+1)*dt.
 n,t=x.shape[0],x.shape[1]-1;prior=np.broadcast_to(p['pi'],(n,2)).copy();res=np.empty((n,t))
 for j in range(t):
  dx=x[:,j+1]-x[:,j];ll=-.5*((dx[:,None]-p['beta']*x[:,j,None]-p['mu'])**2/p['var']+np.log(2*np.pi*p['var']));lp=np.log(np.maximum(prior,1e-300))+ll;lp-=logsumexp(lp,axis=1)[:,None];post=np.exp(lp);res[:,j]=lp[:,0]-lp[:,1];prior=post@p['A']
 return res

def fit_normal(x):
 xx=x[:,:-1].ravel();target=x[:,1:].ravel();co=np.linalg.lstsq(np.c_[xx,np.ones(len(xx))],target,rcond=None)[0];var=np.mean((target-(co[0]*xx+co[1]))**2);return dict(q=float(co[0]),c=float(co[1]),var=float(var))
def observed_score(x,p):
 mean=x[:,0].copy();var=0.;scores=np.zeros((len(x),x.shape[1]-1))
 for j in range(scores.shape[1]):
  mean=p['q']*mean+p['c'];var=p['q']**2*var+p['var'];scores[:,j]=(mean-x[:,j+1])/np.sqrt(var)
 return scores

def threshold(scores):
 # max of pairwise minima is exactly the score of first 2-consecutive crossing.
 maxima=np.minimum(scores[:,:-1],scores[:,1:]).max(1);return float(np.quantile(maxima,.95,method='higher'))
def alarm_times(scores,thr):
 result=np.full(len(scores),np.nan)
 for i,v in enumerate(scores):
  start=first_run(v>thr)
  if start is not None:result[i]=(start+2)*DT
 return result

def statistics(values):
 v=np.asarray(values,float);v=v[np.isfinite(v)];n=len(v)
 if not n:return dict(n=0,mean=None,sd=None,median=None,ci95=None)
 se=v.std(ddof=1)/np.sqrt(n) if n>1 else 0;return dict(n=n,mean=float(v.mean()),sd=float(v.std(ddof=1)) if n>1 else 0.,median=float(np.median(v)),ci95=[float(v.mean()-1.972*se),float(v.mean()+1.972*se)])
def fields(times):
 early=np.isfinite(times)&(times<T1);valid=np.isfinite(times)&(times>=T1);delay=np.where(valid,times-T1,np.nan)
 return early,valid,delay

def main():
 (OUT/'config.json').write_text(json.dumps(CFG,indent=2));rows=[];nullrows=[];examples={};checks={};cal={};start=time.time()
 for gi,sigma in enumerate(SIGMAS):
  normal_train=simulate(200,sigma,1,0,10000+gi,False);normal_val=simulate(500,sigma,1,0,11000+gi,False);normal_test=simulate(500,sigma,1,0,12000+gi,False);baseline=fit_normal(normal_train['x']);thr2=threshold(observed_score(normal_val['x'],baseline));t2null=alarm_times(observed_score(normal_test['x'],baseline),thr2)
  cal[str(sigma)]=dict(normal_AR=baseline,T2_threshold=thr2,HMM_thresholds={})
  for seed in SEEDS:
   ck=np.load(OUT/f'model_{gi}_{seed}.npz');p={k:ck[k] for k in ['beta','mu','var','A','pi']};thr3=threshold(forward_score(normal_val['x'],p));cal[str(sigma)]['HMM_thresholds'][str(seed)]=thr3;t3null=alarm_times(forward_score(normal_test['x'],p),thr3)
   for i in range(500):nullrows.append(dict(sigma=sigma,seed=seed,trajectory=i,T2=t2null[i],T3=t3null[i],T2_alarm=int(np.isfinite(t2null[i])),T3_alarm=int(np.isfinite(t3null[i]))))
   for ti,tau in enumerate(TAUS):
    for si,strength in enumerate(STRENGTHS):
     # Same innovations across tau/strength enable paired comparisons, no test selection.
     data=simulate(200,sigma,tau,strength,13000+gi);s2=observed_score(data['x'],baseline);s3=forward_score(data['x'],p);t2=alarm_times(s2,thr2);t3=alarm_times(s3,thr3);e2,v2,d2=fields(t2);e3,v3,d3=fields(t3)
     for i in range(200):
      rows.append(dict(sigma=sigma,tau=tau,strength=strength,seed=seed,trajectory=i,T1=T1,U50_time=T1+tau*np.log(2),T2=t2[i],T3=t3[i],T2_early=int(e2[i]),T3_early=int(e3[i]),T2_valid=int(v2[i]),T3_valid=int(v3[i]),T2_missing=int(not np.isfinite(t2[i])),T3_missing=int(not np.isfinite(t3[i])),T2_delay=d2[i],T3_delay=d3[i],T2_capped_delay=d2[i] if v2[i] else 15.,T3_capped_delay=d3[i] if v3[i] else 15.,paired_T2_minus_T3=t2[i]-t3[i] if v2[i] and v3[i] else np.nan))
     if seed==11:
      key=f'n{gi}_t{ti}_s{si}';examples.update({key+'_x':data['x'][0],key+'_u':data['u'][0],key+'_y':data['y'][0],key+'_score2':s2[0],key+'_score3':s3[0],key+'_threshold2':thr2,key+'_threshold3':thr3,key+'_T2':t2[0],key+'_T3':t3[0]})
   print('noise',sigma,'seed',seed,'done',round(time.time()-start,1),flush=True)
  # Reproduce previous EM model using exactly its old training X and seed 11.
  retrain,trace=fit(old_generate(200,sigma,1000+gi)['x'],11);old=np.load(OUT/f'model_{gi}_11.npz');err=max(float(np.max(abs(np.asarray(retrain[k])-old[k]))) for k in retrain);checks[f'reproduce_previous_noise_{sigma}']=dict(max_parameter_difference=err,passed=bool(err<1e-8))
 with (OUT/'per_trajectory.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with (OUT/'null_trajectories.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(nullrows[0]));w.writeheader();w.writerows(nullrows)
 (OUT/'calibration.json').write_text(json.dumps(cal,indent=2));(OUT/'reproduction.json').write_text(json.dumps(checks,indent=2));np.savez_compressed(OUT/'examples.npz',**examples);summarize(rows,nullrows)

def summarize(rows,nullrows):
 summary=[];nullsummary={}
 for sigma in SIGMAS:
  sub=[r for r in nullrows if r['sigma']==sigma];nullsummary[str(sigma)]={k:statistics([np.mean([r[k] for r in sub if r['trajectory']==i]) for i in range(500)]) for k in ['T2_alarm','T3_alarm']}
  for tau in TAUS:
   for strength in STRENGTHS:
    sub=[r for r in rows if r['sigma']==sigma and r['tau']==tau and r['strength']==strength];item=dict(sigma=sigma,tau=tau,strength=strength,T1=T1,U50_time=T1+tau*np.log(2))
    for k in ['T2','T3','T2_early','T3_early','T2_valid','T3_valid','T2_missing','T3_missing','T2_delay','T3_delay','T2_capped_delay','T3_capped_delay','paired_T2_minus_T3']:
     vals=[]
     for i in range(200):
      a=np.array([r[k] for r in sub if r['trajectory']==i]);vals.append(np.nanmean(a) if np.isfinite(a).any() else np.nan)
     item[k]=statistics(vals)
    # Explicit paired tau effect from same test innovations.
    tau0=[r for r in rows if r['sigma']==sigma and r['tau']==.1 and r['strength']==strength];diff=[]
    for i in range(200):diff.append(np.mean([r['T3_capped_delay'] for r in sub if r['trajectory']==i])-np.mean([r['T3_capped_delay'] for r in tau0 if r['trajectory']==i]))
    item['paired_T3_capped_delay_vs_tau01']=statistics(diff);summary.append(item)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));(OUT/'null_summary.json').write_text(json.dumps(nullsummary,indent=2))
 for r in summary:
  if r['strength']==1:print({k:r[k] if k in ['sigma','tau'] else r[k]['mean'] for k in ['sigma','tau','T2_delay','T3_delay','T3_valid','T3_early','paired_T2_minus_T3']},flush=True)
if __name__=='__main__':main()

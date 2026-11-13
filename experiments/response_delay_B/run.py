"""Experiment B: paired retrospective inference on exactly experiment A's X."""
import os
os.environ['MPLCONFIGDIR']='/workspace/work/mpl';os.environ['XDG_CACHE_HOME']='/workspace/work/cache'
import csv,json,time,hashlib,gzip
from pathlib import Path
import numpy as np
from scipy.special import expit
from source_A import simulate,forward_score,alarm_times,statistics,DT,SIGMAS,TAUS,STRENGTHS,SEEDS
from support import emission,first_run
ROOT=Path(__file__).parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
HORIZONS=[0,.5,1,2,3,5];N=500
CONFIG=dict(experiment='B only; same A trajectories, frozen HMM, thresholds and first alarms',noise=SIGMAS,tau=TAUS,strength=STRENGTHS,seeds=SEEDS,n_each_cell=200,horizons_after_alarm=HORIZONS,full_reference_end=25,dt=DT,onset_rule='first two consecutive posterior log odds above frozen A threshold, restricted to time already past at A alarm',state_evaluation='same last 5 time units preceding each A alarm, for every future horizon',ACI='KL(HMM smoothed state at endpoint || HMM prior conditioned on X up to state time)',CIR_proxy='trapezoidal integral 0..5 of KL(full25 smoothing || finite-future smoothing)/KL(full25 smoothing || filter); grid 0,.5,1,2,3,5; discard denominator<1e-4',stability='earliest tested extra horizon after which all tested onset estimates within .1 of full25 reference and mean absolute posterior difference<=.02 on same past window',no_Y_U_tau_input_to_inference=True)

def filtering(x,p):
 e,_=emission(x,p);n,t,_=e.shape;prior=np.empty_like(e);post=np.empty_like(e);prior[:,0]=p['pi']
 for j in range(t):
  v=prior[:,j]*e[:,j];post[:,j]=v/v.sum(1)[:,None]
  if j<t-1:prior[:,j+1]=post[:,j]@p['A']
 return e,prior,post

def smooth(e,prior,post,A,ends):
 n,t,_=e.shape;beta=np.ones((n,2));out=np.full((n,t),np.nan)
 for j in range(t-1,-1,-1):
  active=j<ends;v=post[:,j]*beta;out[active,j]=(v/v.sum(1)[:,None])[active,0]
  v=(e[:,j]*beta)@A.T;v/=v.sum(1)[:,None];beta=np.where(active[:,None],v,1.)
 return out

def kl(p,q):
 p=np.clip(p,1e-12,1-1e-12);q=np.clip(q,1e-12,1-1e-12)
 return np.maximum(p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q)),0)
def onset(probs,threshold,end):
 z=first_run(probs[:end]>threshold);return np.nan if z is None else z*DT

def finite_lag(e,prior,A,targets,lag):
 n,w=targets.shape;ii=np.arange(n)[:,None];beta=np.ones((n,w,2));valid=targets>=0;idx=np.clip(targets,0,e.shape[1]-1)
 if lag==0:return np.where(valid,prior[ii,idx,0],np.nan)
 for offset in range(lag-1,-1,-1):
  tt=targets+offset;em=e[ii,np.clip(tt,0,e.shape[1]-1)];em=np.where(((tt>=0)&(tt<e.shape[1]))[:,:,None],em,1.);v=em*beta;v/=v.sum(2)[:,:,None]
  if offset>0:beta=v@A.T
 out=prior[ii,idx]*v;out/=out.sum(2)[:,:,None];return np.where(valid,out[:,:,0],np.nan)

def describe(group):
 # Independent unit: trajectory, not model seed.
 out={};keys=[k for k in group[0] if k not in ['sigma','tau','strength','seed','trajectory','horizon']]
 for key in keys:
  vals=[]
  for i in range(200):
   a=np.array([r[key] for r in group if r['trajectory']==i],float);vals.append(np.nanmean(a) if np.isfinite(a).any() else np.nan)
  out[key]=statistics(vals)
 return out

def evaluate(data,p,thr,alarms,meta,do_info=True):
 e,prior,post=filtering(data['x'],p);size=len(alarms);good=np.isfinite(alarms);ends=np.where(good,np.rint(np.nan_to_num(alarms)/DT),0).astype(int);validA=good&(alarms>=10);probthr=expit(thr);reference=smooth(e,prior,post,p['A'],np.full(size,500));basepast=np.arange(500)[None,:]<ends[:,None];window=basepast&(np.arange(500)[None,:]>=ends[:,None]-100);true=data['y']>0;u_fraction=np.divide(data['u'][:,:-1],np.maximum(data['y'].max(1)[:,None],1e-12));u_fraction=np.clip(u_fraction,0,1);filtered=prior[:,:,0]
 rstart=np.array([onset(reference[i],probthr,ends[i]) if good[i] else np.nan for i in range(size)])
 snapshots=[];allrows=[];onsets=[];diffs=[];present=[]
 for h in HORIZONS:
  endpoint=np.where(good,np.minimum(500,ends+round(h/DT)),0);sm=smooth(e,prior,post,p['A'],endpoint);snapshots.append(sm);est=np.array([onset(sm[i],probthr,ends[i]) if good[i] else np.nan for i in range(size)]);onsets.append(est);delta=np.full(size,np.nan)
  for i in range(size):
   m=window[i];err=abs(est[i]-10) if np.isfinite(est[i]) else np.nan;diff=float(np.mean(abs(sm[i,m]-reference[i,m]))) if good[i] else np.nan;delta[i]=diff
   usable=validA[i] and np.isfinite(est[i]);refusable=validA[i] and np.isfinite(rstart[i]);row=dict(**meta,trajectory=i,horizon=h,A_alarm=alarms[i],A_valid=int(validA[i]),A_early=int(good[i] and alarms[i]<10),A_missing=int(not good[i]),estimate=est[i],estimate_available=int(usable),onset_error=err if usable else np.nan,onset_signed=est[i]-10 if usable else np.nan,onset_error_capped=err if usable else 15.,actual_extra=(endpoint[i]-ends[i])*DT if good[i] else np.nan,horizon_clipped=int(good[i] and ends[i]+round(h/DT)>500),brier_Y=float(np.mean((sm[i,m]-true[i,m])**2)) if validA[i] else np.nan,brier_Ufraction=float(np.mean((sm[i,m]-u_fraction[i,m])**2)) if validA[i] else np.nan,filter_brier_Y=float(np.mean((filtered[i,m]-true[i,m])**2)) if validA[i] else np.nan,ACI_past_mean=float(kl(sm[i,m],filtered[i,m]).mean()) if validA[i] else np.nan,posterior_change_vs_full=diff if validA[i] else np.nan,full_reference_onset_error=abs(rstart[i]-10) if refusable else np.nan,full_reference_estimate=rstart[i] if good[i] else np.nan,paired_error_gain_vs_alarm=abs(alarms[i]-10)-err if usable else np.nan,paired_brier_gain_vs_filter=float(np.mean((filtered[i,m]-true[i,m])**2-(sm[i,m]-true[i,m])**2)) if validA[i] else np.nan)
   allrows.append(row)
  diffs.append(delta)
 # Within identical paired cohorts, additional-future onset gain vs 0.
 for hi,h in enumerate(HORIZONS):
  for i in range(size):
   row=allrows[hi*size+i];both=validA[i] and np.isfinite(onsets[0][i]) and np.isfinite(onsets[hi][i]);row['paired_onset_gain_vs_h0']=abs(onsets[0][i]-10)-abs(onsets[hi][i]-10) if both else np.nan
   if validA[i]:
    m=window[i];row['paired_brier_gain_vs_h0']=float(np.mean((snapshots[0][i,m]-true[i,m])**2-(snapshots[hi][i,m]-true[i,m])**2))
   else:row['paired_brier_gain_vs_h0']=np.nan
 # Diagnostics at full-reference for same fixed past targets, no true event time.
 targets=ends[:,None]-100+np.arange(100)[None,:];ii=np.arange(size)[:,None];idx=np.clip(targets,0,499);ref=reference[ii,idx];filt=prior[ii,idx,0];den=kl(ref,filt);mask=(targets>=0)&validA[:,None]&(den>=1e-4)
 curve=[]
 if do_info:
  for h in HORIZONS:
   finite=finite_lag(e,prior,p['A'],targets,round(h/DT));curve.append(kl(ref,finite)/np.maximum(den,1e-12))
  cir=np.trapz(np.stack(curve),x=np.array(HORIZONS),axis=0)
  tail=curve[-1]
 else:cir=np.full_like(den,np.nan);tail=cir
 inforows=[]
 for i in range(size):
  stable=np.nan
  if validA[i] and np.isfinite(rstart[i]):
   for hi,h in enumerate(HORIZONS):
    ok=all(np.isfinite(onsets[k][i]) and abs(onsets[k][i]-rstart[i])<=.1+1e-8 and diffs[k][i]<=.02 for k in range(hi,len(HORIZONS)))
    if ok:stable=h;break
  m=mask[i];inforows.append(dict(**meta,trajectory=i,A_valid=int(validA[i]),stability_horizon=stable,stability_found=int(np.isfinite(stable)),full_onset_error=abs(rstart[i]-10) if validA[i] and np.isfinite(rstart[i]) else np.nan,CIR5_mean=float(np.mean(cir[i,m])) if do_info and m.any() else np.nan,CIR5_median=float(np.median(cir[i,m])) if do_info and m.any() else np.nan,CIR5_tail_ratio=float(np.mean(tail[i,m])) if do_info and m.any() else np.nan,full_ACI_mean=float(den[i,targets[i]>=0].mean()) if validA[i] else np.nan))
 example=dict(x=data['x'][0],y=data['y'][0],u=data['u'][0],alarm=alarms[0],reference=reference[0],filter=filtered[0],threshold=probthr,**{f'smooth{h}':v[0] for h,v in zip(HORIZONS,snapshots)})
 return allrows,inforows,example

def writecsv(path,rows):
 if path.name=='per_trajectory.csv':
  f=gzip.open(path.with_suffix('.csv.gz'),'wt',newline='')
 else:f=path.open('w')
 with f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 (OUT/'config.json').write_text(json.dumps(CONFIG,indent=2));cal=json.loads((OUT/'calibration_A.json').read_text());arows=list(csv.DictReader((OUT/'A_per_trajectory.csv').open()));lookup={(float(r['sigma']),float(r['tau']),float(r['strength']),int(r['seed']),int(r['trajectory'])):float(r['T3']) for r in arows};rows=[];infos=[];examples={};null=[];repro=[];start=time.time()
 for gi,sigma in enumerate(SIGMAS):
  for seed in SEEDS:
   ck=np.load(OUT/f'model_{gi}_{seed}.npz');p={k:ck[k] for k in ['beta','mu','var','A','pi']};thr=cal[str(sigma)]['HMM_thresholds'][str(seed)]
   for ti,tau in enumerate(TAUS):
    for si,strength in enumerate(STRENGTHS):
     data=simulate(200,sigma,tau,strength,13000+gi);alarm=alarm_times(forward_score(data['x'],p),thr);original=np.array([lookup[(sigma,tau,strength,seed,i)] for i in range(200)]);np.testing.assert_allclose(alarm,original,atol=1e-12,equal_nan=True)
     repro.append(dict(sigma=sigma,tau=tau,strength=strength,seed=seed,T3_match=True))
     rr,ii,example=evaluate(data,p,thr,alarm,dict(sigma=sigma,tau=tau,strength=strength,seed=seed));rows.extend(rr);infos.extend(ii)
     if seed==11:examples.update({f'n{gi}_t{ti}_s{si}_'+k:v for k,v in example.items()})
   # Same independent null test trajectories as A; evaluate every A false alarm.
   data=simulate(500,sigma,1,0,12000+gi,False);alarm=alarm_times(forward_score(data['x'],p),thr);e,prior,post=filtering(data['x'],p);good=np.isfinite(alarm);end=np.where(good,np.rint(np.nan_to_num(alarm)/DT),0).astype(int)
   for h in HORIZONS:
    sm=smooth(e,prior,post,p['A'],np.where(good,np.minimum(500,end+round(h/DT)),0))
    for i in range(500):null.append(dict(sigma=sigma,seed=seed,trajectory=i,horizon=h,A_alarm=int(good[i]),retrospective_confirmation=int(good[i] and np.isfinite(onset(sm[i],expit(thr),end[i])))))
   print('noise',sigma,'seed',seed,'done',round(time.time()-start,1),flush=True)
 writecsv(OUT/'per_trajectory.csv',rows);writecsv(OUT/'information.csv',infos);writecsv(OUT/'null_trajectories.csv',null);np.savez_compressed(OUT/'examples.npz',**examples);(OUT/'reproduction.json').write_text(json.dumps(repro,indent=2));summarize(rows,infos,null)

def summarize(rows,infos,null):
 result=[];ir=[];nr=[]
 for sigma in SIGMAS:
  for tau in TAUS:
   for strength in STRENGTHS:
    for h in HORIZONS:
     sub=[r for r in rows if r['sigma']==sigma and r['tau']==tau and r['strength']==strength and r['horizon']==h];result.append(dict(sigma=sigma,tau=tau,strength=strength,horizon=h,metrics=describe(sub)))
    sub=[r for r in infos if r['sigma']==sigma and r['tau']==tau and r['strength']==strength];ir.append(dict(sigma=sigma,tau=tau,strength=strength,metrics=describe(sub)))
  for h in HORIZONS:
   sub=[r for r in null if r['sigma']==sigma and r['horizon']==h];vals=[np.mean([r['retrospective_confirmation'] for r in sub if r['trajectory']==i]) for i in range(500)];nr.append(dict(sigma=sigma,horizon=h,confirmation=statistics(vals)))
 (OUT/'summary.json').write_text(json.dumps(result,indent=2));(OUT/'information_summary.json').write_text(json.dumps(ir,indent=2));(OUT/'null_summary.json').write_text(json.dumps(nr,indent=2))
 for r in result:
  if r['sigma']==.35 and r['strength']==1:print({k:r[k] for k in ['tau','horizon']}, {k:r['metrics'][k]['mean'] for k in ['onset_error','onset_signed','estimate_available','brier_Y','ACI_past_mean','paired_onset_gain_vs_h0']},flush=True)
if __name__=='__main__':main()

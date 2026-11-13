import os
os.environ['MPLCONFIGDIR']='/workspace/work/mpl';os.environ['XDG_CACHE_HOME']='/workspace/work/cache'
from pathlib import Path
import json,csv,gzip,time
import numpy as np
from source_A import simulate,statistics,SIGMAS,TAUS,STRENGTHS,DT
from stream import scores,calibrate,alarms,episodes,METHODS,CLOCK
ROOT=Path(__file__).parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
CONFIRM=[1,2,3]
CONFIG=dict(stage='C1 basic; C2 independently calibrated reliability; C3 gradual/transient/repeated stress',noise=SIGMAS,tau=TAUS,strength=STRENGTHS,n_each_cell=200,calibration_null=2000,test_null=2000,T_end=25,sample_dt=.05,all_detector_update_dt=.25,candidate_onset_step=.25,tau_grid='same41 log grid .05..5 from C0',confirmation=CONFIRM,primary_confirmation=2,target_trajectory_false_alarm=.05,failure_definition='frozen healthy AR 1-unit-ahead predictions; mean of latest4 matured normalized squared errors; fixed primary k=2 confirmation held constant across C2 variants and independently calibrated whole-trajectory threshold',event_precision_prevalence='balanced 200 event+first200 independent null trajectories; early event false alarm counted FP',C3_shapes=['ramp2','pulse1','repeated2'],C3_tau=[.5,2],C3_onset='random8..12, not supplied',episode_matching_window=5,hysteresis_reset='two consecutive values below half high threshold; compare high-threshold reset',fixed_model_inputs='C0 independent-X normal AR and shared tau; frozen HMM seed11; no Y,U,true tau,T1 supplied to detector',mechanism_interpretation='known first-order step-change hypothesis; does not uniquely distinguish source change from transmission-gain change')

def complex_data(n,sigma,tau,seed,kind):
 rng=np.random.default_rng(seed);on=rng.integers(160,241,n);y=np.zeros((n,500));events=[]
 for i,s in enumerate(on):
  events.append([s*DT]+([(s+120)*DT] if kind=='repeated2' else []))
  if kind=='ramp2':y[i,s:]=np.minimum((np.arange(s,500)-s)/40,1)
  elif kind=='pulse1':y[i,s:s+20]=1
  elif kind=='repeated2':y[i,s:s+40]=1;y[i,s+120:s+160]=1
  else:raise ValueError(kind)
 x=np.zeros((n,501));u=np.zeros_like(x);x[:,0]=rng.normal(10,.3,n);q=np.exp(-.15*DT);r=np.exp(-DT/tau);j=(r-q)/(.15-1/tau);kk=(1-q)/.15
 for t in range(500):x[:,t+1]=q*x[:,t]+1.5*kk-1.5*(y[:,t]*kk+(u[:,t]-y[:,t])*j)+sigma*np.sqrt((1-q*q)/.3)*rng.normal(size=n);u[:,t+1]=y[:,t]+(u[:,t]-y[:,t])*r
 return dict(x=x,y=y,u=u),np.array(on)*DT,events

def evaluate(sc,thresholds,onsets,meta,ks):
 rows=[]
 tx=alarms(sc['M4_X'],thresholds['M4_X']['2'],2)[0];tf=alarms(sc['M4_E'],thresholds['M4_E']['2'],2)[0]
 for k in ks:
  detected={m:alarms(sc[m],thresholds[m][str(k)],k) for m in METHODS}
  for m in METHODS:
   alarm,ix=detected[m];valid=np.isfinite(alarm)&(alarm>=onsets);validx=np.isfinite(tx)&(tx>=onsets);validf=np.isfinite(tf)&(tf>=onsets)
   for i in range(len(alarm)):
    estimate=sc['onset_'+m][i,ix[i]] if m in ['M2','M3'] and valid[i] else np.nan
    rows.append(dict(**meta,method=m,k=k,trajectory=i,T_Y=onsets[i],T_alarm=alarm[i],T_X=tx[i],T_failure=tf[i],valid_alarm=int(valid[i]),early_alarm=int(np.isfinite(alarm[i]) and alarm[i]<onsets[i]),missing_alarm=int(not np.isfinite(alarm[i])),delay=alarm[i]-onsets[i] if valid[i] else np.nan,capped_delay=alarm[i]-onsets[i] if valid[i] else 25-onsets[i],lead_X=tx[i]-alarm[i] if valid[i] and validx[i] else np.nan,lead_failure=tf[i]-alarm[i] if valid[i] and validf[i] else np.nan,coverage_X=int(valid[i] and validx[i] and alarm[i]<tx[i]),coverage_failure=int(valid[i] and validf[i] and alarm[i]<tf[i]),valid_X=int(validx[i]),valid_failure=int(validf[i]),early_failure=int(np.isfinite(tf[i]) and tf[i]<onsets[i]),onset_estimate=estimate,onset_error=abs(estimate-onsets[i]) if np.isfinite(estimate) else np.nan,tau_estimate=sc['tau_M3'][i,ix[i]] if m=='M3' and valid[i] else np.nan))
 return rows

def writecsv(path,rows):
 with gzip.open(path,'wt',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def wilson(success,total):
 if total==0:return dict(n=0,rate=None,ci95=None)
 p=success/total;z=1.96;den=1+z*z/total;center=(p+z*z/(2*total))/den;half=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
 return dict(n=int(total),rate=float(p),ci95=[float(center-half),float(center+half)])

def cluster_rate(numerators,denominators):
 num=np.asarray(numerators,float);den=np.asarray(denominators,float);total=den.sum()
 if total==0:return dict(n_trajectories=len(num),n_units=0,rate=None,ci95=None)
 rng=np.random.default_rng(6101);idx=rng.integers(0,len(num),(1500,len(num)));dd=den[idx].sum(1);nn=num[idx].sum(1);ratio=nn[dd>0]/dd[dd>0]
 return dict(n_trajectories=len(num),n_units=int(total),rate=float(num.sum()/total),ci95=np.quantile(ratio,[.025,.975]).tolist())

def main():
 (OUT/'config.json').write_text(json.dumps(CONFIG,indent=2));c0=json.loads((OUT/'calibration_C0.json').read_text());allrows=[];nullrows=[];ep_rows=[];threshold_out={};examples={};start=time.time()
 for gi,sigma in enumerate(SIGMAS):
  base=c0[str(sigma)]['normal_AR'];fixed=c0[str(sigma)]['shared_index'];ck=np.load(OUT/f'model_{gi}_11.npz');p={k:ck[k] for k in ['beta','mu','var','A','pi']}
  val=simulate(2000,sigma,1,0,31000+gi,False)['x'];v=scores(val,base,p,fixed);thresholds={m:{str(k):calibrate(v[m],k) for k in CONFIRM} for m in METHODS};threshold_out[str(sigma)]=thresholds;del v,val
  normal=simulate(2000,sigma,1,0,32000+gi,False)['x'];ns=scores(normal,base,p,fixed)
  for m in METHODS:
   for k in CONFIRM:
    tt,ii=alarms(ns[m],thresholds[m][str(k)],k)
    for i in range(len(tt)):nullrows.append(dict(sigma=sigma,method=m,k=k,trajectory=i,alarm=int(np.isfinite(tt[i])),alarm_time=tt[i]))
  print('noise',sigma,'C2 calibration/null done',round(time.time()-start,1),flush=True)
  for ti,tau in enumerate(TAUS):
   for si,strength in enumerate(STRENGTHS):
    data=simulate(200,sigma,tau,strength,13000+gi);sc=scores(data['x'],base,p,fixed);allrows.extend(evaluate(sc,thresholds,np.full(200,10.),dict(stage='C1',sigma=sigma,tau=tau,strength=strength,shape='step'),CONFIRM))
    if strength==1:
     key=f'n{gi}_t{ti}_';examples[key+'x']=data['x'][0];examples[key+'y']=data['y'][0];examples[key+'u']=data['u'][0]
     for name in METHODS:examples[key+name]=sc[name][0];examples[key+name+'_threshold']=thresholds[name]['2']
  print('noise',sigma,'C1 done',round(time.time()-start,1),flush=True)
  for ti,tau in enumerate([.5,2]):
   for si,kind in enumerate(CONFIG['C3_shapes']):
    data,on,events=complex_data(200,sigma,tau,33000+gi*10+si,kind);sc=scores(data['x'],base,p,fixed);allrows.extend(evaluate(sc,thresholds,on,dict(stage='C3',sigma=sigma,tau=tau,strength=1,shape=kind),[2]))
    for m in METHODS:
     for hy in [False,True]:
      eps=episodes(sc[m],thresholds[m]['2'],2,hy)
      for i,alerts in enumerate(eps):
       used=set();matched=[]
       for event in events[i]:
        candidates=[j for j,a in enumerate(alerts) if j not in used and event<=a<=event+5]
        if candidates:used.add(candidates[0]);matched.append(1)
        else:matched.append(0)
       ep_rows.append(dict(sigma=sigma,tau=tau,shape=kind,method=m,hysteresis=int(hy),trajectory=i,n_events=len(events[i]),matched_events=sum(matched),second_event_detected=matched[1] if len(matched)>1 else np.nan,n_alarms=len(alerts),unmatched_alarms=len(alerts)-len(used)))
    key=f'C3_n{gi}_t{ti}_s{si}_';examples[key+'x']=data['x'][0];examples[key+'y']=data['y'][0];examples[key+'on']=on[0]
    for name in METHODS:examples[key+name]=sc[name][0];examples[key+name+'_threshold']=thresholds[name]['2']
  print('noise',sigma,'C3 done',round(time.time()-start,1),flush=True)
 writecsv(OUT/'per_trajectory.csv.gz',allrows);writecsv(OUT/'null_trajectories.csv.gz',nullrows);writecsv(OUT/'episodes.csv.gz',ep_rows);np.savez_compressed(OUT/'examples.npz',**examples);(OUT/'thresholds.json').write_text(json.dumps(threshold_out,indent=2));summarize(allrows,nullrows,ep_rows)

def summarize(rows,nullrows,ep_rows):
 from collections import defaultdict
 groups=defaultdict(list);ngroups=defaultdict(list);egroups=defaultdict(list)
 for r in rows:groups[(r['stage'],r['sigma'],r['tau'],r['strength'],r['shape'],r['method'],r['k'])].append(r)
 for r in nullrows:ngroups[(r['sigma'],r['method'],r['k'])].append(r)
 for r in ep_rows:egroups[(r['sigma'],r['tau'],r['shape'],r['method'],r['hysteresis'])].append(r)
 out=[];nsummary=[];esummary=[]
 for key,sub in groups.items():
  stage,sigma,tau,strength,shape,method,k=key;metrics={name:statistics([r[name] for r in sub]) for name in ['T_alarm','T_X','T_failure','delay','capped_delay','lead_X','lead_failure','coverage_X','coverage_failure','valid_X','valid_failure','early_failure','onset_error','tau_estimate']};tp=sum(r['valid_alarm'] for r in sub);early=sum(r['early_alarm'] for r in sub);miss=sum(r['missing_alarm'] for r in sub);null_match=ngroups[(sigma,method,k)];fp_null=sum(r['alarm'] for r in null_match if r['trajectory']<200);validfail=sum(r['valid_failure'] for r in sub)
  out.append(dict(stage=stage,sigma=sigma,tau=tau,strength=strength,shape=shape,method=method,k=k,metrics=metrics,recall=wilson(tp,len(sub)),precision_balanced=wilson(tp,tp+early+fp_null),early_alarm=wilson(early,len(sub)),missing_alarm=wilson(miss,len(sub)),coverage_failure_given_observed=wilson(sum(r['coverage_failure'] for r in sub),validfail)))
 for key,sub in ngroups.items():nsummary.append(dict(sigma=key[0],method=key[1],k=key[2],FPR=wilson(sum(r['alarm'] for r in sub),len(sub))))
 for key,sub in egroups.items():esummary.append(dict(sigma=key[0],tau=key[1],shape=key[2],method=key[3],hysteresis=key[4],event_recall=cluster_rate([r['matched_events'] for r in sub],[r['n_events'] for r in sub]),event_precision=cluster_rate([r['matched_events'] for r in sub],[r['n_alarms'] for r in sub]),second_event_recall=statistics([r['second_event_detected'] for r in sub]),mean_alarms=statistics([r['n_alarms'] for r in sub])))
 (OUT/'summary.json').write_text(json.dumps(out,indent=2));(OUT/'null_summary.json').write_text(json.dumps(nsummary,indent=2));(OUT/'episode_summary.json').write_text(json.dumps(esummary,indent=2))
 for r in out:
  if r['stage']=='C1' and r['sigma']==.35 and r['strength']==1 and r['k']==2:print(r['tau'],r['method'],{m:r['metrics'][m]['mean'] for m in ['delay','lead_X','lead_failure','coverage_failure','valid_failure']},'recall',r['recall']['rate'],flush=True)
if __name__=='__main__':main()

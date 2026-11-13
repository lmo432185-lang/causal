import json,csv,gzip,numpy as np
from collections import defaultdict
from config import *
from source_A import statistics

def read(name):
 with gzip.open(OUT/name,'rt') as f:rows=list(csv.DictReader(f))
 for r in rows:
  for key,value in list(r.items()):
   if key in ['method','score_name']:continue
   r[key]=float(value)
   if key in ['cohort','trajectory','valid_alarm','early_alarm','missing_alarm','coverage_failure','coverage_X','failure_observed','oracle_parameter_given','alarm'] and (key!='alarm' or 'score_name' in r):r[key]=int(r[key])
 return rows

def wilson(n,N):
 if not N:return dict(n=0,rate=None,ci95=None)
 p=n/N;z=1.96;d=1+z*z/N;c=(p+z*z/(2*N))/d;h=z*np.sqrt(p*(1-p)/N+z*z/(4*N*N))/d
 return dict(n=int(N),rate=float(p),ci95=[float(c-h),float(c+h)])
def paired_stats(v):
 result=statistics(v);return result

def cluster_summary(blocks):
 # Each stratum is a fixed (noise, independent cohort), with200 independent
 # innovations; conditions within an innovation are averaged before resampling.
 arr=np.array(blocks);rng=np.random.default_rng(7141);B=4000;means=np.empty((B,arr.shape[-1]))
 for start in range(0,B,100):
  count=min(100,B-start);indices=rng.integers(0,arr.shape[1],(count,arr.shape[0],arr.shape[1]));res=np.take_along_axis(arr[None,:,:,:],indices[:,:,:,None],axis=2);means[start:start+count]=res.mean((1,2))
 point=arr.mean((0,1));out={}
 for j,name in enumerate(['coverage_gain','capped_delay_gain','prediction_gain_at12','prediction_gain_at15']):
  out[name]=dict(mean=float(point[j]),ci95=np.quantile(means[:,j],[.025,.975]).tolist(),decision_ci9875=np.quantile(means[:,j],[.00625,.99375]).tolist(),cohort_means=[float(arr[np.arange(arr.shape[0])%2==cohort,:,j].mean()) for cohort in COHORTS])
 return out

def main():
 rows=read('per_trajectory.csv.gz');null=read('null_trajectories.csv.gz');groups=defaultdict(list);ngroups=defaultdict(list)
 for r in rows:groups[(r['sigma'],r['tau'],r['strength'],r['method'])].append(r)
 for r in null:ngroups[(r['sigma'],r['score_name'])].append(r)
 nsummary=[]
 for (sig,name),sub in ngroups.items():nsummary.append(dict(sigma=sig,score_name=name,FPR=wilson(sum(r['alarm'] for r in sub),len(sub)),prediction_at12=statistics([r['forecast_mse_at12'] for r in sub]),prediction_at15=statistics([r['forecast_mse_at15'] for r in sub])))
 out=[]
 for (sig,tau,strength,m),sub in groups.items():
  for cohort in [-1,0,1]:
   data=sub if cohort==-1 else [r for r in sub if r['cohort']==cohort];metrics={k:statistics([r[k] for r in data]) for k in ['delay','capped_delay','lead_failure','lead_X','coverage_failure','coverage_X','failure_observed','onset_error','tau_estimate','tau_error','forecast_mse_at12','forecast_mse_at15']};name=f'oracle_{tau}' if m=='M2_oracle' else m;ns=ngroups[(sig,name)];wanted=set(range(2*N_EVENT)) if cohort==-1 else set(range(cohort*N_EVENT,(cohort+1)*N_EVENT));fp=sum(r['alarm'] for r in ns if r['trajectory'] in wanted);tp=sum(r['valid_alarm'] for r in data);early=sum(r['early_alarm'] for r in data)
   out.append(dict(sigma=sig,tau=tau,strength=strength,method=m,cohort=cohort,metrics=metrics,recall=wilson(tp,len(data)),precision_balanced=wilson(tp,tp+early+fp),early_alarm=wilson(early,len(data)),missing=wilson(sum(r['missing_alarm'] for r in data),len(data))))
 by={(r['sigma'],r['tau'],r['strength'],r['cohort'],r['trajectory'],r['method']):r for r in rows};pairs=[]
 for sig in SIGMAS:
  for tau in TAUS:
   for strength in STRENGTHS:
    for left,right in [('M3','CUSUM'),('M2_oracle','CUSUM'),('M3','M2'),('M3','M2_oracle')]:
     values=defaultdict(list)
     for cohort in COHORTS:
      for i in range(N_EVENT):
       a=by[(sig,tau,strength,cohort,i,left)];b=by[(sig,tau,strength,cohort,i,right)];values['coverage_gain'].append(a['coverage_failure']-b['coverage_failure']);values['capped_delay_gain'].append(b['capped_delay']-a['capped_delay']);values['paired_lead_gain'].append(a['lead_failure']-b['lead_failure'] if np.isfinite(a['lead_failure']) and np.isfinite(b['lead_failure']) else np.nan);values['prediction_gain_at12'].append(b['forecast_mse_at12']-a['forecast_mse_at12']);values['prediction_gain_at15'].append(b['forecast_mse_at15']-a['forecast_mse_at15'])
     pairs.append(dict(sigma=sig,tau=tau,strength=strength,left=left,right=right,metrics={k:paired_stats(v) for k,v in values.items()}))
 fairness=[];fair=True
 for sig in SIGMAS:
  a=sorted(ngroups[(sig,'M3')],key=lambda r:r['trajectory']);b=sorted(ngroups[(sig,'CUSUM')],key=lambda r:r['trajectory']);diff=paired_stats([x['alarm']-y['alarm'] for x,y in zip(a,b)]);ra=np.mean([r['alarm'] for r in a]);rb=np.mean([r['alarm'] for r in b]);ok=.04<=ra<=.06 and .04<=rb<=.06 and diff['ci95'][0]>=-.01 and diff['ci95'][1]<=.01;fair&=ok;fairness.append(dict(sigma=sig,M3_FPR=float(ra),CUSUM_FPR=float(rb),paired_FPR_difference=diff,criterion_passed=bool(ok)))
 endpoint={}
 for label in ['broad','slow']:
  blocks=[]
  for sig in SIGMAS:
   for cohort in COHORTS:
    per=[]
    for i in range(N_EVENT):
     v=[]
     for tau in TAUS:
      for strength in STRENGTHS:
       if label=='slow' and (tau not in [1.,2.] or strength!=1.):continue
       a=by[(sig,tau,strength,cohort,i,'M3')];b=by[(sig,tau,strength,cohort,i,'CUSUM')];v.append([a['coverage_failure']-b['coverage_failure'],b['capped_delay']-a['capped_delay'],b['forecast_mse_at12']-a['forecast_mse_at12'],b['forecast_mse_at15']-a['forecast_mse_at15']])
     per.append(np.mean(v,axis=0))
    blocks.append(per)
  endpoint[label]=cluster_summary(blocks)
 def succeeds(group):
  qualified=[]
  for k,minimum in [('coverage_gain',.05),('capped_delay_gain',.25)]:
   v=group[k]
   if v['mean']>=minimum and v['decision_ci9875'][0]>0 and all(x>0 for x in v['cohort_means']):qualified.append(k)
  return qualified
 qa=succeeds(endpoint['broad']);qb=succeeds(endpoint['slow']);verdict='A' if fair and qa else 'B' if fair and qb else 'C'
 decision=dict(classification=verdict,FPR_equivalence_passed=bool(fair),broad_qualified_outcomes=qa,slow_qualified_outcomes=qb,meaning={'A':'predeclared broad independent advantage supported','B':'only predeclared slow subgroup advantage supported','C':'prespecified robust independent advantage not established; not evidence of exact equivalence'}[verdict],groups=endpoint,FPR_fairness=fairness,independent_blocks=3*2*N_EVENT,decision_interval='98.75% stratified paired cluster bootstrap,4 primary/group-outcome comparisons; 5pp or0.25 practical minimum; positive direction in each cohort')
 (OUT/'summary.json').write_text(json.dumps(out,indent=2));(OUT/'null_summary.json').write_text(json.dumps(nsummary,indent=2));(OUT/'paired_summary.json').write_text(json.dumps(pairs,indent=2));(OUT/'decision.json').write_text(json.dumps(decision,indent=2));print(json.dumps(decision,indent=2))
if __name__=='__main__':main()

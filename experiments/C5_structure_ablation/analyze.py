import json,csv,gzip,numpy as np
from collections import defaultdict
from config import *
from source_A import statistics

def read(name):
 with gzip.open(OUT/name,'rt') as f:rows=list(csv.DictReader(f))
 for r in rows:
  for k,v in list(r.items()):
   if k=='model':continue
   r[k]=float(v)
   if k in ['cohort','trajectory','valid_alarm','early_alarm','missing_alarm','failure_observed'] or k=='alarm' and 'alarm_time' in r:r[k]=int(r[k])
 return rows

def wilson(success,total):
 if not total:return dict(n=0,rate=None,ci95=None)
 p=success/total;z=1.96;d=1+z*z/total;cc=(p+z*z/(2*total))/d;hh=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/d;return dict(n=int(total),rate=float(p),ci95=[float(cc-hh),float(cc+hh)])

def bootstrap(blocks):
 arr=np.array(blocks);rng=np.random.default_rng(5171);B=4000;values=np.empty(B)
 for start in range(0,B,100):
  count=min(100,B-start);ix=rng.integers(0,arr.shape[1],(count,arr.shape[0],arr.shape[1]));out=np.take_along_axis(arr[None,:,:],ix,axis=2);values[start:start+count]=out.mean((1,2))
 return dict(mean=float(arr.mean()),ci95=np.quantile(values,[.025,.975]).tolist(),corrected_ci99375=np.quantile(values,[.003125,.996875]).tolist(),cohort_means=[float(arr[np.arange(6)%2==c].mean()) for c in COHORTS])

def main():
 rows=read('per_trajectory.csv.gz');null=read('null_trajectories.csv.gz');groups=defaultdict(list);ngroups=defaultdict(list)
 for r in rows:groups[(r['sigma'],r['tau'],r['strength'],r['model'])].append(r)
 for r in null:ngroups[(r['sigma'],r['model'])].append(r)
 summary=[];nsummary=[]
 for (sig,tau,strength,model),sub in groups.items():
  for cohort in [-1,0,1]:
   data=sub if cohort==-1 else [r for r in sub if r['cohort']==cohort];keys=['delay','capped_delay','lead_failure','failure_observed','onset_error','shape_estimate','tau_error','generic_mean_lag_error','forecast_mse_at12','forecast_mse_at15']+[f'coverage_{m}' for m in MARGINS];metrics={k:statistics([r[k] for r in data]) for k in keys};summary.append(dict(sigma=sig,tau=tau,strength=strength,model=model,cohort=cohort,metrics=metrics,recall=wilson(sum(r['valid_alarm'] for r in data),len(data)),early_alarm=wilson(sum(r['early_alarm'] for r in data),len(data)),miss=wilson(sum(r['missing_alarm'] for r in data),len(data))))
 for (sig,model),data in ngroups.items():nsummary.append(dict(sigma=sig,model=model,FPR=wilson(sum(r['alarm'] for r in data),len(data)),forecast_at12=statistics([r['forecast_mse_at12'] for r in data]),forecast_at15=statistics([r['forecast_mse_at15'] for r in data])))
 by={(r['sigma'],r['tau'],r['strength'],r['cohort'],r['trajectory'],r['model']):r for r in rows};comparators=['Direct','Ablation','GLR_Ramp','M2'];pairs=[];groupsout={};fair={}
 for model in comparators+['CUSUM']:
  checks=[]
  for sig in SIGMAS:
   aa=sorted(ngroups[(sig,'M3')],key=lambda r:r['trajectory']);bb=sorted(ngroups[(sig,model)],key=lambda r:r['trajectory']);d=statistics([x['alarm']-y['alarm'] for x,y in zip(aa,bb)]);ra=np.mean([r['alarm'] for r in aa]);rb=np.mean([r['alarm'] for r in bb]);ok=.04<=ra<=.06 and .04<=rb<=.06 and d['ci95'][0]>=-.01 and d['ci95'][1]<=.01;checks.append(dict(sigma=sig,M3_FPR=float(ra),control_FPR=float(rb),paired_difference=d,passed=bool(ok)))
  fair[model]=dict(passed=all(c['passed'] for c in checks),details=checks)
  for sig in SIGMAS:
   for tau in TAUS:
    for strength in STRENGTHS:
     vv=defaultdict(list)
     for cohort in COHORTS:
      for i in range(N):
       a=by[(sig,tau,strength,cohort,i,'M3')];b=by[(sig,tau,strength,cohort,i,model)]
       for m in MARGINS:vv[f'coverage_gain_{m}'].append(a[f'coverage_{m}']-b[f'coverage_{m}'])
       vv['capped_delay_gain'].append(b['capped_delay']-a['capped_delay']);vv['prediction_gain_at12'].append(b['forecast_mse_at12']-a['forecast_mse_at12']);vv['prediction_gain_at15'].append(b['forecast_mse_at15']-a['forecast_mse_at15']);vv['paired_lead_gain'].append(a['lead_failure']-b['lead_failure'] if np.isfinite(a['lead_failure']) and np.isfinite(b['lead_failure']) else np.nan)
     pairs.append(dict(sigma=sig,tau=tau,strength=strength,control=model,metrics={k:statistics(v) for k,v in vv.items()}))
  groupsout[model]={}
  for label in ['broad','slow']:
   blocks=[]
   for sig in SIGMAS:
    for cohort in COHORTS:
     bb=[]
     for i in range(N):
      vals=[]
      for tau in TAUS:
       for strength in STRENGTHS:
        if label=='slow' and (tau not in [1.,2.] or strength!=1):continue
        a=by[(sig,tau,strength,cohort,i,'M3')];b=by[(sig,tau,strength,cohort,i,model)];vals.append(a['coverage_0.5']-b['coverage_0.5'])
      bb.append(np.mean(vals))
     blocks.append(bb)
   result=bootstrap(blocks);result['practical_corrected_superiority']=bool(result['mean']>=.05 and result['corrected_ci99375'][0]>0 and all(v>0 for v in result['cohort_means']) and fair[model]['passed']);groupsout[model][label]=result
 # Exact matched-control equality is a required model-identity check, not a result to hide.
 exact=True
 for key,a in by.items():
  if key[-1]!='M3':continue
  b=by[key[:-1]+('Ablation',)]
  for field in ['alarm','onset_estimate','shape_estimate','forecast_mse_at12','forecast_mse_at15']:
   exact&=bool(np.isclose(a[field],b[field],equal_nan=True,atol=0,rtol=0))
 strict={label:all(groupsout[m][label]['practical_corrected_superiority'] for m in comparators) for label in ['broad','slow']};kernel={label:all(groupsout[m][label]['practical_corrected_superiority'] for m in ['Direct','GLR_Ramp']) for label in ['broad','slow']};adapt={label:groupsout['M2'][label]['practical_corrected_superiority'] for label in ['broad','slow']}
 decision=dict(strict_structure_superiority=strict,response_kernel_subclaim=kernel,adaptive_tau_subclaim=adapt,exact_M3_Ablation_equality=bool(exact),comparisons=groupsout,FPR_fairness=fair,primary_margin=.5,corrected_interval=.99375,independent_innovation_blocks=1200,interpretation='Explicit Y/U labels do not add observable information in the frozen deterministic step-response family; non-equivalent kernel/shape contrasts must be assessed separately')
 tails=[]
 for (sig,tau,strength,model),data in groups.items():
  if strength!=1:continue
  for origin in [12,15]:
   a=np.array([r[f'forecast_mse_at{origin}'] for r in data]);tails.append(dict(sigma=sig,tau=tau,model=model,origin=origin,mean=float(a.mean()),median=float(np.median(a)),p95=float(np.quantile(a,.95)),p99=float(np.quantile(a,.99)),max=float(a.max()),top1pct_share=float(np.sort(a)[-4:].sum()/a.sum())))
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));(OUT/'null_summary.json').write_text(json.dumps(nsummary,indent=2));(OUT/'paired_summary.json').write_text(json.dumps(pairs,indent=2));(OUT/'decision.json').write_text(json.dumps(decision,indent=2));(OUT/'prediction_tails.json').write_text(json.dumps(tails,indent=2));print(json.dumps(decision,indent=2))
if __name__=='__main__':main()

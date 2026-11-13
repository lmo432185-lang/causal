import json,numpy as np
from collections import defaultdict
from analyze import read
from config import *
from generate import event_data
from detectors import all_scores,forecast_from_scores
from calibrate import load
from source_A import statistics

def main():
 rows=read('per_trajectory.csv.gz');by={(r['sigma'],r['tau'],r['strength'],r['cohort'],r['trajectory'],r['method']):r for r in rows};margins=[]
 for margin in [.25,.5,1.,2.]:
  blocks=defaultdict(list);rates=defaultdict(list)
  for r in rows:
   if r['method']!='M3':continue
   key=(r['sigma'],r['tau'],r['strength'],r['cohort'],r['trajectory']);b=by[key+('CUSUM',)];ca=int(np.isfinite(r['lead_failure']) and r['lead_failure']>=margin-1e-12);cb=int(np.isfinite(b['lead_failure']) and b['lead_failure']>=margin-1e-12);blocks[(r['sigma'],r['cohort'],r['trajectory'])].append(ca-cb);rates['M3'].append(ca);rates['CUSUM'].append(cb)
  margins.append(dict(minimum_lead=margin,M3_coverage=float(np.mean(rates['M3'])),CUSUM_coverage=float(np.mean(rates['CUSUM'])),paired_cluster_coverage_gain=statistics([np.mean(v) for v in blocks.values()])))
 (OUT/'lead_margin_sensitivity.json').write_text(json.dumps(margins,indent=2))
 tails=[]
 for sig in SIGMAS:
  for tau in TAUS:
   for method in ['M1','M2','M2_oracle','M3','CUSUM']:
    for origin in [12,15]:
     vals=np.array([r[f'forecast_mse_at{origin}'] for r in rows if r['sigma']==sig and r['tau']==tau and r['strength']==1 and r['method']==method]);sort=np.sort(vals);tails.append(dict(sigma=sig,tau=tau,method=method,origin=origin,mean=float(vals.mean()),median=float(np.median(vals)),p95=float(np.quantile(vals,.95)),p99=float(np.quantile(vals,.99)),max=float(vals.max()),top1pct_error_share=float(sort[-4:].sum()/sort.sum())))
 (OUT/'prediction_tails.json').write_text(json.dumps(tails,indent=2))
 # Failure diagnosis only: selected largest errors, not representative examples.
 c0=json.loads((OUT/'calibration_C0.json').read_text());cal=json.loads((OUT/'calibration.json').read_text());bad=[]
 for model in ['M3','M2_oracle']:
  candidates=[r for r in rows if r['sigma']==.35 and r['tau']==2 and r['strength']==1 and r['method']==model];selected=sorted(candidates,key=lambda r:r['forecast_mse_at12'],reverse=True)[:3]
  for r in selected:
   x=event_data(.35,2,1,1,r['cohort'])['x'][r['trajectory']:r['trajectory']+1,:241];base,fixed,p=load(1,c0);sc=all_scores(x,base,p,fixed,cal['0.35']['scales']);name='M3' if model=='M3' else 'oracle_2.0';tau=float(sc[f'tau_{name}'][0,-1]);on=float(sc[f'on_{name}'][0,-1]);amp=float(sc[f'amp_{name}'][0,-1]);pred=forecast_from_scores(x,sc,base,p,2.,12)[model][0]
   a=-np.log(base['q'])/DT;q=base['q'];rr=np.exp(-DT/tau);j=(rr-q)/(a-1/tau);kk=(1-q)/a;u=amp*(1-np.exp(-(12-on)/tau));xx=x[0,-1]
   for _ in range(40):xx=q*xx+base['c']-amp*kk-(u-amp)*j;u=amp+(u-amp)*rr
   np.testing.assert_allclose(pred,xx,atol=1e-7,rtol=1e-9)
   bad.append(dict(model=model,cohort=r['cohort'],trajectory=r['trajectory'],origin=12,onset_estimate=on,tau_estimate=tau,source_effect_amplitude=amp,prediction=pred,MSE=r['forecast_mse_at12'],formula_matches_independent_model_rollout=True,selection='posthoc top-error diagnosis only'))
 (OUT/'prediction_failure_diagnosis.json').write_text(json.dumps(bad,indent=2));print('MARGINS',json.dumps(margins,indent=2));print('BAD',json.dumps(bad,indent=2))
if __name__=='__main__':main()

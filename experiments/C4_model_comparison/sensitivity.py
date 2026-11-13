"""Post-primary diagnostic: every prefixed CUSUM component, not best test-selected."""
import json,numpy as np,csv,gzip
from config import *
from generate import null_data,event_data
from detectors import cusum_components,calibrate,alarms
from calibrate import load
from source_A import statistics

def main():
 from analyze import read,wilson
 raw=read('per_trajectory.csv.gz');ref={(r['sigma'],r['tau'],r['strength'],r['cohort'],r['trajectory']):r for r in raw if r['method']=='M3'};c0=json.loads((OUT/'calibration_C0.json').read_text());result=[];npairs=[];allrecords=[]
 for gi,sigma in enumerate(SIGMAS):
  base,_,_=load(gi,c0);cv=cusum_components(null_data(N_CAL,sigma,SEEDS['final_calibration'][gi]),base);nt=cusum_components(null_data(N_NULL,sigma,SEEDS['null_test'][gi]),base);ths=[calibrate(cv[:,j,:],2) for j in range(5)];null_rates=[wilson(np.isfinite(alarms(nt[:,j,:],ths[j],2)[0]).sum(),N_NULL) for j in range(5)]
  for tau in TAUS:
   for strength in STRENGTHS:
    bydelta=[[] for _ in range(5)]
    for cohort in COHORTS:
     x=event_data(sigma,tau,strength,gi,cohort)['x'];cs=cusum_components(x,base)
     for j,delta in enumerate(CUSUM_ALLOWANCES):
      aa=alarms(cs[:,j,:],ths[j],2)[0]
      for i,alarm in enumerate(aa):
       r=ref[(sigma,tau,strength,cohort,i)];valid=np.isfinite(alarm) and alarm>=10;vf=np.isfinite(r['failure']) and r['failure']>=10;cov=int(valid and vf and alarm<r['failure']);cap=alarm-10 if valid else 15.;row=dict(sigma=sigma,tau=tau,strength=strength,cohort=cohort,trajectory=i,allowance=delta,M3_minus_component_coverage=r['coverage_failure']-cov,component_minus_M3_capped_delay=cap-r['capped_delay']);bydelta[j].append(row);allrecords.append(row)
    for j,delta in enumerate(CUSUM_ALLOWANCES):result.append(dict(sigma=sigma,tau=tau,strength=strength,allowance=delta,FPR=null_rates[j],coverage_gain=statistics([r['M3_minus_component_coverage'] for r in bydelta[j]]),capped_delay_gain=statistics([r['component_minus_M3_capped_delay'] for r in bydelta[j]])))
 # Cluster over shared innovations across all36 conditions for each fixed delta.
 for delta in CUSUM_ALLOWANCES:
  sub=[r for r in allrecords if r['allowance']==delta];vals=[]
  for sig in SIGMAS:
   for cohort in COHORTS:
    for i in range(N_EVENT):vals.append(np.mean([r['M3_minus_component_coverage'] for r in sub if r['sigma']==sig and r['cohort']==cohort and r['trajectory']==i]))
  npairs.append(dict(allowance=delta,broad_coverage_gain=statistics(vals)))
 (OUT/'fixed_CUSUM_sensitivity.json').write_text(json.dumps(result,indent=2));(OUT/'fixed_CUSUM_overview.json').write_text(json.dumps(npairs,indent=2));print(json.dumps(npairs,indent=2))
if __name__=='__main__':main()

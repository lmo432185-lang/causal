import csv,gzip,json,numpy as np
from run import OUT,SIGMAS,TAUS,STRENGTHS,statistics
with gzip.open(OUT/'per_trajectory.csv.gz','rt') as f:raw=list(csv.DictReader(f))
result=[]
for sigma in SIGMAS:
 for tau in TAUS:
  for strength in STRENGTHS:
   rows=[r for r in raw if float(r['sigma'])==sigma and float(r['tau'])==tau and float(r['strength'])==strength];by={(r['model'],int(r['trajectory'])):r for r in rows};item=dict(sigma=sigma,tau=tau,strength=strength)
   for old in ['M1','M2']:
    for key in ['onset_error_capped','U_rmse','forecast_mse_h40']:
     diff=[float(by[(old,i)][key])-float(by[('M3',i)][key]) for i in range(200)];item[f'{old}_minus_M3_{key}']=statistics(diff)
   result.append(item)
(OUT/'paired_summary.json').write_text(json.dumps(result,indent=2));print('Paired comparisons saved')

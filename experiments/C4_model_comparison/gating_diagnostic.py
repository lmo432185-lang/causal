"""Post-primary readout diagnostic. No parameters or primary endpoints changed."""
import json,numpy as np
from collections import defaultdict
from config import *
from analyze import read,cluster_summary
from source_A import statistics
rows=read('per_trajectory.csv.gz');by={(r['sigma'],r['tau'],r['strength'],r['cohort'],r['trajectory'],r['method']):r for r in rows};result=[]
for mode in ['common_CUSUM_gate','own_detector_gate']:
 for origin in [12,15]:
  blocks=defaultdict(list);examples=[]
  for r in rows:
   if r['method']!='M3':continue
   key=(r['sigma'],r['tau'],r['strength'],r['cohort'],r['trajectory']);cu=by[key+('CUSUM',)];healthy=by[key+('X',)][f'forecast_mse_at{origin}'];gc=np.isfinite(cu['alarm']) and cu['alarm']<=origin;gm=gc if mode=='common_CUSUM_gate' else np.isfinite(r['alarm']) and r['alarm']<=origin
   em=r[f'forecast_mse_at{origin}'] if gm else healthy;ec=cu[f'forecast_mse_at{origin}'] if gc else healthy;blocks[(r['sigma'],r['cohort'],r['trajectory'])].append(ec-em)
   if r['sigma']==.35 and r['tau']==2 and r['strength']==1:examples.append((em,ec,gm,gc))
  ex=np.array(examples);result.append(dict(mode=mode,origin=origin,all_condition_block_prediction_gain=statistics([np.mean(v) for v in blocks.values()]),example_medium_tau2=dict(M3_MSE=float(ex[:,0].mean()),local_drift_MSE=float(ex[:,1].mean()),M3_gate_rate=float(ex[:,2].mean()),CUSUM_gate_rate=float(ex[:,3].mean())),note='exploratory readout check; threshold/gate depends only on X available before origin; all400 trials retained; primary ungated prediction result unchanged'))
(OUT/'gating_diagnostic.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

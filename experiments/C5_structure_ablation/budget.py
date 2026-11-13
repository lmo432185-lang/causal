import json,time,numpy as np
from config import *
from generate import normal
from detectors import generic_scores
from legacy_stream import structured_scores
from calibrate import load
records=[]
for gi,sigma in enumerate(SIGMAS):
 base,fixed,old=load(gi);x=normal(N_CAL,sigma,SEEDS['calibration'][gi])[:200];tt=[];gg=[]
 for _ in range(3):
  start=time.perf_counter();structured_scores(x,base,fixed);tt.append(time.perf_counter()-start);start=time.perf_counter();generic_scores(x,base);gg.append(time.perf_counter()-start)
 candidates=sum(max(0,u-1) for u in range(100));records.append(dict(sigma=sigma,batch=200,M3_M2_backend_median_seconds=float(np.median(tt)),GLR_Direct_backend_median_seconds=float(np.median(gg)),M3_shape_candidates=41,GLR_shape_candidates=41,onset_candidates_over_all_updates=4851,shape_onset_candidates_each=41*4851,normalization_variance_shared=True,matched_mean_lag_support=[.05,5],no_runtime_selection=True))
(OUT/'computation_budget.json').write_text(json.dumps(records,indent=2));print(records)

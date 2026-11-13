import os
os.environ['MPLCONFIGDIR']='/workspace/work/mpl';os.environ['XDG_CACHE_HOME']='/workspace/work/cache'
import json,time,numpy as np
from config import *
from generate import null_data,validation_X
from detectors import cusum_components,local_forecast,all_scores,calibrate

def load(gi,c0):
 sigma=SIGMAS[gi];base=c0[str(sigma)]['normal_AR'];fixed=c0[str(sigma)]['shared_index'];ck=np.load(OUT/f'model_{gi}_11.npz');p={k:ck[k] for k in ['beta','mu','var','A','pi']};return base,fixed,p

def main():
 (OUT/'PROTOCOL.json').write_text(json.dumps(PROTOCOL,indent=2));(OUT/'calibration.json').unlink(missing_ok=True);c0=json.loads((OUT/'calibration_C0.json').read_text());saved={};start=time.time()
 for gi,sigma in enumerate(SIGMAS):
  base,fixed,p=load(gi,c0);x=null_data(N_SCALE,sigma,SEEDS['cusum_scale'][gi]);cu=cusum_components(x,base);scales=[max(calibrate(cu[:,j,:],1),1e-8) for j in range(len(CUSUM_ALLOWANCES))]
  vx=validation_X(sigma,gi);vx=np.concatenate([vx,x[:N_VALID]],axis=0);loss=[]
  for win in DRIFT_WINDOWS:loss.append(float(np.mean([(local_forecast(vx,base,t,win)-vx[:,round((t+2)/DT)])**2 for t in [12,15]])))
  window=DRIFT_WINDOWS[int(np.argmin(loss))];del vx,cu,x
  x=null_data(N_CAL,sigma,SEEDS['final_calibration'][gi]);sc=all_scores(x,base,p,fixed,scales);names=['M1','M2','M3','CUSUM','X','FAIL']+ORACLES;thresholds={name:calibrate(sc[name],2) for name in names};assert all(np.isfinite(v) for v in thresholds.values());assert all(np.isfinite(sc[name]).all() for name in names)
  saved[str(sigma)]=dict(scales=scales,drift_window=window,drift_validation_MSE=dict(zip(map(str,DRIFT_WINDOWS),loss)),thresholds=thresholds)
  print('calibration noise',sigma,'done',round(time.time()-start,1),'local window',window,flush=True)
 (OUT/'calibration.json').write_text(json.dumps(saved,indent=2))
if __name__=='__main__':main()

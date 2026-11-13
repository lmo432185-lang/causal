import os
os.environ['MPLCONFIGDIR']='/workspace/work/mpl';os.environ['XDG_CACHE_HOME']='/workspace/work/cache'
import json,time,numpy as np
from config import *
from generate import normal
from detectors import scores,calibrate

def load(gi):
 c0=json.loads((OUT/'frozen_C0.json').read_text());c4=json.loads((OUT/'frozen_C4.json').read_text());sig=SIGMAS[gi];return c0[str(sig)]['normal_AR'],c0[str(sig)]['shared_index'],c4[str(sig)]

def main():
 (OUT/'PROTOCOL.json').write_text(json.dumps(PROTOCOL,indent=2));out={};begin=time.time()
 for gi,sigma in enumerate(SIGMAS):
  base,fixed,old=load(gi);x=normal(N_CAL,sigma,SEEDS['calibration'][gi]);sc=scores(x,base,fixed,old['scales']);th={m:calibrate(sc[m],2) for m in MODELS};assert all(np.isfinite(sc[m]).all() for m in MODELS);assert th['M3']==th['Ablation'];out[str(sigma)]=dict(thresholds=th,failure_threshold=old['thresholds']['FAIL'],X_threshold=old['thresholds']['X']);print('calibration',sigma,'done',round(time.time()-begin,1),flush=True)
 (OUT/'calibration.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()

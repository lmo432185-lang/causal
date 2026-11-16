import json,time
import numpy as np,pandas as pd
from config import *
from core import generate,ridge
from predictive import fit_x,apply_x

def run():
 rows=[];models=[]
 for ci,cell in enumerate(STRUCTURES):
  for seed in SEEDS:
   # Separate structural dataset seeds; graph name/labels never enter fit_x.
   data_seed=seed+20000+ci*1000;x,truth=generate(cell['params'],data_seed);out=O/'structure'/cell['name']/str(seed);out.mkdir(parents=True,exist_ok=True);model=fit_x(x[:600],x[600:800]);(out/'frozen_predictive.json').write_text(json.dumps(dict(time=time.time(),model=model),indent=2));lv,_,_=apply_x(x[600:800],model);lt,pred,y=apply_x(x[800:],model);nd=truth.shape[-1]-1;targetv=truth[600:800,19:492,:nd].reshape(-1,nd);target=truth[800:,19:492,:nd];recon,probe=ridge(lv,targetv,lt);recon=recon.reshape(200,473,nd);(out/'diagnostic_probe.json').write_text(json.dumps(probe,indent=2))
   for i in range(200):
    for j,key in enumerate(['Y','Z'][:nd]):
     err=recon[i,:,j]-target[i,:,j];rows.append(dict(group=cell['name'],seed=seed,trajectory=i,state=key,MSE=float((err*err).mean()),NMSE=float((err*err).mean()/targetv[:,j].var()),corr=float(np.corrcoef(recon[i,:,j],target[i,:,j])[0,1])))
   models.append(dict(group=cell['name'],seed=seed,rank=model['rank'],test_future_MSE=float(((pred-y)**2).mean()),validation_MSE=model['validation_MSE'],graph_declaration='insufficient evidence: generic predictive dimension does not identify causal graph',causal_refused=True));np.savez_compressed(out/'examples.npz',X=x[800:803],truth=target[:3],reconstruction=recon[:3],L=lt.reshape(200,473,-1)[:3],future_prediction=pred.reshape(200,473,4)[:3]);print('STRUCTURE',cell['name'],seed,'rank',model['rank'],flush=True)
 pd.DataFrame(rows).to_csv(O/'structure_state_metrics.csv.gz',index=False);(O/'structure_models.json').write_text(json.dumps(models,indent=2))

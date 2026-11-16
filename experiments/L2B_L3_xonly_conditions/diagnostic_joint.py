"""Post-hoc diagnostic: current-X bypass belongs to total predictive state.
No predictive model, selected rank, test delay or system parameter is changed.
All new diagnostic readouts fit validation hidden answers only.
"""
import json,time,hashlib
import numpy as np,pandas as pd
from config import R,O,STRUCTURES,SEEDS
from core import generate,ridge
from predictive import apply_x,data

def main():
 (R/'JOINT_DIAGNOSTIC_REGISTERED.json').write_text(json.dumps(dict(time=time.time(),sha256=hashlib.sha256((R/'diagnostic_joint.py').read_bytes()).hexdigest(),status='post-hoc evaluation diagnostic, not original primary protocol',reason='predictive learner explicitly residualizes L against current X and uses current-X bypass; evaluate joint available state without retraining'),indent=2));rows=[]
 for ci,cell in enumerate(STRUCTURES):
  for seed in SEEDS:
   m=json.loads((O/'structure'/cell['name']/str(seed)/'frozen_predictive.json').read_text())['model'];x,s=generate(cell['params'],seed+20000+ci*1000);lv,_,_=apply_x(x[600:800],m);lt,_,_=apply_x(x[800:],m);hv,_=data(x[600:800]);ht,_=data(x[800:]);n=s.shape[-1]-1;val=s[600:800,19:492,:n].reshape(-1,n);truth=s[800:,19:492,:n];maps={}
   for meth,a,b in [('current_X',hv[:,-1:],ht[:,-1:]),('joint_X_L',np.c_[hv[:,-1:],lv],np.c_[ht[:,-1:],lt]),('history20',hv,ht)]:
    p,mp=ridge(a,val,b);maps[meth]=mp;p=p.reshape(200,473,n)
    for i in range(200):
     for j,key in enumerate(['Y','Z'][:n]):
      err=p[i,:,j]-truth[i,:,j];rows.append(dict(group=cell['name'],seed=seed,trajectory=i,state=key,method=meth,MSE=float((err*err).mean()),NMSE=float((err*err).mean()/val[:,j].var()),corr=float(np.corrcoef(p[i,:,j],truth[i,:,j])[0,1])))
   (O/'structure'/cell['name']/str(seed)/'posthoc_joint_probe_maps.json').write_text(json.dumps(maps,indent=2))
 d=pd.DataFrame(rows);d.to_csv(O/'posthoc_joint_metrics.csv.gz',index=False);d.groupby(['group','state','method'])[['MSE','NMSE','corr']].mean().to_csv(O/'posthoc_joint_summary.csv');print(d.groupby(['group','state','method'])[['NMSE','corr']].mean().round(4).to_string())
if __name__=='__main__':main()

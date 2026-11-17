import json,hashlib
import numpy as np,pandas as pd
from config import *
from core import *
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();reg=json.loads((R/'REGISTERED.json').read_text());assert all(sha(R/f)==h for f,h in reg['files'].items());out={'pretest_hashes':'PASS'}
models=json.loads((O/'models.json').read_text())
for g in GROUPS:
 for seed in SEEDS:
  d=O/g['name']/str(seed);fr=json.loads((d/'FREEZE.json').read_text());assert sha(d/'frozen_X_model.json')==fr['model_sha256']
# Exact no-future property of encoder and matrix prediction invariance.
g=GROUPS[1];x,s=generate(g,960000+1000+101);mp=json.loads((O/g['name']/'101/frozen_X_model.json').read_text());xx=x[800:803].copy();changed=xx.copy();changed[:,251:]+=100;orig=encode(xx,mp);mod=encode(changed,mp);cut=251-19;err=float(np.max(abs(orig[:,:cut]-mod[:,:cut])));assert err==0;out['future_perturbation_encoder_error']=err
# Readout rotation equivalence, fit only validation truth.
lv=encode(x[600:800],mp,2).reshape(-1,2);lt=encode(x[800:],mp,2).reshape(-1,2);val=s[600:800,19:500,:2].reshape(-1,2);q=np.array([[np.sqrt(.5),-np.sqrt(.5)],[np.sqrt(.5),np.sqrt(.5)]]);a=np.c_[x[600:800,19:500].ravel(),lv];b=np.c_[x[800:,19:500].ravel(),lt];ar=np.c_[a[:,0],lv@q];br=np.c_[b[:,0],lt@q];p1=olsapply(b,olsfit(a,val));p2=olsapply(br,olsfit(ar,val));rec=float(np.max(abs(p1-p2)));assert rec<1e-10;out['joint_state_rotation_reconstruction_error']=rec
# Verify all forecast windows stop at t and targets are t+h.
w=hist(xx);assert np.array_equal(w[0,0],xx[0,:20]) and np.array_equal(w[0,-1],xx[0,472:492]);out['alignment']='PASS'
e=pd.DataFrame(json.loads((O/'edge_summary.json').read_text()));em=pd.read_csv(O/'edge_trajectory_metrics.csv.gz');sm=pd.read_csv(O/'state_metrics.csv.gz');assert len(e)==564 and len(em)==112800 and len(sm)==12000;keys=['group','seed','family','stage','control','horizon','source','target','lag'];assert not e.duplicated(keys).any();assert not em.duplicated(keys+['trajectory']).any();assert not sm.duplicated(['group','seed','trajectory','state','method']).any();assert em.groupby(keys).size().eq(200).all();assert all(not m['validated_direct_states'] for m in models)
sel=e[(e.family=='selected')&(((e.stage=='direct')&(e.control=='full_X20'))|((e.stage=='direction')&(e.control=='all_nodes_plus_full_X_covered')))];mx=float(sel.delta_mean.abs().max());assert mx<1e-10;assert not sel.test_supported.any();out['full_history_added_state_MSE_change_max']=mx
rot=json.loads((O/'rotations.json').read_text());out['rotation_forecast_max_error']=max(v['forecast_max_difference'] for v in rot);assert out['rotation_forecast_max_error']<1e-10;out['models']=len(models);out['edge_summaries']=len(e);out['edge_trajectory_rows']=len(em);out['state_trajectory_rows']=len(sm);out['status']='PASS';(O/'audit.json').write_text(json.dumps(out,indent=2));print(out)

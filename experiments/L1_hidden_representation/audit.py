import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from run import Net,design
R=Path(__file__).parent;O=R/'results';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();reg=json.loads((R/'REGISTERED.json').read_text());assert all(sha(R/f)==h for f,h in reg['files'].items());out={}
for seed in [101,202,303]:
 d=O/f'seed{seed}';f=json.loads((d/'FREEZE.json').read_text());assert sha(d/'frozen.pt')==f['checkpoint_sha256'];assert sha(R/'run.py')==f['run_sha256'];x=np.load(d/'X_only.npz')['X'];truth=np.load(d/'sealed_truth.npz');assert x.shape==(1000,500);w,y=design(x);assert w.shape==(480000,20);assert np.array_equal(w[0],x[0,:20]);assert y[0]==x[0,20];assert np.array_equal(w[480],x[1,:20]);ck=torch.load(d/'frozen.pt',weights_only=False);m=Net();m.load_state_dict(ck['state']);m.eval();t=100;xx=(x[:1]-ck['X_mean'])/ck['X_sd'];changed=xx.copy();changed[:,t+1:]+=100;wx,_=design(xx);wc,_=design(changed)
 with torch.no_grad():
  e=float((m.encoder(torch.from_numpy(wx[:t-18]))-m.encoder(torch.from_numpy(wc[:t-18]))).abs().max())
 assert e==0;Y=truth['Y'];Z=truth['Z'];innov=np.stack([(Y[:,1:]-.8*Y[:,:-1]).ravel(),(Z[:,1:]-.7*Z[:,:-1]-.5*Y[:,:-1]).ravel(),(x[:,1:]-.5*x[:,:-1]-.8*Z[:,:-1]).ravel()],1);v=innov.var(0);assert np.max(abs(v-.25))<.003;cc=np.corrcoef(innov.T);assert np.max(abs(cc-np.eye(3)))<.008
 out[str(seed)]=dict(checkpoint_hash='PASS',future_perturbation_encoder_difference=e,innovation_variances=v.tolist(),innovation_max_crosscorr=float(np.max(abs(cc-np.eye(3)))))
d=pd.read_csv(O/'trajectory_metrics.csv.gz');assert len(d)==10800;assert not d.duplicated(['seed','trajectory','task','method']).any();out['records']=len(d);out['status']='PASS';(O/'audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

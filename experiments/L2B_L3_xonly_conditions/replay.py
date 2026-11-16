"""Recreate a complete scan case from archived seeds/parameters. No training required."""
import argparse,json
import numpy as np,torch
from config import R,O,CELLS
from core import generate,plant,kalman,windows,apply
from encoder import Net,encode
torch.set_num_threads(2)
p=argparse.ArgumentParser();p.add_argument('--cell',required=True,choices=[c['name'] for c in CELLS]);p.add_argument('--seed',type=int,choices=[101,202,303],default=101);p.add_argument('--out',default='replayed_case.npz');a=p.parse_args();cell=next(c for c in CELLS if c['name']==a.cell);dr=O/'scan'/a.cell/str(a.seed)
if a.cell=='base':
 x=np.load(R/'inputs'/str(a.seed)/'X_only.npz')['X'];t=np.load(R/'inputs'/str(a.seed)/'sealed_truth.npz');states=np.stack([t['Y'],t['Z'],x],2)
else:x,states=generate(cell['params'],a.seed)
ck=torch.load(dr/'matched.pt',weights_only=False);m=Net();m.load_state_dict(ck['state']);m.eval();h=windows((x[800:]-np.float32(ck['X_mean']))/np.float32(ck['X_sd']));l=encode(m,h);mp=json.loads((dr/'maps.json').read_text());pred=apply(l,mp['L']).reshape(200,480,2);A,Q,obs=plant(**cell['params']);mf,ms,pf,ps=kalman(x[800:],A,Q,obs);np.savez_compressed(a.out,X=x[800:],truth=states[800:],L=l.reshape(200,480,2),matched_L_estimate=pred,KF=mf,RTS=ms,filter_cov=pf,smooth_cov=ps);print(a.out)

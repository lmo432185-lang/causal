"""Reproduce all test estimates for a registered group/seed, no refitting."""
import argparse,json
import numpy as np
from config import *
from core import *
p=argparse.ArgumentParser();p.add_argument('--group',choices=[g['name'] for g in GROUPS],required=True);p.add_argument('--seed',type=int,choices=SEEDS,default=101);p.add_argument('--out',default='replayed_estimates.npz');a=p.parse_args();gi=next(i for i,g in enumerate(GROUPS) if g['name']==a.group);x,truth=generate(GROUPS[gi],960000+1000*gi+a.seed);x=x[800:];d=O/a.group/str(a.seed);m=json.loads((d/'frozen_X_model.json').read_text());L=encode(x,m);k=m['rank'];maps=json.loads((d/'diagnostic_probes.json').read_text());features=dict(L_only=L.reshape(200*481,k),current_X=x[:,19:500].reshape(-1,1),joint_X_L=np.concatenate([x[:,19:500,None],L],2).reshape(200*481,k+1),history20=hist(x,19,500).reshape(-1,20));pred={key:olsapply(v,maps[key]).reshape(200,481,-1) for key,v in features.items()};np.savez_compressed(a.out,X=x,truth=truth[800:,19:500,:-1],L=L,forecast=forecast(x,m).reshape(200,473,4),**pred);print(a.out)

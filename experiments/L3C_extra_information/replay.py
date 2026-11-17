"""Recreate all 200 physical test trajectories for one registered case."""
import argparse,numpy as np
from config import *
from physics import simulate,waveform
from run import data_seed
p=argparse.ArgumentParser();p.add_argument('--pool',type=int,choices=[0,1,2,3,4],required=True);p.add_argument('--truth',type=int,default=0);p.add_argument('--seed',type=int,choices=SEEDS,default=101);p.add_argument('--noise',type=float,choices=NOISE,default=.25);p.add_argument('--info',choices=INFOS+['C3_weak_qY_X'],default='C0_X');p.add_argument('--out',default='replayed_case.npz');a=p.parse_args()
if a.pool==0:models=PRIMARY
elif a.pool in [1,2,3]:models=PAIRS[a.pool-1]['models']
else:
 d=PAIRS[0]['models'][0][1].copy();d['qy']=.25;d['beta1']=float(.8*np.sqrt(2));models=[('direct_original',PAIRS[0]['models'][0][1]),('indirect',PAIRS[0]['models'][1][1]),('direct_rescaled',d)]
assert 0<=a.truth<len(models);obs,states=simulate(models[a.truth][1],data_seed(a.pool,a.truth,a.seed),a.noise,a.info,N);np.savez_compressed(a.out,observations=obs,physical_states=states,registered_input=waveform());print(a.out)

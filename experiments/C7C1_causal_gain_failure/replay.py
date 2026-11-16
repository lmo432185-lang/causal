import argparse
import numpy as np
from config import *
from physics import simulate
p=argparse.ArgumentParser();p.add_argument('--noise-index',type=int,default=1,choices=range(3));p.add_argument('--plant',type=int,default=1,choices=range(4));p.add_argument('--cohort',type=int,default=0,choices=COHORTS);p.add_argument('--policy',choices=POLICIES,default='square');p.add_argument('--mode',choices=PRIMARY_MODES+SECONDARY_MODES,default='gradual');p.add_argument('--output',default='replayed_inputs.npz');a=p.parse_args();seed=seed_for('test',a.noise_index,a.plant,co=a.cohort);d=simulate(N,SIGMAS[a.noise_index],KTEST[a.plant],a.policy,seed,a.mode);np.savez_compressed(a.output,**d);print(a.output)

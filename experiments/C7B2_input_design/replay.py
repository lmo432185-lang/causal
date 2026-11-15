import argparse
import numpy as np
from config import *
from physics import *
p=argparse.ArgumentParser();p.add_argument('--program',default=REFERENCE,choices=IDS);p.add_argument('--split',choices=['validation','test','stress'],default='test');p.add_argument('--noise-index',type=int,default=1,choices=range(3));p.add_argument('--cohort',type=int,default=0,choices=COHORTS);p.add_argument('--tau',type=float,default=.5);p.add_argument('--output',default='replayed_inputs.npz');a=p.parse_args();n=NVAL if a.split=='validation' else N;seed=SEEDS['validation'][a.noise_index] if a.split=='validation' else SEEDS[a.split][a.cohort][a.noise_index];I=command(n,a.program,seed);data={str(t):simulate(I,SIGMAS[a.noise_index],np.full(n,a.tau),t,seed) for t in [0,1]};np.savez_compressed(a.output,**{key+'_'+f:vv for key,dd in data.items() for f,vv in dd.items()});print(a.output)

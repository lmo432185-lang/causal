import argparse
import numpy as np
from config import *
from physics import *
p=argparse.ArgumentParser();p.add_argument('--noise-index',type=int,default=1,choices=range(3));p.add_argument('--cohort',type=int,default=0,choices=COHORTS);p.add_argument('--kappa',type=float,default=.5,choices=KTEST);p.add_argument('--input',default='I2_pulse',choices=INPUTS);p.add_argument('--output',default='replayed_inputs.npz');a=p.parse_args();seed=SEEDS['test'][a.cohort][a.noise_index];commands=make_commands(N,seed);I=commands['blank'] if a.input=='I0_passive' else commands[a.input];data={str(truth):simulate(I,SIGMAS[a.noise_index],np.full(N,a.kappa),truth,seed,passive=a.input=='I0_passive') for truth in [0,1]};np.savez_compressed(a.output,**{key+'_'+k:v for key,d in data.items() for k,v in d.items()});print(a.output)

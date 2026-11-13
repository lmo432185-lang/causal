"""Rebuild complete paired test inputs from protocol, including original batch draw size."""
import argparse
import numpy as np
from config import *
from generate import draw,equivalent_pair
p=argparse.ArgumentParser();p.add_argument('--noise-index',type=int,default=1,choices=range(3));p.add_argument('--cohort',type=int,default=0,choices=COHORTS);p.add_argument('--kappa',type=float,default=.5,choices=KTEST);p.add_argument('--mode',choices=['passive','intervention','equivalent'],default='passive');p.add_argument('--output',default='replayed_inputs.npz');args=p.parse_args()
ni=args.noise_index;co=args.cohort;seed=SEEDS[args.mode][co][ni]
if args.mode=='equivalent':
 a,b=equivalent_pair(N,SIGMAS[ni],args.kappa,seed);data={'H1':a,'H2':b}
else:data=draw(N,SIGMAS[ni],args.kappa,seed,mode='active' if args.mode=='intervention' else 'passive')
np.savez_compressed(args.output,**{m+'_'+k:v for m,d in data.items() for k,v in d.items() if isinstance(v,np.ndarray)})
print(args.output)

import argparse
import numpy as np
from config import *
from generate import draw,equivalent_pair
p=argparse.ArgumentParser();p.add_argument('--noise-index',type=int,default=1,choices=range(3));p.add_argument('--cohort',type=int,default=0,choices=COHORTS);p.add_argument('--kappa',type=float,default=.5,choices=KTEST);p.add_argument('--family',choices=['passive','equivalent'],default='passive');p.add_argument('--output',default='replayed_inputs.npz');a=p.parse_args();ni=a.noise_index;seed=SEEDS[a.family][a.cohort][ni]
if a.family=='passive':data=draw(N,SIGMAS[ni],a.kappa,seed)
else:data=dict(zip(['H1','H2'],equivalent_pair(N,SIGMAS[ni],a.kappa,seed)))
np.savez_compressed(a.output,**{m+'_'+k:v for m,d in data.items() for k,v in d.items() if isinstance(v,np.ndarray)});print(a.output)

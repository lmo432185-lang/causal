"""Recreate a main test cell. Truth is returned for evaluation, never to inference."""
import argparse,numpy as np
from config import *
from physics import simulate
p=argparse.ArgumentParser();p.add_argument('--cell',type=int,required=True);p.add_argument('--cohort',type=int,default=0);p.add_argument('--out',default='replayed.npz');a=p.parse_args();c=CELLS[a.cell];params={k:c[k] for k in ['kind','a','c','alpha','beta','beta1']};params['sigma']=SIGMAS[c['ni']];d=simulate(params,N,cell_seed(c['cell'],a.cohort),rho=c['rho']);start=0 if c['startup'] else NBURN;np.savez_compressed(a.out,**{k:d[k][:,start:start+NOBS] for k in ['x','Y','Z','W']});print(a.out)

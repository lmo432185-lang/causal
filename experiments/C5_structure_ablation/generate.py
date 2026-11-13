from config import *
from source_A import simulate

def normal(n,sigma,seed):return simulate(n,sigma,1,0,seed,False)['x']
def event(gi,tau,strength,cohort):return simulate(N,SIGMAS[gi],tau,strength,SEEDS['events'][cohort][gi])

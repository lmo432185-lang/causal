from source_A import simulate
from source_C0 import mixture
from config import *

def null_data(n,sigma,seed):return simulate(n,sigma,1.,0.,seed,False)['x']
def event_data(sigma,tau,strength,gi,cohort):return simulate(N_EVENT,sigma,tau,strength,SEEDS['events'][cohort][gi])
def validation_X(sigma,gi):return mixture(N_VALID,sigma,SEEDS['prediction_validation'][gi])

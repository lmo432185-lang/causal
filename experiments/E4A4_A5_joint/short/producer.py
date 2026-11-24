"""Private generator and exact observational-law certificates. Never imported by learner."""
import json,hashlib
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from settings import ROOT,TMAX,SEEDS
SYSTEMS=json.loads((ROOT/'SYSTEMS.json').read_text())

def matrices(meta):
 if meta['family']=='D':
  r=meta['radius'];w=np.pi*meta['frequency_pi'];F=np.array([[2*r*np.cos(w),-r*r],[1.,0.]]);Q=np.diag([1.,0.]);P=solve_discrete_lyapunov(F,Q);Q/=P[0,0];C=np.array([1.,0.])
 else:
  poles=np.array(meta['poles']);v=np.array(meta['variance']);F=np.diag(poles);q=v*(1-poles*poles);Q=np.diag(q);C=np.ones(len(poles))
  if len(poles)==2:Q[0,1]=Q[1,0]=meta.get('rho',0)*np.sqrt(q.prod())
 return F,Q,C,meta['R']

def raw_generate(index,seed,stage,count,truth=False):
 meta=SYSTEMS[index];F,Q,C,R=matrices(meta);P=solve_discrete_lyapunov(F,Q);rng=np.random.default_rng(np.random.SeedSequence([20261123,440,index,seed,stage]));z=rng.multivariate_normal(np.zeros(len(F)),P,size=count);w,V=np.linalg.eigh(Q);L=V*np.sqrt(np.maximum(w,0));x=np.empty((count,TMAX));s=np.empty((count,TMAX,len(F))) if truth else None
 for t in range(TMAX):
  if t:z=z@F.T+rng.normal(size=z.shape)@L.T
  x[:,t]=z@C+rng.normal(size=count)*np.sqrt(R)
  if truth:s[:,t]=z
 if truth:return x,s
 return x

def jobs():return [dict(case=m['id']+'_'+str(seed),index=i,seed=seed) for i,m in enumerate(SYSTEMS) for seed in SEEDS]
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def arma_alias(model):
 a,b=np.diag(model['F']);qu,qv=np.diag(model['Q']);g0=qu*(1+b*b)+qv*(1+a*a);g1=-(b*qu+a*qv);v=(g0+np.sqrt(max(0,g0*g0-4*g1*g1)))/2;theta=g1/v;G=np.array([1.,theta]);F=np.array([[a+b,1.],[-a*b,0.]]);Q=v*np.outer(G,G)
 return dict(F=F,Q=Q,C=np.array([1.,0.]),a=0.,R=model['R'],P=solve_discrete_lyapunov(F,Q),scale=model.get('scale',1.),kind='arma21_alias',factor_count=1,state_dimension=2)

# Prefix training and confirmation; common full-length held-out tests.
def generate(index,seed,stage,count,truth=False):
 value=raw_generate(index,seed,stage,count,truth=truth)
 if stage in [0,10]:
  if truth:return value[0][:,:1000].copy(),value[1][:,:1000].copy()
  return value[:,:1000].copy()
 return value

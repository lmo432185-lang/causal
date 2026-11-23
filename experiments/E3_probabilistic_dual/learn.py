"""Only X arrays passed here; no simulation metadata, hidden states or test access."""
import time,numpy as np
from settings import *
from ssm import *

def fit_bank(x,v,route,seed):
 scale=max(float(np.std(x)),1e-6);xx=x/scale;models=[];proposals=[];scores=[]
 for k in range(KMAX+1):
  tic=time.process_time()
  if route=='recursive' and k>=2:init,proposal=recursive(xx,models[-1],k,seed+k);proposals.append(proposal)
  else:init=initial(xx,k)
  m=em(xx,init);m['scale']=scale;m['K']=k;m['cpu']=time.process_time()-tic;models.append(m);scores.append(selection_score(v,m))
 return dict(models=models,proposals=proposals,selection_scores=np.array(scores))

def initial_rank(bank,witness):
 scores=(bank['selection_scores']+witness['selection_scores'])/2;means=scores.mean(1)+.001*np.arange(KMAX+1);best=int(np.argmin(means));se=float(scores[best].std(ddof=1)/np.sqrt(scores.shape[1]));rank=int(np.flatnonzero(means<=means[best]+se)[0]);return rank

def fit_all(xr,xd,v,n,seed):
 tic=time.process_time();result={}
 result['direct']=fit_bank(xr[:n],v,'direct',seed+100000)
 result['witness']=fit_bank(xd[:n],v,'direct',seed+200000)
 result['recursive']=fit_bank(xr[:n],v,'recursive',seed+300000)
 result['rank_direct']=initial_rank(result['direct'],result['witness']);result['rank_recursive']=initial_rank(result['recursive'],result['witness']);result['n']=n;result['cpu']=time.process_time()-tic;return result

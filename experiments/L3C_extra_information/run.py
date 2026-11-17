from pathlib import Path
import hashlib,json,time
import numpy as np,pandas as pd
from scipy.special import softmax
from config import *
from physics import *

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def data_seed(pool,truth,seed):return int(np.random.SeedSequence([940000,pool,truth,seed]).generate_state(1)[0])

def evaluate(models,allowed,pool,truth_indices,infos,rows,examples):
 for truth_idx in truth_indices:
  name,p=models[truth_idx]
  for seed in SEEDS:
   ds=data_seed(pool,truth_idx,seed)
   for sigma in NOISE:
    for info in infos:
     # C3 uses X with no intervention, and only changes the admissible candidate set.
     obs,states=simulate(p,ds,sigma,info,N);scores={k:[] for k in PREFIXES}
     for candidate,pp in models:
      ll=log_likelihood(obs,pp,sigma,info,PREFIXES)
      for length in PREFIXES:scores[length].append(ll[length])
     mask=np.asarray(allowed,dtype=bool) if info in ['C3_constraints_X','C3_weak_qY_X'] else np.ones(len(models),bool)
     for length in PREFIXES:
      ll=np.array(scores[length]).T;ll[:,~mask]=-np.inf;post=softmax(ll,axis=1);pred=post.argmax(1);confidence=post.max(1);declare=confidence>=CONF;prior_valid=bool(mask[truth_idx])
      for i in range(N):rows.append(dict(pool=pool,truth=name,truth_idx=truth_idx,seed=seed,trajectory=i,noise=sigma,info=info,length=length,candidate_count=len(models),admissible_count=int(mask.sum()),prediction=models[pred[i]][0],posterior=confidence[i],declared=int(declare[i]),correct_declared=int(declare[i] and pred[i]==truth_idx),wrong_declared=int(declare[i] and pred[i]!=truth_idx),refused=int(not declare[i]),forced_correct=int(pred[i]==truth_idx),prior_valid=prior_valid))
     if seed==SEEDS[0] and sigma==.25:
      key=f'pool{pool}_truth{truth_idx}_{info}';examples[key+'_observed']=obs[:2];examples[key+'_states']=states[:2];examples[key+'_posterior']=post[:2]
  print('POOL',pool,name,'complete',flush=True)

def main():
 (R/'PROTOCOL.json').write_text(json.dumps(PROTOCOL,indent=2));files=['config.py','physics.py','run.py','PROTOCOL.json'];(R/'REGISTERED.json').write_text(json.dumps(dict(time=time.time(),files={f:sha(R/f) for f in files}),indent=2));rows=[];examples={};proofs=[]
 # Pairwise law differences computed before empirical simulations, not chosen by test performance.
 for pair in PAIRS:
  for info in INFOS:
   p0,p1=[m[1] for m in pair['models']];k0=kernel(p0,info,.25);k1=kernel(p1,info,.25);m0=controlled_mean(p0,info);m1=controlled_mean(p1,info);cd=float(np.max(abs(k0-k1))/max(np.max(abs(k0)),1e-12));md=float(np.max(abs(m0-m1)));proofs.append(dict(pair=pair['name'],info=info,covariance_relative_difference=cd,controlled_mean_max_difference=md,observationally_equivalent=bool(cd<1e-10 and md<1e-10),prior_allowed=pair['prior_allowed'] if info=='C3_constraints_X' else [True,True]))
 # A matched Z marginal in pair1 prevents trivial discrimination by sensor variance alone.
 p0,p1=[m[1] for m in PAIRS[0]['models']];z0=kernel(p0,'C1_XZ',.25)[:,1,1];z1=kernel(p1,'C1_XZ',.25)[:,1,1];assert np.max(abs(z0-z1))<1e-10
 (O/'pretest_law_comparisons.json').write_text(json.dumps(dict(rows=proofs,pair1_Z_marginal_covariance_max_difference=float(np.max(abs(z0-z1)))),indent=2))
 evaluate(PRIMARY,[True]*3,0,list(range(3)),INFOS,rows,examples)
 for pi,pair in enumerate(PAIRS,1):evaluate(pair['models'],pair['prior_allowed'],pi,[0,1],INFOS,rows,examples)
 # Weaker physical calibration: fixing qY does not help when direct gain remains free.
 direct=PAIRS[0]['models'][0][1].copy();direct['qy']=.25;direct['beta1']=float(.8*np.sqrt(2));pool=[('direct_original',PAIRS[0]['models'][0][1]),('indirect',PAIRS[0]['models'][1][1]),('direct_rescaled',direct)];assert np.max(abs(kernel(pool[1][1],'C0_X',.25)-kernel(pool[2][1],'C0_X',.25)))<1e-10;evaluate(pool,[False,True,True],4,[1,2],['C3_weak_qY_X'],rows,examples)
 pd.DataFrame(rows).to_csv(O/'trajectory_metrics.csv.gz',index=False);np.savez_compressed(O/'examples.npz',**examples);print('ALL COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()

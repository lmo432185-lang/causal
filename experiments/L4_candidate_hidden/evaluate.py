"""Final evaluation only, gated by the complete X-only freeze marker."""
import json,time,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import numpy as np
import pandas as pd
from settings import ROOT,OUT,MAXITER
from ssm import initialize,estep,mstep,restore,serialize

def fit_oracle(y,kind,delay,seed):
 # No true parameters: all observed variables standardized using training data.
 p=initialize(kind,delay,seed,oracle=True);trace=[];old=None;last=None;reason='iteration_cap';total=y.shape[0]*y.shape[1]*3
 for it in range(MAXITER):
  e=estep(y,p);ll=e['loglik'];trace.append(float(ll))
  if last is not None and ll<last-1e-6*total:p=old;trace.pop();reason='monotonicity_safeguard';break
  if last is not None and it>=8 and (ll-last)/total<3e-5:reason='likelihood_tolerance';break
  if it==MAXITER-1:break
  old=p;last=ll;p=mstep(e,p)
 return p,dict(trace=trace,stop=reason,iterations=len(trace))

def alignment(x,y,z):
 mu=x.mean(0);sd=np.maximum(x.std(0),1e-8);a=np.c_[np.ones(len(x)),(x-mu)/sd];b=np.c_[np.ones(len(z)),(z-mu)/sd];coef=np.linalg.pinv(a.T@a,rcond=1e-10)@(a.T@y);return b@coef,dict(mean=mu.tolist(),sd=sd.tolist(),coef=coef.tolist(),feature_rank=int(np.linalg.matrix_rank(a)))

def evaluate_job(meta):
 started=time.time();case=meta['case'];d=OUT/case;publicX=np.load(ROOT/'data'/case/'X.npy');private=np.load(ROOT/'data'/case/'sealed_states.npz')['physical'];frozen=json.loads((d/'selected_parameters.json').read_text());xc=(publicX-frozen['mu'])/frozen['sd'];modelname=json.loads((d/'X_only_decisions.json').read_text())['selected_by_validation'];states=[];maps={};examples={}
 for family in ['selected','single']:
  p=restore(frozen['models'][modelname if family=='selected' else 'single']);v=estep(xc[600:800],p);t=estep(xc[800:],p);dim=1 if family=='single' else 2;targetv=private[600:800,100:,:2].reshape(-1,2);target=private[800:,100:,:2];vvar=targetv.var(0)
  for mode,key in [('filter','filtered'),('smoother_future','smoothed')]:
   a=v[key][:,100:,:dim].reshape(-1,dim);z=t[key][:,100:,:dim].reshape(-1,dim);pred,mp=alignment(a,targetv,z);pred=pred.reshape(200,900,2);maps[family+'_'+mode]=mp
   for i in range(200):
    for j,name in enumerate(['Y','Z']):
     true=target[i,:,j];pr=pred[i,:,j];mse=float(np.mean((pr-true)**2));corr=float(np.corrcoef(pr,true)[0,1]) if np.std(pr)>1e-12 else 0.;states.append(dict(case=case,family=family,mode=mode,trajectory=i,state=name,MSE=mse,NMSE=mse/float(vvar[j]),corr=corr,alignment_feature_rank=mp['feature_rank']))
   examples[family+'_'+mode]=pred[:3]
  examples[family+'_filtered_latent']=t['filtered'][:3,100:,:dim];examples[family+'_smoothed_latent']=t['smoothed'][:3,100:,:dim]
 # Evaluation-only same-X information bound: true parameters supplied here, never to fitter.
 from generate import system
 A0,Q0,P0=system(meta);refp=dict(A=A0,Q=Q0,P0=P0,r=meta['params']['obs']**2,kind=meta['kind'],delay=meta['params']['delay'],oracle=False);kr=estep(publicX[800:],refp);tv=private[600:800,100:,:2].reshape(-1,2);vv=tv.var(0);target=private[800:,100:,:2]
 for mode,key in [('filter','filtered'),('smoother_future','smoothed')]:
  pp=kr[key][:,100:,:2]
  for i in range(200):
   for j,name in enumerate(['Y','Z']):
    mse=float(np.mean((pp[i,:,j]-target[i,:,j])**2));states.append(dict(case=case,family='known_param_X_reference',mode=mode,trajectory=i,state=name,MSE=mse,NMSE=mse/float(vv[j]),corr=float(np.corrcoef(pp[i,:,j],target[i,:,j])[0,1]),alignment_feature_rank=3))
  examples['known_param_X_'+mode]=pp[:3]
 (d/'known_parameter_X_reference.json').write_text(json.dumps(dict(input='X only, true parameters known only to final reference',filter_posterior_variance=kr['cov_filter'][100:,:2,:2].mean(0).tolist(),smoother_posterior_variance=kr['cov_smooth'][100:,:2,:2].mean(0).tolist()),indent=2))
 (d/'evaluation_alignment_maps.json').write_text(json.dumps(maps,indent=2));pd.DataFrame(states).to_csv(d/'hidden_state_metrics.csv.gz',index=False);np.savez_compressed(d/'evaluation_examples.npz',X=publicX[800:803],truth=private[800:803,100:,:2],**examples)
 # True-YZ oracle is isolated here, after all X-only predictions are immutable.
 oracle=np.concatenate([private[:,:,:2].astype(float),publicX[:,:,None]],2);mu=oracle[:600].mean((0,1));sd=oracle[:600].std((0,1));oracle=(oracle-mu)/sd;tr,va,te=oracle[:600],oracle[600:800],oracle[800:];models={};logrows=[]
 for ki,kind in enumerate(['G1','G2','G3','G4']):
  tries=[]
  for delay in ([1] if kind=='G1' else [1,2]):
   p,log=fit_oracle(tr,kind,delay,983000+meta['seed']+100*int(case[4:7])+10*ki+delay);ll=estep(va,p,stats=False)['trajectory_ll'];k=10 if kind=='G3' else 9;score=float(ll.sum()-.5*k*np.log(200*900)-(np.log(2) if kind!='G1' else 0));tries.append((score,p));logrows.append(dict(kind=kind,delay=delay,validation_score=score,fit=log,parameters=serialize(p)))
  models[kind]=max(tries,key=lambda q:q[0])[1]
 kinds=['G1','G2','G3','G4'];ll=np.stack([estep(te,models[k],stats=False)['trajectory_ll'] for k in kinds],1);pen=np.array([9,9,10,9])*.5*np.log(900)+np.array([0,1,1,1])*np.log(2);from scipy.special import softmax
 post=softmax(ll-pen,axis=1);pred=post.argmax(1);(d/'oracle_results.json').write_text(json.dumps(dict(models={k:serialize(v) for k,v in models.items()},mu=mu.tolist(),sd=sd.tolist(),predictions=[kinds[j] for j in pred],posterior=post.tolist(),chosen_delays={k:int(v['delay']) for k,v in models.items()},fits=logrows),indent=2));print(case,'EVALUATED',round(time.time()-started,1),'sec',flush=True);return case

if __name__=='__main__':
 gate=ROOT/'ALL_X_ONLY_FROZEN.json'
 while not gate.exists() or not (ROOT/'ALL_SUPPLEMENTARY_AUDIT_FROZEN.json').exists():time.sleep(5)
 frozen=json.loads(gate.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert all(sha(ROOT/f)==h for f,h in frozen['files'].items());jobs=json.loads((ROOT/'data/sealed_manifest.json').read_text());print('GLOBAL X FREEZE VERIFIED; opening truth for evaluation',flush=True)
 with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool:
  for i,f in enumerate(as_completed([pool.submit(evaluate_job,j) for j in jobs]),1):f.result();print('EVAL PROGRESS',i,'/',len(jobs),flush=True)
 assert all(sha(ROOT/f)==h for f,h in frozen['files'].items());(ROOT/'FINAL_EVALUATION_DONE.json').write_text(json.dumps(dict(time=time.time(),case_count=len(jobs),X_only_hashes_unchanged=True),indent=2));print('FINAL EVALUATION COMPLETE',flush=True)

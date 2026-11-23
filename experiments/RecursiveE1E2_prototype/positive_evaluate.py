"""Final unseen X scoring; latent truth read only after all X predictions freeze."""
import json,time,hashlib
import numpy as np,pandas as pd
from scipy.special import ndtr
from scipy.stats import chi2,norm
from settings import *
from core import *
from fit import features_for
from io_model import load_model

def score(head,features,target):
 mean=predict(head,features);cov=head['cov'];sd=np.sqrt(np.diag(cov));r=target-mean;z=r/sd;pit=ndtr(z);iv=np.linalg.inv(cov);mah=np.einsum('nti,ij,ntj->nt',r,iv,r);D=target.shape[-1];N=len(target);out=dict(NLL=nll_series(head,features,target).mean(1),CRPS=(sd*(z*(2*pit-1)+2*np.exp(-z*z/2)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))).mean((1,2)),MSE=(r*r).mean((1,2)),joint_cover90=(mah<=chi2.ppf(.9,D)).mean(1),marginal_cover90=(abs(z)<=norm.ppf(.95)).mean((1,2)),width90=np.repeat(float(np.mean(2*norm.ppf(.95)*sd)),N))
 for h in [1,2,5,10]:out[f'NLL_h{h}']=np.mean(.5*(np.log(2*np.pi*cov[h-1,h-1])+z[:,:,h-1]**2),1);out[f'cover90_h{h}']=(abs(z[:,:,h-1])<=norm.ppf(.95)).mean(1)
 return out,mean,mah

def main():
 mark=json.loads((ROOT/'POSITIVE_DECISIONS_FROZEN.json').read_text());assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in mark['files'].items());reg=json.loads((ROOT/'POSITIVE_EVALUATION_REGISTERED.json').read_text());assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in reg['files'].items());dec=pd.read_csv(OUT/'decisions_positive.csv').set_index('case');rows=[];checks=[];weights={};test_scores={}
 for job in json.loads((ROOT/'positive_public_jobs.json').read_text()):
  case=job['case'];model=load_model(OUT/case/'model.json');x=np.load(DATA/case/'test_X.npy');q=feature_bundle(x,model['short'],model['long'],model['recursive_encoder'],model['B'],model['match']);f=features_for(q,model);rng=np.random.default_rng(job['seed']+600000);f['random']=np.concatenate([f['initial'],rng.normal(size=(*f['initial'].shape[:2],1))],-1);examples=dict(time=np.arange(START,END),X=x[0],W1=q['W1'][0,:,0],Wy=q['A'][0,:,model['candidate_index']]);ss={}
  for name,feature in f.items():
   if name.startswith('W1_'):continue
   metrics,mu,mah=score(model['heads'][name],feature,q['future']);ss[name]=metrics['NLL'];
   for i in range(len(x)):rows.append(dict(case=case,seed=job['seed'],method=name,trajectory=i,**{k:float(v[i]) for k,v in metrics.items()}))
   examples[name+'_mean']=mu[0];examples[name+'_sd']=np.sqrt(np.diag(model['heads'][name]['cov']));examples[name+'_nll_by_time']=nll_series(model['heads'][name],feature,q['future'])[0]
  operational='recursive' if bool(dec.loc[case,'accept_new_coordinate']) else 'initial' if bool(dec.loc[case,'initial_state_supported']) else 'current'
  for row in [r for r in rows if r['case']==case and r['method']==operational]:rows.append({**row,'method':'operational'})
  base=affine(f['initial']);ar=q['A'][:,:,model['candidate_index']:model['candidate_index']+1]-base@model['A_base_regression'];bw=(q['B'][:,:,:model['rank_B']]-base@model['B_base_regression'])@model['B_axis'];pc=float(np.corrcoef(ar.ravel(),bw.ravel())[0,1]) if model['rank_B'] else 0.;rawerr=float(np.max(abs(apply_raw_filter(x,model['raw_filter'])-q['A'][:,:,model['candidate_index']])));duperr=float(np.max(abs(predict(model['heads']['initial'],f['initial'])-predict(model['heads']['duplicate'],f['duplicate']))));checks.append(dict(case=case,partial_corr=abs(pc),conditional_residual_fraction=float(np.var(ar)/np.var(q['A'][:,:,model['candidate_index']])),raw_X_reexpression_error=rawerr,duplicate_prediction_error=duperr,operational_method=operational))
  np.savez_compressed(OUT/case/'test_predictions.npz',**examples);test_scores[case]=ss;weights[case]=model;print('TEST X SCORED',case,flush=True)
 pd.DataFrame(rows).to_csv(OUT/'trajectory_metrics_positive.csv.gz',index=False);pd.DataFrame(checks).to_csv(OUT/'test_checks_positive.csv',index=False);files={}
 for j in json.loads((ROOT/'positive_public_jobs.json').read_text()):
  p=OUT/j['case']/'test_predictions.npz';files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 for file in ['trajectory_metrics_positive.csv.gz','test_checks_positive.csv']:files['results/'+file]=hashlib.sha256((OUT/file).read_bytes()).hexdigest()
 (ROOT/'POSITIVE_TEST_PREDICTIONS_FROZEN.json').write_text(json.dumps(dict(time=time.time(),files=files,hidden_truth_not_loaded=True),indent=2))
 # Only now reveal complete simulation answers; never feed them back to encoders, ranks or heads.
 private=json.loads((DATA/'sealed_positive_manifest.json').read_text());labels={m['case']:m for m in private};hidden=[]
 for case,model in weights.items():
  meta=labels[case];target=np.load(DATA/case/'sealed_states.npz')['states'][-100:,START:END];x=np.load(DATA/case/'test_X.npy');q=feature_bundle(x,model['short'],model['long'],model['recursive_encoder'],model['B'],model['match']);w=q['A'][:,:,model['candidate_index']]
  for state,name in enumerate(meta['names']):
   for feature,value in [('W1',q['W1'][:,:,0]),('Wy',w)]:hidden.append(dict(case=case,feature=feature,true_hidden=name,corr=float(np.corrcoef(value.ravel(),target[:,:,state].ravel())[0,1]),scope='post-freeze descriptive correlation; no true-state matching or alignment fitted'))
 pd.DataFrame(hidden).to_csv(OUT/'hidden_correlations_positive.csv',index=False)
 # Cross-seed reproducibility on independent common X probes. Do not select or modify anything.
 stability=[]
 for index in [5,6]:
  x=np.load(DATA/f'probe_case{index:02}_X.npy');states={}
  for seed in SEEDS:
   model=weights[f'case{index:02}_{seed}'];q=feature_bundle(x,model['short'],model['long'],model['recursive_encoder'],model['B'],model['match']);z=np.concatenate([q['W1'],q['A'][:,:,model['candidate_index']:model['candidate_index']+1]],-1).reshape(-1,2);current=q['current'].reshape(-1,1);z=z-affine(current)@regfit(affine(current),z);states[seed]=z
  for i,s1 in enumerate(SEEDS):
   for s2 in SEEDS[i+1:]:
    a=states[s1];b=states[s2];aa=whitening(a.T@a/len(a));bb=whitening(b.T@b/len(b));sv=np.linalg.svd(aa@(a.T@b/len(a))@bb,compute_uv=False);stability.append(dict(system_index=index,seed1=s1,seed2=s2,W1_abs_corr=float(abs(np.corrcoef(a[:,0],b[:,0])[0,1])),subspace_min_CCA=float(sv[-1])))
 pd.DataFrame(stability).to_csv(OUT/'cross_seed_stability_positive.csv',index=False);(ROOT/'POSITIVE_EVALUATION_DONE.json').write_text(json.dumps(dict(time=time.time(),cases=6,trajectory_metric_rows=len(rows),full_hidden_truth_used_after_predictions_frozen=True),indent=2));print('ALL EVALUATED',len(rows))
if __name__=='__main__':main()

import json,hashlib,datetime
from pathlib import Path
import numpy as np,pandas as pd,torch
from kalman import A,Q,H,R,estimate,initial_covariance
from encoder import Net,encode,apply_map
ROOT=Path(__file__).parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True);torch.set_num_threads(2)
CONFIG=dict(name='L2-A known-model causal MMSE and retrospective smoothing diagnosis',source='unaltered L1 input archives, weights and validation probes',seeds=[101,202,303],split=[600,200,200],length=500,warmup=200,initial_covariance='exact 201 transitions from zero; no pre-retention X given to estimator',A=A.tolist(),Q=Q.tolist(),H=H.tolist(),R=R,evaluation_times=list(range(19,499)),filter='E[S_t | X_0,...,X_t], true model known; initial zero mean',smoother='E[S_t | X_0,...,X_499], true model known; not real time',baselines='unaltered L1 current-X and 20-history ridge maps plus frozen original encoder L and L-only validation map',extra='known-model 20-point filter diagnostic for equal information budget',no_training=True,known_model_prior='true system parameters used only by oracle references, not provided to original encoder',statistics='paired trajectories within each seed; no timepoint IID claims; seed-specific reporting',delay_innovation_floor=dict(Y=.25*(1+.8**2),Z=.25*(1+.5**2),joint=[[.41,.1],[.1,.3125]]))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2))
def main():
 dump(ROOT/'PROTOCOL.json',CONFIG);files=['kalman.py','encoder.py','run.py','PROTOCOL.json','SOURCE_MANIFEST.json'];dump(ROOT/'REGISTERED.json',dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),sha256={f:sha(ROOT/f) for f in files}));rows=[];variance=[];parity=[]
 old=pd.read_csv(ROOT/'inputs/L1_trajectory_metrics.csv.gz');idx=np.arange(19,499)
 for seed in CONFIG['seeds']:
  d=ROOT/'inputs'/f'seed{seed}';o=OUT/f'seed{seed}';o.mkdir(exist_ok=True);x=np.load(d/'X_only.npz')['X'];test=x[800:];ck=torch.load(d/'frozen.pt',weights_only=False);model=Net();model.load_state_dict(ck['state']);model.eval();maps=json.loads((d/'probe_maps.json').read_text());xx=(test-np.float32(ck['X_mean']))/np.float32(ck['X_sd']);hist=np.lib.stride_tricks.sliding_window_view(xx,20,axis=1)[:,:-1].copy().reshape(-1,20);l=encode(model,hist)
  methods={name:apply_map(feat,maps[key]).reshape(200,480,2) for name,key,feat in [('A1_current','X_current',hist[:,-1:]),('A2_history20','X_history',hist),('A3_original_L','L_only',l)]}
  # Means use X only. All evaluation truth is loaded after estimates are computed.
  mf,ms,pf,ps=estimate(test);methods['A4_Kalman_filter']=mf[:,idx,:2];methods['A5_RTS_smoother']=ms[:,idx,:2];windows=np.lib.stride_tricks.sliding_window_view(test,20,axis=1)[:,:-1].copy().reshape(-1,20);short,_,p20,_=estimate(windows,smooth=False);methods['D_known_model_history20']=short[:,-1,:2].reshape(200,480,2)
  truth=np.load(d/'sealed_truth.npz');target=np.stack([truth[k][800:,idx] for k in ['Y','Z']],axis=2);vvar=np.array([truth[k][600:800,19:499].var() for k in ['Y','Z']]);predsave=dict(truth=target,X=test,L=l.reshape(200,480,2),filter_full=mf,smooth_full=ms,filter_cov=pf,smooth_cov=ps,**methods);np.savez_compressed(o/'estimates.npz',**predsave)
  for meth,pr in methods.items():
   for i in range(200):
    for j,key in enumerate(['Y','Z']):
     err=pr[i,:,j]-target[i,:,j];mse=float(np.mean(err**2));post=pf[idx,j,j] if meth=='A4_Kalman_filter' else ps[idx,j,j] if meth=='A5_RTS_smoother' else np.repeat(p20[-1,j,j],480) if meth=='D_known_model_history20' else None
     rows.append(dict(seed=seed,trajectory=i,task=key,method=meth,MSE=mse,RMSE=float(np.sqrt(mse)),NMSE=float(mse/vvar[j]),corr=float(np.corrcoef(pr[i,:,j],target[i,:,j])[0,1]),posterior_variance=float(post.mean()) if post is not None else np.nan,posterior_95_coverage=float(np.mean(abs(err)<=1.95996398454*np.sqrt(post))) if post is not None else np.nan))
  for t in range(500):
   for j,key in enumerate(['Y','Z']):variance.append(dict(seed=seed,time=t,task=key,filter_variance=float(pf[t,j,j]),smooth_variance=float(ps[t,j,j]),window20_variance=float(p20[-1,j,j]) if t>=19 else np.nan))
  # Recompute old frozen-map metrics and verify these are genuinely L1 results.
  for meth,oldmeth in [('A1_current','X_current'),('A2_history20','X_history'),('A3_original_L','L_only')]:
   for key in ['Y','Z']:
    fresh=np.array([r['MSE'] for r in rows if r['seed']==seed and r['method']==meth and r['task']==key]);arch=old[(old.seed==seed)&(old.method==oldmeth)&(old.task==key)].sort_values('trajectory').MSE.to_numpy();dif=float(np.max(abs(fresh-arch)));assert dif<2e-6;parity.append(dict(seed=seed,method=meth,task=key,max_L1_metric_difference=dif))
  print('completed',seed,flush=True)
 pd.DataFrame(rows).to_csv(OUT/'trajectory_metrics.csv.gz',index=False);pd.DataFrame(variance).to_csv(OUT/'posterior_variance_by_time.csv.gz',index=False);dump(OUT/'L1_parity.json',parity);np.savez_compressed(OUT/'model_covariances.npz',initial=initial_covariance(),filter=pf,smoother=ps,window20=p20)
if __name__=='__main__':main()

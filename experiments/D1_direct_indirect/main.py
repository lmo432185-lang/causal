import os,json,time,hashlib
from datetime import datetime,timezone
import numpy as np,pandas as pd,torch
from config import *
from physics import *
from spectral import *
from states import filter_smooth
from representation import fit,encode,diagnostic_readout,Latent

def log(v):print(datetime.now(timezone.utc).isoformat(),v,flush=True)
def dump(p,x):p.write_text(json.dumps(x,indent=2))
def corr(a,b):
 if np.std(a)<1e-10 or np.std(b)<1e-10:return 0.
 return float(np.corrcoef(a,b)[0,1])

def train():
 learned=[]
 for ni,sigma in enumerate(SIGMAS):
  tr=random_pool(NTRAIN,sigma,SEEDS['train'][ni]);va=random_pool(NVAL,sigma,SEEDS['validation'][ni]);mu=float(tr['x'].mean());sd=float(tr['x'].std());qx=diffusion_from_x(tr['x']);models=[];readouts=[]
  for optseed in NET_SEEDS:
   model,score=fit(tr['x'],va['x'],optseed,mu,sd);readout=diagnostic_readout(model,va,mu,sd);torch.save(dict(state=model.state_dict(),mu=mu,sd=sd,readout={k:v.tolist() for k,v in readout.items()},validation_X_loss=score),OUT/f'latent_n{ni}_s{optseed}.pt');models.append(model);readouts.append(readout);log(f'Latent train noise={sigma},seed={optseed}')
  learned.append(dict(qx=qx,mu=mu,sd=sd));dump(OUT/'training.json',learned)
 return learned

def load_models(ni):
 models=[];probes=[]
 for seed in NET_SEEDS:
  ck=torch.load(OUT/f'latent_n{ni}_s{seed}.pt',weights_only=False);m=Latent();m.load_state_dict(ck['state']);models.append(m.eval());probes.append({k:np.asarray(v) for k,v in ck['readout'].items()})
 return models,probes

def run_cells(learned):
 files=['config.py','physics.py','spectral.py','states.py','representation.py','main.py','PROTOCOL.json','PROTOCOL_REGISTERED.json','results/training.json']+[f'results/latent_n{ni}_s{s}.pt' for ni in range(3) for s in NET_SEEDS];dump(OUT/'TEST_START.json',dict(time=datetime.now(timezone.utc).isoformat(),sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}));records=[];examples={}
 for ni in range(3):
  info=learned[ni];bank=library(info['qx'],NOBS);prefixbank=library(info['qx'],801);models,probes=load_models(ni);noise_cells=[c for c in CELLS if c['ni']==ni]
  for ci,cell in enumerate(noise_cells):
   params={key:cell[key] for key in ['kind','a','c','alpha','beta','beta1']};params['sigma']=SIGMAS[ni]
   for co in COHORTS:
    d=simulate(params,N,cell_seed(cell['cell'],co),rho=cell['rho']);start=0 if cell['startup'] else NBURN;x=d['x'][:,start:start+NOBS];Y=d['Y'][:,start:start+NOBS];W=d['W'][:,start:start+NOBS];res=infer(x,bank);pref=infer(x[:,:801],prefixbank);ids=np.arange(800,NOBS,20);zpred={key:[] for key in ['Y','W','Y_base','W_base']}
    for model,probe in zip(models,probes):
     z=encode(model,x,info['mu'],info['sd'],ids);design=np.c_[np.ones(len(x)*len(ids)),x[:,ids].ravel(),z.reshape(-1,2)];base=np.c_[np.ones(len(x)*len(ids)),x[:,ids].ravel()]
     for key in ['Y','W']:zpred[key].append((design@probe[key]).reshape(N,-1));zpred[key+'_base'].append((base@probe[key+'_base']).reshape(N,-1))
    truths={'Y':Y[:,ids],'W':W[:,ids]};conditional={key:np.mean(val,axis=0) for key,val in zpred.items()}
    for i in range(N):
     # Parameters/class from prefix only; the following forward state estimates never use future X.
     pred=int(pref['pred'][i]);theta=pref['theta'][i];mf,ms,_=filter_smooth(x[i],pred,theta);yf=mf[ids,0];ys=ms[ids,0];wf=mf[ids,1] if pred in [2,3] else np.zeros(len(ids));ws=ms[ids,1] if pred in [2,3] else np.zeros(len(ids));truth=CLASS.index(cell['kind']);theta_full=res['theta'][i];chosen=int(res['pred'][i]);declared=bool(res['declare'][i]);row=dict(**cell,cohort=co,trajectory=i,truth=truth,prediction=chosen,declared=int(declared),correct_declared=int(declared and chosen==truth),wrong_declared=int(declared and chosen!=truth),refused=int(not declared),posterior=float(res['p'][i,chosen]),profile_gap=float(res['gap'][i]),prefix_prediction=pred,prefix_declared=int(pref['declare'][i]),beta_direct_est=float(theta_full[2]),gamma_indirect_est=float(theta_full[3]),a_est=float(theta_full[0]),c_est=float(theta_full[1]),source_corr_filter=corr(yf,truths['Y'][i]),source_corr_smooth=corr(ys,truths['Y'][i]),source_NMSE_filter=float(np.mean((yf-truths['Y'][i])**2)/max(np.var(truths['Y'][i]),1e-10)),source_NMSE_smooth=float(np.mean((ys-truths['Y'][i])**2)/max(np.var(truths['Y'][i]),1e-10)),middle_corr_filter=corr(wf,truths['W'][i]),middle_corr_smooth=corr(ws,truths['W'][i]),wrong_middle=int(declared and truth==1 and chosen in [2,3]))
     for key in ['Y','W']:
      yp=conditional[key][i];base=conditional[key+'_base'][i];target=truths[key][i];row['L_'+key+'_corr']=corr(yp,target);row['L_'+key+'_incremental_R2']=float((np.mean((base-target)**2)-np.mean((yp-target)**2))/max(np.var(target),1e-10)) if np.var(target)>1e-10 else np.nan
     records.append(row)
     if co==0 and i==0 and cell['a']==1 and cell['c']==2 and cell['beta']==1.5 and cell['kind']=='indirect':
      examples[f'cell{cell["cell"]}_x']=x[i];examples[f'cell{cell["cell"]}_Y']=Y[i];examples[f'cell{cell["cell"]}_Yfilter']=mf[:,0];examples[f'cell{cell["cell"]}_Ysmooth']=ms[:,0]
   if (ci+1)%8==0:log(f'Main noise={SIGMAS[ni]},cells={ci+1}/{len(noise_cells)}')
  pd.DataFrame(records).to_csv(OUT/'structure_state_metrics.csv.gz',index=False)
 np.savez_compressed(OUT/'examples.npz',**examples)

if __name__=='__main__':
 t=time.time();learned=train();run_cells(learned);log(f'MAIN COMPLETE {time.time()-t:.1f}s')

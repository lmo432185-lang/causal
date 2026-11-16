"""X-only predictive representation; hidden truth opened after checkpoint freeze."""
import os,json,hashlib,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
ROOT=Path(__file__).parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
torch.set_num_threads(2)
CONFIG=dict(seeds=[101,202,303],trajectories_per_seed=1000,split=[600,200,200],length=500,burn=200,noise_std=[.5,.5,.5],history=20,latent=2,steps=4000,batch=512,lr=.001,validation_every=100,patience=8,noise='mutually independent Gaussian innovations',alignment='history ends at t; probe Y_t,Z_t; predict X_t+1',predictor_controls='ridge alpha=1e-3 in standardized features; linear heads; no test fitting')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,indent=2))
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.encoder=nn.Sequential(nn.Linear(20,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,2));self.head=nn.Linear(3,1)
 def forward(self,x):
  l=self.encoder(x);return self.head(torch.cat([x[:,-1:],l],1)),l

def generate(seed,d):
 rng=np.random.default_rng(seed);s=np.zeros((1000,3));all=[]
 for t in range(700):
  y,z,x=s.T;e=rng.normal(0,.5,s.shape);s=np.stack([.8*y,.7*z+.5*y,.5*x+.8*z],1)+e
  if t>=200:all.append(s.copy())
 a=np.stack(all,1).astype('float32');np.savez_compressed(d/'X_only.npz',X=a[:,:,2]);np.savez_compressed(d/'sealed_truth.npz',Y=a[:,:,0],Z=a[:,:,1])

def design(x):
 return np.lib.stride_tricks.sliding_window_view(x,20,axis=1)[:,:-1].copy().reshape(-1,20),x[:,20:].reshape(-1,1)
def encode(m,x):
 with torch.no_grad():
  p=[];l=[]
  for v in torch.from_numpy(x.astype('float32')).split(8192):
   a,b=m(v);p.append(a.numpy());l.append(b.numpy())
 return np.concatenate(p),np.concatenate(l)
def ridge(a,y,b):
 mu=a.mean(0);sd=a.std(0);sd=np.maximum(sd,1e-7);a=np.c_[np.ones(len(a)),(a-mu)/sd];b=np.c_[np.ones(len(b)),(b-mu)/sd];reg=np.eye(a.shape[1])*1e-3;reg[0,0]=0;coef=np.linalg.solve(a.T@a+reg,a.T@y);return b@coef,dict(mean=mu.tolist(),sd=sd.tolist(),coef=coef.tolist())
def main():
 save(ROOT/'PROTOCOL.json',CONFIG);save(ROOT/'REGISTERED.json',dict(time=time.time(),files={f:sha(ROOT/f) for f in ['run.py','PROTOCOL.json']}));metrics=[];corrs=[];models={}
 for seed in CONFIG['seeds']:
  start=time.time();d=OUT/f'seed{seed}';d.mkdir(exist_ok=True);generate(seed,d)
  # Only this file is opened while fitting/validating predictive representation.
  x=np.load(d/'X_only.npz')['X'];mu=x[:600].mean();sd=x[:600].std();xx=(x-mu)/sd
  tr,ty=design(xx[:600]);va,vy=design(xx[600:800]);te,ey=design(xx[800:]);torch.manual_seed(seed);rng=np.random.default_rng(seed+10000);m=Net();opt=torch.optim.Adam(m.parameters(),lr=.001);best=float('inf');stale=0;curve=[]
  tx=torch.from_numpy(tr);target=torch.from_numpy(ty)
  # Fixed X-only validation subset for early stopping; no latent truth is read.
  vi=np.random.default_rng(seed+11000).choice(len(va),16000,replace=False);vx=torch.from_numpy(va[vi]);vtarget=torch.from_numpy(vy[vi])
  for step in range(1,4001):
   m.train();idx=rng.integers(0,len(tr),512);pred,_=m(tx[idx]);loss=((pred-target[idx])**2).mean();opt.zero_grad();loss.backward();opt.step()
   if step%100==0:
    m.eval()
    with torch.no_grad():vl=float(((m(vx)[0]-vtarget)**2).mean())
    curve.append([step,vl])
    if vl<best-1e-5:
     best=vl;stale=0;torch.save(dict(state=m.state_dict(),X_mean=float(mu),X_sd=float(sd),step=step),d/'frozen.pt')
    else:stale+=1
    if stale>=8:break
  ck=torch.load(d/'frozen.pt',weights_only=False);m.load_state_dict(ck['state']);m.eval();save(d/'FREEZE.json',dict(time=time.time(),checkpoint_sha256=sha(d/'frozen.pt'),run_sha256=sha(ROOT/'run.py'),best_step=ck['step'],validation_X_MSE=best,truth_read_before_freeze=False));save(d/'training_curve.json',curve)
  pv,lv=encode(m,va);pt,lt=encode(m,te);ptr,ltr=encode(m,tr)
  # First opening of Y/Z for any fitting: only after checkpoint + freeze marker.
  truth=np.load(d/'sealed_truth.npz');vt=np.stack([truth[k][600:800,19:499].reshape(-1) for k in ['Y','Z']],1);et=np.stack([truth[k][800:,19:499].reshape(-1) for k in ['Y','Z']],1)
  probe_features={'L_only':(lv,lt),'X_current':(va[:,-1:],te[:,-1:]),'X_history':(va,te),'X_current_L':(np.c_[va[:,-1:],lv],np.c_[te[:,-1:],lt]),'X_history_L':(np.c_[va,lv],np.c_[te,lt])};statepred={};readouts={}
  for name,(a,b) in probe_features.items():
   pred,params=ridge(a,vt,b);readouts[name]=params;statepred[name]=pred
   for i in range(200):
    for j,key in enumerate(['Y','Z']):
     sl=slice(i*480,(i+1)*480);tar=et[sl,j];pr=pred[sl,j];mse=float(np.mean((pr-tar)**2));metrics.append(dict(seed=seed,trajectory=i,task=key,method=name,MSE=mse,NMSE=mse/float(np.var(vt[:,j])),corr=float(np.corrcoef(pr,tar)[0,1])))
  randomtr=np.random.default_rng(seed+13000).normal(size=(len(tr),2));randomte=np.random.default_rng(seed+14000).normal(size=(len(te),2))
  traintruth=np.stack([truth[k][:600,19:499].reshape(-1) for k in ['Y','Z']],1);testtruth=et
  controls={'A1_current':(tr[:,-1:],te[:,-1:]),'A2_history':(tr,te),'A3_history_L':(np.c_[tr,ltr],np.c_[te,lt]),'A4_history_trueYZ':(np.c_[tr,traintruth],np.c_[te,testtruth]),'A_random_history':(np.c_[tr,randomtr],np.c_[te,randomte])};forecast={};predictors={}
  # A4 is supervised oracle only; isolated from frozen encoder and ordinary heads.
  for name,(a,b) in controls.items():
   pr,param=ridge(a,ty*sd+mu,b);forecast[name]=pr;predictors[name]=param
  forecast['encoder_native']=pt*sd+mu
  for name,pr in forecast.items():
   err=((pr-(ey*sd+mu))**2).reshape(200,480)
   for i in range(200):metrics.append(dict(seed=seed,trajectory=i,task='X_next',method=name,MSE=float(err[i].mean()),NMSE=float(err[i].mean()/np.var(ty*sd)),corr=np.nan))
  # Random latent probe, independent random features, fitted on validation truth.
  rv=np.random.default_rng(seed+15000).normal(size=(len(va),2));rt=np.random.default_rng(seed+16000).normal(size=(len(te),2));rp,rparam=ridge(rv,vt,rt)
  for i in range(200):
   for j,key in enumerate(['Y','Z']):
    sl=slice(i*480,(i+1)*480);mse=float(np.mean((rp[sl,j]-et[sl,j])**2));metrics.append(dict(seed=seed,trajectory=i,task=key,method='random_L',MSE=mse,NMSE=mse/float(np.var(vt[:,j])),corr=float(np.corrcoef(rp[sl,j],et[sl,j])[0,1])))
  for k in range(2):
   for j,key in enumerate(['Y','Z']):corrs.append(dict(seed=seed,latent=k+1,target=key,test_corr=float(np.corrcoef(lt[:,k],et[:,j])[0,1])))
  save(d/'probe_maps.json',readouts);save(d/'forecast_heads.json',predictors);np.savez_compressed(d/'test_predictions.npz',L=lt.reshape(200,480,2),truth=et.reshape(200,480,2),**{k:v.reshape(200,480,2) for k,v in statepred.items()},forecast=np.stack([v.reshape(200,480) for v in forecast.values()]),forecast_names=np.array(list(forecast)))
  print('seed',seed,'done',round(time.time()-start,1),'seconds, best step',ck['step'],flush=True)
 pd.DataFrame(metrics).to_csv(OUT/'trajectory_metrics.csv.gz',index=False);pd.DataFrame(corrs).to_csv(OUT/'component_correlations.csv',index=False)
if __name__=='__main__':main()

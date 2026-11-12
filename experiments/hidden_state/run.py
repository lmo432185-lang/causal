import os,json,time,copy,csv
os.environ['MPLCONFIGDIR']='/workspace/work/mpl'
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
ROOT=Path(__file__).parent; OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
torch.set_num_threads(2)
DT=.05; M=21; HS=[1,5,10]; SEEDS=[11,22,33]
# Frozen before inspecting test outcomes; Y is never passed into latent training.
CONFIG=dict(dt=DT,integration_dt=.005,duration=60,burnin=10,train=60,validation=20,test=100,embedding=M,horizons=HS,seeds=SEEDS,steps=1000,batch=512,k=1,noise={'A1':[.025,.05],'A2':[.5,1.],'A3':[1.,2.]})
(OUT/'config.json').write_text(json.dumps(CONFIG,indent=2))
def simulate(n,noise,seed,dynamic=False,sub=10):
 r=np.random.default_rng(seed); dt=DT/sub; steps=int(70/dt); x=r.normal(.5,.3,n); y=r.normal(.2,.3,n); data=[]
 for i in range(steps):
  t=(i+1)*dt-10; ga=2+np.clip((t-30)/10,0,1) if dynamic else 2
  dx=(-.5*x+ga*x*y+.5)*dt+noise[0]*np.sqrt(dt)*r.normal(size=n)
  dy=(-.5*y-ga*x*x+1)*dt+noise[1]*np.sqrt(dt)*r.normal(size=n)
  x,y=x+dx,y+dy
  if (i+1)%sub==0 and t>=-1e-8: data.append(np.stack([x,y],-1))
 a=np.stack(data,1).astype('float32'); assert np.isfinite(a).all(); return a

def windows(a,mu,sd):
 x=(a[:,:,0]-mu)/sd
 ids=np.arange(M-1,x.shape[1]-max(HS)); h=np.stack([x[:,ids-j] for j in range(M)],-1)
 targets=np.stack([x[:,ids+j]-x[:,ids] for j in HS],-1)
 nxt=np.stack([x[:,ids+1-j] for j in range(M)],-1)
 return [torch.tensor(v.reshape(-1,v.shape[-1])) for v in (h,nxt,targets)],ids

def net(a,b): return nn.Sequential(nn.Linear(a,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,b))
class Latent(nn.Module):
 def __init__(self):
  super().__init__(); self.enc=net(M,1); self.obs=net(2,3); self.dyn=net(2,1)
 def forward(self,h,n):
  z=self.enc(h); zn=self.enc(n); state=torch.cat([h[:,:1],z],1)
  return self.obs(state), self.dyn(state)+z, z,zn

def fit_lat(data,seed):
 torch.manual_seed(seed); mod=Latent(); opt=torch.optim.Adam(mod.parameters(),lr=.002); h,n,y=data
 for step in range(CONFIG['steps']):
  idx=torch.randint(len(h),(512,)); p,d,z,zn=mod(h[idx],n[idx]);
  loss=((p-y[idx])**2).mean()+.2*((d-zn)**2).mean()+.02*z.mean()**2+.02*(z.std()-1)**2
  opt.zero_grad(); loss.backward(); opt.step()
 return mod.eval()

def fit_pred(tr,va,seed):
 torch.manual_seed(seed); f=net(tr[0].shape[1],3); op=torch.optim.Adam(f.parameters(),lr=.002); best=1e20; checkpoint=None
 for step in range(800):
  idx=torch.randint(len(tr[0]),(512,)); loss=((f(tr[0][idx])-tr[1][idx])**2).mean(); op.zero_grad(); loss.backward(); op.step()
  if step%100==99:
   with torch.no_grad(): score=((f(va[0])-va[1])**2).mean().item()
   if score<best: best=score; checkpoint=copy.deepcopy(f.state_dict())
 f.load_state_dict(checkpoint); return f.eval()

def arr(t): return t.detach().numpy()
def ci(v):
 v=np.asarray(v); return dict(mean=float(v.mean()),sd=float(v.std(ddof=1)),ci95=[float(v.mean()-1.984*v.std(ddof=1)/np.sqrt(len(v))),float(v.mean()+1.984*v.std(ddof=1)/np.sqrt(len(v)))])
def ridge(a,b): return np.linalg.solve(a.T@a+np.eye(a.shape[1])*.01,a.T@b)
def features(x,z): return np.stack([np.ones_like(x),x,x*x,z,x*z],-1)
def learn_ssm(h,n,z,zn,y):
 x=arr(h)[:,0]; zz=z[:,0]; f=features(x,zz); a=ridge(f,zn[:,0]); b=ridge(f,arr(y)[:,0]); q=max(np.mean((zn[:,0]-f@a)**2),1e-5); r=max(np.mean((arr(y)[:,0]-f@b)**2),1e-5)
 return a,b,float(q),float(r)
def information(ssm,x):
 # Conditional linear Gaussian approximation: z transition and X increment linear in z.
 a,b,q,r=ssm; N,T=x.shape; fm=np.zeros((N,T)); fp=np.ones((N,T)); pm=np.zeros((N,T)); pp=np.ones((N,T)); aa=a[3]+a[4]*x
 for t in range(T-1):
  xx=x[:,t]; c=b[3]+b[4]*xx; obs=x[:,t+1]-xx-(b[0]+b[1]*xx+b[2]*xx**2)
  # Observation X_(t+1) updates z_t; filtered at t excludes this future observation.
  v=c*c*fp[:,t]+r; k=fp[:,t]*c/v; post=fm[:,t]+k*(obs-c*fm[:,t]); postp=np.maximum(fp[:,t]*(1-k*c),1e-8)
  pm[:,t+1]=aa[:,t]*post+a[0]+a[1]*xx+a[2]*xx**2; pp[:,t+1]=aa[:,t]**2*postp+q
  fm[:,t+1]=pm[:,t+1]; fp[:,t+1]=pp[:,t+1]
 # Backward recursion includes each future X increment in the z_t likelihood.
 def back(sm,sp,t):
  xx=x[:,t]; c=b[3]+b[4]*xx; v=c*c*fp[:,t]+r; k=fp[:,t]*c/v
  post=fm[:,t]+k*(x[:,t+1]-xx-b[0]-b[1]*xx-b[2]*xx**2-c*fm[:,t]); p=np.maximum(fp[:,t]*(1-k*c),1e-8)
  gain=p*aa[:,t]/pp[:,t+1]
  return post+gain*(sm-pm[:,t+1]),np.maximum(p+gain**2*(sp-pp[:,t+1]),1e-8)
 sm=fm.copy(); sp=fp.copy()
 for t in range(T-2,-1,-1): sm[:,t],sp[:,t]=back(sm[:,t+1],sp[:,t+1],t)
 aci=.5*(np.log(fp/sp)+(sp+(sm-fm)**2)/fp-1)
 # Finite-future approximation to CIR: integral KL(full||lag)/KL(full||filter), up to 2 time units.
 L=40; count=T-L; numer=np.zeros((N,count)); denom=aci[:,:count];
 for lag in range(L+1):
  lm=fm[:,lag:lag+count].copy(); lp=fp[:,lag:lag+count].copy()
  for j in range(lag-1,-1,-1):
   ix=np.arange(count)+j; xx=x[:,ix]; c=b[3]+b[4]*xx; p0=fp[:,ix]; k=p0*c/(c*c*p0+r)
   post=fm[:,ix]+k*(x[:,ix+1]-xx-b[0]-b[1]*xx-b[2]*xx**2-c*fm[:,ix]); p=np.maximum(p0*(1-k*c),1e-8); gain=p*aa[:,ix]/pp[:,ix+1]
   lm=post+gain*(lm-pm[:,ix+1]); lp=np.maximum(p+gain**2*(lp-pp[:,ix+1]),1e-8)
  kl=.5*(np.log(lp/sp[:,:count])+(sp[:,:count]+(sm[:,:count]-lm)**2)/lp-1)
  numer+=np.maximum(kl,0)*DT*(.5 if lag in (0,L) else 1)
 cir=numer/np.maximum(denom,1e-6); cir[denom<1e-4]=np.nan
 return aci,cir

def main():
 metrics=[]; examples={}; start=time.time()
 for gi,(group,noise) in enumerate(CONFIG['noise'].items()):
  train=simulate(60,noise,100+gi); val=simulate(20,noise,200+gi); test=simulate(100,noise,300+gi)
  mu=float(train[:,:,0].mean()); sd=float(train[:,:,0].std()); tr,ids=windows(train,mu,sd); va,_=windows(val,mu,sd); te,_=windows(test,mu,sd)
  for seed in SEEDS:
   latent=fit_lat(tr,seed)
   with torch.no_grad():
    z=arr(latent.enc(tr[0])); zv=arr(latent.enc(va[0])); zt=arr(latent.enc(te[0])); zn=arr(latent.enc(tr[1]))
   # Y alignment is post hoc only, fitted on validation trajectories, never test.
   vy=val[:,ids,1].reshape(-1); coef=np.linalg.lstsq(np.c_[np.ones(len(zv)),zv[:,0]],vy,rcond=None)[0]
   ssm=learn_ssm(tr[0],tr[1],z,zn,tr[2]); models={}
   for name in ['M1','M2','M3','M4']:
    def inp(d,zz,a):
     h=d[0]
     if name=='M1': return h[:,:1]
     if name=='M2': return h
     extra=torch.tensor(zz) if name=='M3' else torch.tensor(a[:,ids,1].reshape(-1,1))
     return torch.cat([h,extra],1)
    models[name]=fit_pred((inp(tr,z,train),tr[2]),(inp(va,zv,val),va[2]),seed)
   evals=[(group,test,te,zt)]
   if group=='A2':
    a4=simulate(100,noise,401,True); t4,_=windows(a4,mu,sd)
    with torch.no_grad(): z4=arr(latent.enc(t4[0]))
    evals.append(('A4',a4,t4,z4))
   for label,data,d,zz in evals:
    ny=len(ids); yh=(coef[0]+coef[1]*zz[:,0]).reshape(100,ny); yy=data[:,ids,1]
    aci,cir=information(ssm,(data[:,:,0]-mu)/sd)
    pred={}
    for name,f in models.items():
     h=d[0]; xx=h[:,:1] if name=='M1' else h
     if name in ['M3','M4']: xx=torch.cat([h,torch.tensor(zz) if name=='M3' else torch.tensor(yy.reshape(-1,1))],1)
     with torch.no_grad(): pred[name]=arr(f(xx)).reshape(100,ny,3)*sd+data[:,ids,0,None]
    for j in range(100):
     row=dict(group=label,seed=seed,trajectory=j,corr=float(np.corrcoef(zz.reshape(100,ny)[j],yy[j])[0,1]),nrmse=float(np.sqrt(np.mean((yh[j]-yy[j])**2))/np.std(yy[j])),aci=float(aci[j,100:-50].mean()),cir=float(np.nanmean(cir[j,100:-50])))
     for name,p in pred.items():
      for hi,lag in enumerate(HS):
       err=p[j,:,hi]-data[j,ids+lag,0]; row[f'{name}_mse_h{lag}']=float(np.mean(err**2)); row[f'{name}_mae_h{lag}']=float(np.mean(np.abs(err)))
     for period,lo,hi in [('pre',10,30),('ramp',30,40),('post',40,60)]:
      tt=np.arange(aci.shape[1])*DT; row['aci_'+period]=float(aci[j,(tt>=lo)&(tt<hi)].mean())
      mask=(ids*DT>=lo)&(ids*DT<hi); row['err_'+period]=float(np.mean((pred['M3'][j,mask,0]-data[j,ids[mask]+1,0])**2))
     metrics.append(row)
    if seed==11: examples[label]=dict(data=data[0],ids=ids,z=zz.reshape(100,ny)[0],aligned=yh[0],aci=aci[0],cir=cir[0],**{k:v[0] for k,v in pred.items()})
   torch.save(dict(latent=latent.state_dict(),models={k:v.state_dict() for k,v in models.items()},mu=mu,sd=sd,ssm=ssm,alignment=coef),OUT/f'{group}_{seed}.pt')
   print(group,seed,'done',round(time.time()-start,1),flush=True)
 with (OUT/'per_trajectory.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(metrics[0])); w.writeheader(); w.writerows(metrics)
 summary={}
 for group in ['A1','A2','A3','A4']:
  rows=[r for r in metrics if r['group']==group]; summary[group]={}
  # Average model seeds within trajectory before CI, preserving the trajectory as independent unit.
  for key in rows[0]:
   if key in ['group','seed','trajectory']: continue
   values=[np.mean([r[key] for r in rows if r['trajectory']==j]) for j in range(100)]
   if key=='corr': values=[np.mean([abs(r[key]) for r in rows if r['trajectory']==j]) for j in range(100)]
   summary[group][key]=ci(values)
  for lag in HS:
   vals=[np.mean([r[f'M2_mse_h{lag}']-r[f'M3_mse_h{lag}'] for r in rows if r['trajectory']==j]) for j in range(100)]
   summary[group][f'M2_minus_M3_h{lag}']=ci(vals)
  summary[group]['seed_abs_corr']={str(s):float(np.mean([abs(r['corr']) for r in rows if r['seed']==s])) for s in SEEDS}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)); np.savez_compressed(OUT/'examples.npz',**{g+'_'+k:v for g,d in examples.items() for k,v in d.items()})
 fig,axes=plt.subplots(4,3,figsize=(16,13))
 for i,(g,d) in enumerate(examples.items()):
  t=np.arange(len(d['data']))*DT; ix=d['ids']; axes[i,0].plot(t,d['data'][:,0],label='X'); axes[i,0].plot(t[ix],d['data'][ix,1],label='Y truth',alpha=.7); axes[i,0].plot(t[ix],d['aligned'],label='aligned Z',alpha=.7); axes[i,0].set_title(g+' state (test trajectory 0)'); axes[i,0].legend(fontsize=7)
  axes[i,1].plot(t,d['aci'],label='ACI (nats)'); axes[i,1].plot(t[:len(d['cir'])],d['cir'],label='truncated CIR'); axes[i,1].set_title(g+' learned Gaussian approximation'); axes[i,1].legend(fontsize=7)
  vals=[summary[g][f'{m}_mse_h5']['mean'] for m in ['M1','M2','M3','M4']]; axes[i,2].bar(['M1','M2','M3','M4'],vals); axes[i,2].set_title(g+' forecast MSE, horizon 0.25')
 fig.tight_layout(); fig.savefig(OUT/'overview.png',dpi=160); plt.close(fig)
 print(json.dumps({g:{k:v for k,v in s.items() if k in ['corr','nrmse','M2_mse_h1','M3_mse_h1','M4_mse_h1','M2_minus_M3_h1','aci_pre','aci_post','cir']} for g,s in summary.items()},indent=2),flush=True)
if __name__=='__main__': main()

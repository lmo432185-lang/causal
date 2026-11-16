import json,time
import numpy as np,pandas as pd,torch
from config import *
from core import *
from encoder import Net,encode
SRC=R/'inputs'
torch.set_num_threads(2)
def train_x(x,seed,path):
 mu=x[:600].mean();sd=x[:600].std();tr=windows((x[:600]-mu)/sd);ty=((x[:600,20:]-mu)/sd).reshape(-1,1);va=windows((x[600:800]-mu)/sd);vy=((x[600:800,20:]-mu)/sd).reshape(-1,1);torch.manual_seed(seed);rng=np.random.default_rng(seed+10000);m=Net();opt=torch.optim.Adam(m.parameters(),lr=.001);tx=torch.from_numpy(tr);ty=torch.from_numpy(ty);vi=np.random.default_rng(seed+11000).choice(len(va),16000,False);vx=torch.from_numpy(va[vi]);vt=torch.from_numpy(vy[vi]);best=np.inf;stale=0
 for step in range(1,4001):
  idx=rng.integers(0,len(tx),512);p,_=m(tx[idx]);loss=((p-ty[idx])**2).mean();opt.zero_grad();loss.backward();opt.step()
  if step%100==0:
   with torch.no_grad():vl=float(((m(vx)[0]-vt)**2).mean())
   if vl<best-1e-5:best=vl;stale=0;torch.save(dict(state=m.state_dict(),X_mean=float(mu),X_sd=float(sd),step=step),path)
   else:stale+=1
   if stale>=8:break
 ck=torch.load(path,weights_only=False);m.load_state_dict(ck['state']);return m.eval(),ck

def run():
 rows=[];timing=[];manifest=[]
 for cell in CELLS:
  for seed in SEEDS:
   t0=time.time();out=O/'scan'/cell['name']/str(seed);out.mkdir(parents=True,exist_ok=True);inp=SRC/str(seed);old=torch.load(inp/'frozen.pt',weights_only=False);oldm=Net();oldm.load_state_dict(old['state']);oldm.eval();oldmap=json.loads((inp/'probe_maps.json').read_text())
   if cell['name']=='base':
    x=np.load(inp/'X_only.npz')['X'];z=np.load(inp/'sealed_truth.npz');states=np.stack([z['Y'],z['Z'],x],2);m=oldm;ck=old;torch.save(ck,out/'matched.pt')
   else:x,states=generate(cell['params'],seed);m,ck=train_x(x,seed,out/'matched.pt')
   (out/'FREEZE.json').write_text(json.dumps(dict(time=time.time(),checkpoint_sha256=__import__('hashlib').sha256((out/'matched.pt').read_bytes()).hexdigest(),truth_not_used_in_encoder=True,base_reused=cell['name']=='base',step=ck['step']),indent=2))
   xx=(x-np.float32(ck['X_mean']))/np.float32(ck['X_sd']);hv=windows(xx[600:800]);ht=windows(xx[800:]);lv=encode(m,hv);lt=encode(m,ht);targetv=states[600:800,19:499,:2].reshape(-1,2);truth=states[800:,19:499,:2];pred,mp=ridge(lv,targetv,lt);hp,hmap=ridge(hv,targetv,ht);oldh=windows((x[800:]-np.float32(old['X_mean']))/np.float32(old['X_sd']));oldl=encode(oldm,oldh);op=apply(oldl,oldmap['L_only']);A,Q,obsR=plant(**cell['params']);mf,ms,pf,ps=kalman(x[800:],A,Q,obsR);preds={'L1_matched':pred.reshape(200,480,2),'L1_frozen_OOD':op.reshape(200,480,2),'history20':hp.reshape(200,480,2),'KF_known':mf[:,19:499,:2],'RTS_future':ms[:,19:499,:2]};vvar=targetv.var(0);(out/'maps.json').write_text(json.dumps(dict(L=mp,history=hmap),indent=2));eig=np.linalg.eigvalsh(np.cov(lt.T));manifest.append(dict(cell=cell['name'],seed=seed,L_corr=float(np.corrcoef(lt.T)[0,1]),PC1=float(eig[-1]/eig.sum()),steps=ck['step'],base_reused=cell['name']=='base'))
   for method,pp in preds.items():
    for i in range(200):
     for j,key in enumerate(['Y','Z']):
      e=pp[i,:,j]-truth[i,:,j];mse=float((e*e).mean());pv=float(pf[19:499,j,j].mean()) if method=='KF_known' else float(ps[19:499,j,j].mean()) if method=='RTS_future' else np.nan;rows.append(dict(cell=cell['name'],factor=cell['factor'],level=cell['level'],seed=seed,trajectory=i,state=key,method=method,MSE=mse,NMSE=mse/float(vvar[j]),corr=float(np.corrcoef(pp[i,:,j],truth[i,:,j])[0,1]),posterior_variance=pv))
   np.savez_compressed(out/'examples.npz',X=x[800:803],truth=truth[:3],L=lt.reshape(200,480,2)[:3],**{k:v[:3] for k,v in preds.items()},filter_cov=pf,smooth_cov=ps)
   timing.append(dict(cell=cell['name'],seed=seed,seconds=time.time()-t0));print('SCAN',cell['name'],seed,round(time.time()-t0,1),flush=True)
  pd.DataFrame(rows).to_csv(O/'scan_metrics.csv.gz',index=False);pd.DataFrame(manifest).to_csv(O/'scan_models.csv',index=False);pd.DataFrame(timing).to_csv(O/'scan_timing.csv',index=False)

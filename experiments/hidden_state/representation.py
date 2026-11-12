import torch,numpy as np,json
from run import simulate,windows,Latent,OUT,SEEDS,ci
res={}
for i,(g,noise) in enumerate([('A1',[.025,.05]),('A2',[.5,1]),('A3',[1,2]),('A4',[.5,1])]):
 va=simulate(20,noise,200+min(i,1) if g=='A4' else 200+i); te=simulate(100,noise,401 if g=='A4' else 300+i,dynamic=g=='A4'); vals=[]
 for seed in SEEDS:
  ck=torch.load(OUT/f'{"A2" if g=="A4" else g}_{seed}.pt',weights_only=False); mod=Latent(); mod.load_state_dict(ck['latent']); v,ids=windows(va,ck['mu'],ck['sd']); t,_=windows(te,ck['mu'],ck['sd'])
  with torch.no_grad(): zv=mod.enc(v[0]).numpy(); zt=mod.enc(t[0]).numpy()
  vy=va[:,ids,1].reshape(-1); ty=te[:,ids,1]; row={}
  for name,ev,et in [('X',v[0][:,:1].numpy(),t[0][:,:1].numpy()),('XZ',np.c_[v[0][:,:1],zv],np.c_[t[0][:,:1],zt])]:
   coef=np.linalg.lstsq(np.c_[np.ones(len(ev)),ev],vy,rcond=None)[0]; pred=(np.c_[np.ones(len(et)),et]@coef).reshape(100,-1); row[name]=np.mean((pred-ty)**2,1)
  vals.append((row['X']-row['XZ'])/np.var(ty,axis=1))
 res[g]={'incremental_Y_R2_over_X':ci(np.mean(vals,0)),'note':'post-hoc linear probes fitted on validation Y only; no feedback to model training'}
(OUT/'representation.json').write_text(json.dumps(res,indent=2)); print(json.dumps(res,indent=2))

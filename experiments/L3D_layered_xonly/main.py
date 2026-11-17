import hashlib,json,time
import numpy as np,pandas as pd
from config import *
from core import *

def dump(p,obj):p.write_text(json.dumps(obj,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pearson(a,b):return float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-12 and np.std(b)>1e-12 else 0.
def compare(train,val,test,full,reduced,y,ns,rng,meta,rows):
 # Feature/target tuples already separate trajectories before any regression.
 fitF=olsfit(full[0],y[0]);fitR=olsfit(reduced[0],y[0]);vf=olsapply(full[1],fitF);vr=olsapply(reduced[1],fitR);tf=olsapply(full[2],fitF);tr=olsapply(reduced[2],fitR);vfm=float(np.mean((vf-y[1])**2));vrm=float(np.mean((vr-y[1])**2));vg=(vrm-vfm)/max(vrm,1e-15);ef=((tf-y[2])**2).reshape(200,ns,-1).mean((1,2));er=((tr-y[2])**2).reshape(200,ns,-1).mean((1,2));delta=er-ef;gain=float(delta.mean()/max(er.mean(),1e-15));ci=np.quantile(delta[rng.integers(0,200,(1000,200))].mean(1),[.025,.975]);candidate=bool(vg>=GAIN);supported=bool(candidate and gain>=GAIN and ci[0]>0)
 for i in range(200):rows.append(dict(**meta,trajectory=i,MSE_full=float(ef[i]),MSE_reduced=float(er[i]),delta=float(delta[i])))
 return dict(**meta,validation_gain=float(vg),test_gain=gain,delta_mean=float(delta.mean()),CI95=ci.tolist(),validation_candidate=candidate,test_supported=supported,MSE_full=float(ef.mean()),MSE_reduced=float(er.mean())),dict(full=fitF,reduced=fitR)

def direct_tests(xparts,Lparts,seed,group,label,rows):
 rng=np.random.default_rng(seed+6000);summaries=[];saved={};k=Lparts[0].shape[-1]
 for baseline in ['current_X','full_X20']:
  designs=[forecast_design(x,L,baseline) for x,L in zip(xparts,Lparts)];base=[v[0] for v in designs];lat=[v[1] for v in designs];ys=[v[2] for v in designs]
  for j in range(k):
   f=[np.c_[b,l] for b,l in zip(base,lat)];rr=[np.c_[b,np.delete(l,j,axis=1)] for b,l in zip(base,lat)]
   for hj,h in enumerate(HORIZONS):
    yy=[y[:,hj:hj+1] for y in ys];meta=dict(group=group,seed=seed,family=label,stage='direct',control=baseline,horizon=h,source=f'L{j+1}',target='X',lag=0);ss,mm=compare(None,None,None,f,rr,yy,473,rng,meta,rows);summaries.append(ss);saved[f'{baseline}_L{j+1}_h{h}']=mm
 return summaries,saved

def direction_tests(xparts,Sparts,seed,group,label,rows,covered_options=[False,True],lags=LAGS,names=None):
 rng=np.random.default_rng(seed+7000);summ=[];maps={};k=Sparts[0].shape[-1];names=names or ['X']+[f'L{j}' for j in range(1,k)]
 for p in lags:
  for covered in covered_options:
   for t in range(k):
    for source in range(k):
     if t==source:continue
     designs=[direction_design(x,S,t,source,p,covered) for x,S in zip(xparts,Sparts)];f=[v[0] for v in designs];rr=[v[1] for v in designs];yy=[v[2] for v in designs];meta=dict(group=group,seed=seed,family=label,stage='direction',control='all_nodes_plus_full_X_covered' if covered else 'all_node_histories',horizon=1,source=names[source],target=names[t],lag=p);ss,mm=compare(None,None,None,f,rr,yy,designs[0][3],rng,meta,rows);summ.append(ss);maps[f'{p}_{covered}_{source}_{t}']=mm
 return summ,maps

def main():
 dump(R/'PROTOCOL.json',PROTOCOL);files=['config.py','core.py','main.py','PROTOCOL.json'];dump(R/'REGISTERED.json',dict(time=time.time(),files={f:sha(R/f) for f in files}));edge_rows=[];edges=[];states_rows=[];modelrows=[];rotations=[]
 for gi,g in enumerate(GROUPS):
  for seed in SEEDS:
   out=O/g['name']/str(seed);out.mkdir(parents=True,exist_ok=True);x,truth=generate(g,960000+1000*gi+seed);xp=[x[:600],x[600:800],x[800:]];model=fit_predictive(xp[0],xp[1]);dump(out/'frozen_X_model.json',model);dump(out/'FREEZE.json',dict(time=time.time(),model_sha256=sha(out/'frozen_X_model.json'),truth_used_in_learning=False));Lparts=[encode(v,model) for v in xp];k=model['rank'];S=[np.concatenate([v[:,19:500,None],l],2) for v,l in zip(xp,Lparts)];ss,mm=direct_tests(xp,Lparts,seed,g['name'],'selected',edge_rows);edges+=ss;dump(out/'direct_predictors.json',mm);ss,mm=direction_tests(xp,S,seed,g['name'],'selected',edge_rows);edges+=ss;dump(out/'direction_predictors.json',mm)
   # Fixed rank2 is a separately labeled capacity/basis diagnostic, not chosen hidden count.
   l2=[encode(v,model,2) for v in xp];ref=forecast(xp[2],model,2);rotgraphs=[]
   for angle in [0.,np.pi/8,np.pi/4,np.pi/2]:
    q=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);rot=[l@q for l in l2];Srot=[np.concatenate([v[:,19:500,None],l],2) for v,l in zip(xp,rot)];label=f'rank2_rotate_{angle:.6f}';ss,_=direction_tests(xp,Srot,seed,g['name'],label,edge_rows,covered_options=[False],lags=[1]);edges+=ss;yy=((xp[2][:,19:492]-model['mu'])/model['sd']).reshape(-1);cc=np.c_[np.ones(len(yy)),yy];pr=(cc@np.array(model['by'])+rot[2][:,:473].reshape(-1,2)@(q.T@np.array(model['head'])[:2]))*model['sd']+model['mu'];err=float(np.max(abs(pr-ref)));assert err<1e-10;links=[f"{e['source']}->{e['target']}" for e in ss if e['test_supported']];rotations.append(dict(group=g['name'],seed=seed,angle=angle,forecast_max_difference=err,supported_edges=links))
   # Only now expose hidden truth for independent validation readout and oracle.
   targets=[a[:,19:500,:-1] for a in [truth[:600],truth[600:800],truth[800:]]];testtruth=targets[2];nstate=testtruth.shape[-1];preds={};maps={};names=['Y'] if nstate==1 else ['Y','Z'];vt=targets[1].reshape(-1,nstate);vvar=vt.var(0)
   for name,a,b in [('L_only',Lparts[1].reshape(200*481,k),Lparts[2].reshape(200*481,k)),('current_X',xp[1][:,19:500].reshape(-1,1),xp[2][:,19:500].reshape(-1,1)),('joint_X_L',S[1].reshape(200*481,k+1),S[2].reshape(200*481,k+1)),('history20',hist(xp[1],19,500).reshape(-1,20),hist(xp[2],19,500).reshape(-1,20))]:
    mp=olsfit(a,vt);pp=olsapply(b,mp).reshape(200,481,nstate);preds[name]=pp;maps[name]=mp
    for i in range(200):
     for j,key in enumerate(names):
      e=pp[i,:,j]-testtruth[i,:,j];states_rows.append(dict(group=g['name'],seed=seed,trajectory=i,state=key,method=name,MSE=float((e*e).mean()),NMSE=float((e*e).mean()/vvar[j]),corr=pearson(pp[i,:,j],testtruth[i,:,j])))
   dump(out/'diagnostic_probes.json',maps);coords=[]
   for j in range(k):
    for jj,key in enumerate(names):coords.append(dict(coordinate=f'L{j+1}',state=key,validation_corr=pearson(Lparts[1][:,:,j].ravel(),targets[1][:,:,jj].ravel()),test_corr=pearson(Lparts[2][:,:,j].ravel(),targets[2][:,:,jj].ravel())))
   dump(out/'coordinate_diagnostics.json',coords)
   oracle=[a[:,19:500,:] for a in [truth[:600],truth[600:800],truth[800:]]];onames=names+['X'];ss,mm=direction_tests(xp,oracle,seed,g['name'],'oracle_true_states',edge_rows,covered_options=[False],names=onames);edges+=ss;dump(out/'oracle_predictors.json',mm)
   # Oracle direct-to-X conditions use all current hidden states, distinguish horizons.
   realparts=[a[:,:,:-1] for a in oracle];ss,mm=direct_tests(xp,realparts,seed,g['name'],'oracle_hidden',edge_rows);edges+=ss;dump(out/'oracle_direct_predictors.json',mm)
   hpred=forecast(xp[2],model).reshape(200,473,4);hy=np.stack([xp[2][:,19+h:492+h] for h in HORIZONS],2);eig=np.linalg.eigvalsh(np.cov(Lparts[2].reshape(-1,k).T)) if k>1 else np.array([1.]) if k else np.array([0.]);validdirect=[e['source'] for e in edges if e['group']==g['name'] and e['seed']==seed and e['family']=='selected' and e['stage']=='direct' and e['control']=='full_X20' and e['horizon']==1 and e['test_supported']];modelrows.append(dict(group=g['name'],seed=seed,rank=k,test_forecast_MSE_by_horizon=((hpred-hy)**2).mean((0,1)).tolist(),PC1=float(eig[-1]/eig.sum()) if eig.sum() else None,validated_direct_states=validdirect,validated_upper_chain=[] if not validdirect else 'inspect supported latent edges',causal_structure='refuse unique physical graph: X-only representation basis and observational equivalence unresolved'))
   np.savez_compressed(out/'test_estimates.npz',X=xp[2],truth=testtruth,L=Lparts[2],forecast=hpred,**preds);dump(out/'truth_evaluation_config.json',g);print(g['name'],seed,'rank',k,'done',flush=True)
   pd.DataFrame(edge_rows).to_csv(O/'edge_trajectory_metrics.csv.gz',index=False);pd.DataFrame(states_rows).to_csv(O/'state_metrics.csv.gz',index=False);dump(O/'edge_summary.json',edges);dump(O/'models.json',modelrows);dump(O/'rotations.json',rotations)
 print('ALL COMPLETE',flush=True)
if __name__=='__main__':main()

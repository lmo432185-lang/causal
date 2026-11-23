"""Post-freeze truth audit. Calibrated readout, never truth-fed model training."""
import json,time,hashlib,csv,gzip
import numpy as np
from scipy.special import logsumexp,ndtr
from scipy.stats import chi2
from concurrent.futures import ProcessPoolExecutor,as_completed
from settings import *
from ssm import unpack,spd,component
from predict import layout,weights,Aabs
from producer import simulate
import native

def posterior(x,m,graph,group,cache):
 ids,prior=layout(m,graph,group);online=group not in ['A','C'];parts=[]
 for b,k in ids:
  key=(b,k,online)
  if key not in cache:
   z=m[b]['models'][k];p=native.run(x/z['scale'],z,stop=100000 if online else 40);p['ll']-=np.log(z['scale']);cache[key]=p
  parts.append(cache[key])
 w=weights(np.stack([p['ll'] for p in parts],-1),prior,online);return ids,parts,w

def work(j):
 x,z=simulate(j['index'],j['seed'],3,80,truth=True)
 if z.shape[-1]==0:return []
 xa,ya=x[:40],z[:40,40:];xt,yt=x[40:],z[40:,40:];N,T,d=yt.shape;m=unpack(json.loads((OUT/j['case']/'model_100.json').read_text()));graph=json.loads((OUT/j['case']/'graph.json').read_text())['100'];rows=[];calcache={};testcache={};base_mean=ya.mean((0,1));base_cov=spd(np.cov(ya.reshape(-1,d).T).reshape(d,d));br=yt-base_mean;baseline=-.5*(np.einsum('nti,ij,ntj->nt',br,np.linalg.inv(base_cov),br)+np.linalg.slogdet(base_cov)[1]+d*np.log(2*np.pi));basevar=float(np.mean(br*br))
 for group in ['A','B','C','D','E','F']:
  ids,cal,wc=posterior(xa,m,graph,group,calcache);_,test,w=posterior(xt,m,graph,group,testcache);w=w[:,40:];mus=[];covs=[];lls=[];readout_noise=[]
  for cp,tp in zip(cal,test):
   cmean=cp['mean'][:,40:];tmean=tp['mean'][:,40:];k=cmean.shape[-1];D=np.concatenate([np.ones((*cmean.shape[:2],1)),cmean],-1);coef=np.linalg.lstsq(D.reshape(-1,k+1),ya.reshape(-1,d),rcond=1e-7)[0];mu=np.concatenate([np.ones((*tmean.shape[:2],1)),tmean],-1)@coef;B=coef[1:];res=ya-D@coef;rescov=res.reshape(-1,d).T@res.reshape(-1,d)/res.shape[0]/res.shape[1];Pcal=np.einsum('ki,tkl,lj->tij',B,cp['cov'][40:],B) if k else np.zeros((T,d,d));delta=(rescov-Pcal.mean(0));noise=spd(delta,1e-5);P=np.einsum('ki,tkl,lj->tij',B,tp['cov'][40:],B)+noise if k else np.broadcast_to(noise,(T,d,d));P=np.array([spd(v,1e-6) for v in P]);r=yt-mu;ll=-.5*(np.einsum('nti,tij,ntj->nt',r,np.linalg.inv(P),r)+np.linalg.slogdet(P)[1]+d*np.log(2*np.pi));mus.append(mu);covs.append(P);lls.append(ll);readout_noise.append(float(np.trace(noise)))
  mus=np.stack(mus,2);covs=np.stack(covs,1);lls=np.stack(lls,2);lp=logsumexp(np.log(np.maximum(w,1e-300))+lls,axis=2);sd=np.sqrt(np.diagonal(covs,axis1=2,axis2=3));pit=np.sum(w[:,:,:,None]*ndtr((yt[:,:,None]-mus)/sd[None]),axis=2);s1=np.sum(w[:,:,:,None]*Aabs(yt[:,:,None]-mus,sd[None]),axis=2);s2=0
  for a in range(len(ids)):
   for b in range(len(ids)):s2=s2+w[:,:,a,None]*w[:,:,b,None]*Aabs(mus[:,:,a]-mus[:,:,b],np.sqrt(sd[None,:,a]**2+sd[None,:,b]**2))
  mean=np.sum(w[:,:,:,None]*mus,axis=2);second=np.einsum('ntm,tmij->ntij',w,covs)+np.einsum('ntm,ntmi,ntmj->ntij',w,mus,mus)-np.einsum('nti,ntj->ntij',mean,mean);iv=np.linalg.inv(second);mah=np.einsum('nti,ntij,ntj->nt',yt-mean,iv,yt-mean)
  for i in range(N):rows.append(dict(case=j['case'],seed=j['seed'],n=100,group=group,trajectory=i,true_state_dimension=d,readout_NLL=float(-lp[i].mean()/d),baseline_NLL=float(-baseline[i].mean()/d),readout_gain=float((lp[i]-baseline[i]).mean()/d),readout_CRPS=float((s1[i]-.5*s2[i]).mean()),marginal_cover90=float(((pit[i]>=.05)&(pit[i]<=.95)).mean()),moment_ellipsoid_cover90=float((mah[i]<=chi2.ppf(.9,d)).mean()),readout_R2=float(1-np.mean((mean[i]-yt[i])**2)/basevar),mean_readout_noise_trace=float(np.mean(readout_noise))))
 print('TRUTH AUDIT',j['case'],flush=True);return rows

def main():
 for label in ['TEST_A_FROZEN','TEST_B_FROZEN']:
  for f,h in json.loads((ROOT/(label+'.json')).read_text())['files'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
 path=OUT/'state_readout.csv.gz';assert not path.exists();count=0
 with gzip.open(path,'wt') as stream:
  writer=None
  with ProcessPoolExecutor(max_workers=2) as p:
   for f in as_completed([p.submit(work,j) for j in json.loads((ROOT/'jobs.json').read_text())]):
    rows=f.result()
    if not rows:continue
    if writer is None:writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader()
    writer.writerows(rows);count+=len(rows)
 (ROOT/'TRUTH_EVALUATION_DONE.json').write_text(json.dumps(dict(time=time.time(),rows=count,only_after_all_X_predictions_frozen=True,separate_calibration_and_evaluation_trajectories=True,not_true_physical_state_posterior=True,sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2));print('TRUTH AUDIT DONE',count)
if __name__=='__main__':main()

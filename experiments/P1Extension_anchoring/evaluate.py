"""Sealed full test answers are opened only after all permitted-information models freeze."""
import json,time,hashlib
import numpy as np,pandas as pd
from settings import *
from scipy.stats import t as student_t
from model import infer,plant,covariance
from prob_metrics import state_scores,forecast_scores
from theory_audit import aliases

def fitted_audit(theta,delay,method,effective):
 # Does not use test hidden values or true parameters.
 target=np.array(theta).copy();C0=covariance(target,delay,effective,False);kept=[];boundary=bool(min(target[:3])<1e-6 or min(abs(target[i]-target[j]) for i,j in [(0,1),(0,2),(1,2)])<1e-5)
 if boundary:candidates=[target.copy()]
 else:
  try:candidates=aliases(target)
  except (ArithmeticError,ValueError,np.linalg.LinAlgError):candidates=[target.copy()];boundary=True
 if not any(np.max(abs(a[:7]-target[:7]))<1e-6 for a in candidates):candidates.append(target.copy())
 for alt in candidates:
  if method=='E1' and not (alt[0]>alt[1]>alt[2]):continue
  if method=='E2' and abs(alt[2]-target[2])>1e-7:continue
  for lag in DELAYS:
   a=alt.copy()
   if effective=='E3':a[9]=target[9]*target[3]*target[4]/(a[3]*a[4])
   if effective.startswith('E4'):
    p=plant(a,lag,'E0')[3];q=plant(target,delay,'E0')[3]
    if effective in ['E4Y','E4YZ']:a[7]=target[7]*np.sqrt(q[0,0]/p[0,0])
    if effective in ['E4Z','E4YZ']:a[8]=target[8]*np.sqrt(q[1,1]/p[1,1])
   C=covariance(a,lag,effective,False);err=float(np.max(abs(C-C0))/max(1.,np.max(abs(C0))))
   if err<1e-7 and not any(lag==v['delay'] and np.max(abs(a-v['theta']))<1e-5 for v in kept):kept.append(dict(theta=a.tolist(),delay=lag,error=err))
 free=[0,1,2,3,4,5,6]
 if method=='E2':free.remove(2)
 if effective=='E3':free.append(9)
 if effective in ['E4Y','E4YZ']:free.append(7)
 if effective in ['E4Z','E4YZ']:free.append(8)
 C,D=covariance(target,delay,effective,True);L=np.linalg.cholesky(C);W=[]
 for k in free:W.append((np.linalg.solve(L,np.linalg.solve(L,D[k]).T).T*max(abs(target[k]),.05)).ravel())
 sv=np.linalg.svd(np.array(W).T,compute_uv=False);rank=int(np.sum(sv>sv[0]*1e-9));return dict(retained_known_equivalents=len(kept),retained_pole_orders=len({tuple(np.round(v['theta'][:3],7)) for v in kept}),retained_delays=sorted({v['delay'] for v in kept}),boundary_or_repeated_poles=boundary,local_rank=rank,local_parameters=len(free),known_family_unique=bool(len(kept)==1 and not boundary and rank==len(free)),candidates=kept,not_global_identifiability_proof=True)

def main():
 freeze=json.loads((ROOT/'ALL_PERMITTED_MODELS_FROZEN.json').read_text());assert freeze['cases']==36
 for f,h in freeze['files'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
 ereg=json.loads((ROOT/'EVALUATION_REGISTERED.json').read_text());assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in ereg['files'].items())
 private=json.loads((DATA/'sealed_manifest.json').read_text());rows=[];cases=[];bins=[];ident=[];params=[]
 for meta in private:
  name=meta['case'];dest=OUT/name;dat=DATA/name;x=np.load(dat/'X.npy')[-TEST:];truth=np.load(dat/'sealed_states.npz')['states'][-TEST:,:,0:2];I=np.load(dat/'known_input.npy')[-TEST:];p=meta['params'];base=np.array([p['a'],p['b'],p['c'],p['u'],p['v'],p['qx'],p['R'],1.,1.,1.]);marg=base.copy();source_scale=np.sqrt(1+INPUT_STD**2) if meta['policy']=='active' else 1.;marg[3]*=source_scale;mx=infer(x,marg,p['delay'],'E0');mx['mean'][:,:,0]*=source_scale;mx['cov'][:,0,:]*=source_scale;mx['cov'][:,:,0]*=source_scale;prior=plant(marg,p['delay'],'E0')[3][:2,:2];prior[0,:]*=source_scale;prior[:,0]*=source_scale
  mi=infer(x,base,p['delay'],'E3',I) if meta['policy']=='active' else mx;full=np.load(dest/'permitted_predictions.npz');common=dict(case=name,tag=meta['tag'],policy=meta['policy'],seed=meta['seed'],valid_order=meta['valid_order']);example=dict(time=np.arange(START,499),truth=truth[0,START:499],X=x[0,START:499],I=I[0,START:499])
  methods={m:(full[m+'_mean'][:,START:499],full[m+'_cov'][START:499]) for m in METHODS};methods['C_X']=(mx['mean'][:,START:499],mx['cov'][START:499]);methods['C_XI']=(mi['mean'][:,START:499],mi['cov'][START:499])
  for m,(mu,C) in methods.items():
   rr,cc,bb,ex=state_scores(truth[:,START:499],mu,C,prior,common,m);rows.extend(rr);cases.extend(cc);bins.extend(bb)
   for key,v in ex.items():example[m+'_'+key]=v
  for m in METHODS:
   record=json.loads((dest/f'{m}_models.json').read_text());s=record['selected'];theta=np.array(s['theta']);lag=s['delay'];effective=record.get('effective_model','E0');audit=fitted_audit(theta,lag,m,effective)
   if effective=='E0' or m in ['E0','E1','E2']:
    val_delays=list(DELAYS)
   else:
    records=record['candidates'];chosen={dd:max([r for r in records if r['delay']==dd],key=lambda r:r['validation_LL']) for dd in DELAYS};ref=np.array(chosen[lag]['validation_trajectory_LL']);val_delays=[lag];critical=student_t.ppf(1-.05/6,VALID-1)
    for dd in DELAYS:
     if dd==lag:continue
     diff=ref-np.array(chosen[dd]['validation_trajectory_LL']);lower=diff.mean()-critical*diff.std(ddof=1)/np.sqrt(VALID)
     if lower<=0:val_delays.append(dd)
    val_delays=sorted(val_delays)
   save=dict(validation_compatible_delays=json.dumps(val_delays),validation_unique_delay_correct=val_delays==[p['delay']],**common,method=m,true_delay=p['delay'],selected_delay=lag,forced_delay_correct=lag==p['delay'],delay_uniquely_retained=len(audit['retained_delays'])==1,unique_delay_correct=audit['retained_delays']==[p['delay']],**{k:v for k,v in audit.items() if k!='candidates'});ident.append(save);(dest/f'{m}_equivalence_audit.json').write_text(json.dumps(audit,indent=2));params.append(dict(**common,method=m,**{k:float(v) for k,v in zip(['a','b','c','u','v','qx','R','hY','hZ','lambda'],theta)}));pr=full[m+'_forecast'][:,START:499];var=full[m+'_forecast_var'][START:499];rr,cc=forecast_scores(x[:,START:499],pr,var,common,m);rows.extend(rr);cases.append(cc)
   if m in ['E0','E1','E2']:
    for d in [2,3]:example[m+f'_d{d}_mean']=full[m+f'_d{d}_mean'][0,START:499]
  for m,v in [('C_X',mx),('C_XI',mi)]:
   rr,cc=forecast_scores(x[:,START:499],v['forecast'][:,START:499],v['forecast_var'][START:499],common,m);rows.extend(rr);cases.append(cc)
  np.savez_compressed(dest/'evaluation_examples.npz',**example);print('EVALUATED',name,flush=True)
 pd.DataFrame(rows).to_csv(OUT/'trajectory_metrics.csv.gz',index=False);pd.DataFrame(cases).to_csv(OUT/'case_metrics.csv',index=False);pd.DataFrame(bins).to_csv(OUT/'time_calibration.csv',index=False);pd.DataFrame(ident).to_csv(OUT/'identification.csv',index=False);pd.DataFrame(params).to_csv(OUT/'estimated_parameters.csv',index=False);(ROOT/'EVALUATION_DONE.json').write_text(json.dumps(dict(time=time.time(),cases=len(private),trajectory_metric_rows=len(rows),parameters_not_modified=True),indent=2));print('ALL EVALUATED',len(rows),flush=True)
if __name__=='__main__':main()

import os,json,csv,time,hashlib
from datetime import datetime,timezone
import numpy as np
from scipy.special import logsumexp
from config import *
from mechanisms import library,active_library,response
from generate import draw,normal,equivalent_pair
from inference import evidence,posterior,active_evidence,confidence
from source_A import fit_normal
from legacy_stream import structured_scores,reference_scores,calibrate,alarms
from warning_detectors import generic_scores

def log(s):print(datetime.now(timezone.utc).isoformat(),s,flush=True)
def dump(p,x):p.write_text(json.dumps(x,indent=2))
def scorewarnings(x,bg):
 out={'M3':structured_scores(x,bg,0)['M3'],'GLR':generic_scores(x,bg)['GLR_Ramp']};out['failure_reference']=reference_scores(x,bg)['M4_E'];return out

def prepare():
 dump(ROOT/'PROTOCOL.json',PROTOCOL)
 settings=[]
 for ni,sigma in enumerate(SIGMAS):
  bg=fit_normal(normal(N_NORMAL,sigma,SEEDS['normal_train'][ni]));bank=library(bg)
  train=draw(N_TRAIN_PER_CLASS,sigma,1,SEEDS['unlabelled_train'][ni],train=True)
  pooled=np.concatenate([train[m]['x'] for m in LABELS]);np.random.default_rng(88700+ni).shuffle(pooled)
  vals=[float(logsumexp(evidence(pooled,bank,bg,s),axis=1).sum()-len(pooled)*np.log(4)) for s in AMP_PRIOR_CANDIDATES];amp=AMP_PRIOR_CANDIDATES[int(np.argmax(vals))]
  rng=np.random.default_rng(SEEDS['sensor_train'][ni]);vvars=[float(np.mean((v*rng.normal(size=(N_SENSOR_NORMAL,T)))**2)) for v in SENSOR]
  thresholds={m:calibrate(sc,2) for m,sc in scorewarnings(normal(N_CAL,sigma,SEEDS['warning_calibration'][ni]),bg).items()}
  audits={}
  for m,sc in scorewarnings(normal(N_NULL,sigma,SEEDS['warning_null'][ni]),bg).items():
   ts,_=alarms(sc,thresholds[m],2);audits[m]=dict(n=N_NULL,count=int(np.isfinite(ts).sum()),fpr=float(np.isfinite(ts).mean()))
  settings.append(dict(bg=bg,amp_prior=amp,training_mixture_evidence=vals,sensor_variances=vvars,thresholds=thresholds,null_audit=audits));dump(OUT/'trained_settings.json',settings);log(f'Calibration {ni+1}/3 complete: {audits}')
 return settings

records=[];warn=[];latent=[]
def record(family,obs,logev,true,ni,co,k,extra=None):
 pp=posterior(logev);pred,mx,abst,size=confidence(pp)
 for i in range(len(pp)):
  row=dict(family=family,observation=obs,noise=SIGMAS[ni],cohort=co,kappa=k,trajectory=i,truth=true,prediction=int(pred[i]),correct=int(pred[i]==true),max_posterior=float(mx[i]),abstain=int(abst[i]),credible_size=int(size[i]),true_log_odds=float(logev[i,true]-logsumexp(np.delete(logev[i],true))))
  for j in range(pp.shape[1]):row['p'+str(j)]=float(pp[i,j])
  if extra is not None:row.update(extra[i])
  records.append(row)

def test(settings):
 # Marker hash covers inference/generation/protocol and learned settings before reading test samples.
 names=['config.py','mechanisms.py','generate.py','inference.py','run.py','PROTOCOL.json','results/trained_settings.json']
 dump(OUT/'TEST_START.json',dict(time=datetime.now(timezone.utc).isoformat(),sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in names}))
 for ni,sigma in enumerate(SIGMAS):
  ss=settings[ni];bg=ss['bg'];bank=library(bg);abank=active_library(bg);eqbank={'A_source_step':bank['A_first'],'E_source_smooth':dict(bank['A_first'],y=bank['A_first']['u'])}
  for co in COHORTS:
   for k in KTEST:
    data=draw(N,sigma,k,SEEDS['passive'][co][ni])
    for truth,m in enumerate(LABELS):
     d=data[m];le,meta=evidence(d['x'],bank,bg,ss['amp_prior'],details=True);record('main_passive','X',le,truth,ni,co,k)
     pred=le.argmax(1)
     for i,p in enumerate(pred):
      item=meta[p];ii=item['index'][i];latent.append(dict(noise=sigma,cohort=co,kappa=k,trajectory=i,truth=truth,prediction=int(p),onset_estimate=float(bank[LABELS[p]]['on'][ii]),onset_error=float(abs(bank[LABELS[p]]['on'][ii]-10)),kappa_estimate=float(bank[LABELS[p]]['k'][ii]) if p!=1 else None))
     record('main_passive','X_prefix15',evidence(d['x'][:,:301],bank,bg,ss['amp_prior']),truth,ni,co,k)
     for si,sv in enumerate(SENSOR):record('main_passive',f'X_VU_{sv}',evidence(d['x'],bank,bg,ss['amp_prior'],d['U']+sv*d['eta'],ss['sensor_variances'][si]),truth,ni,co,k)
     ws=scorewarnings(d['x'],bg);ft,_=alarms(ws['failure_reference'],ss['thresholds']['failure_reference'],2)
     for method in ['M3','GLR']:
      at,_=alarms(ws[method],ss['thresholds'][method],2)
      for i in range(N):
       valid=np.isfinite(at[i]) and at[i]>=10;fr=np.isfinite(ft[i]) and ft[i]>=10
       warn.append(dict(noise=sigma,cohort=co,kappa=k,trajectory=i,truth=truth,method=method,alarm=None if not np.isfinite(at[i]) else float(at[i]),failure_reference=None if not np.isfinite(ft[i]) else float(ft[i]),early_false_alarm=int(np.isfinite(at[i]) and at[i]<10),detected=int(valid),delay=float(at[i]-10) if valid else None,lead=float(ft[i]-at[i]) if valid and fr else None,**{f'cover_{margin}':int(valid and fr and ft[i]-at[i]>=margin) for margin in [.25,.5,1.]}))
    a,b=equivalent_pair(N,sigma,k,SEEDS['equivalent'][co][ni])
    for truth,d in enumerate([a,b]):
     record('causal_equivalent','X',evidence(d['x'],eqbank,bg,ss['amp_prior']),truth,ni,co,k)
     for si,sv in enumerate(SENSOR):
      record('causal_equivalent',f'X_VU_{sv}',evidence(d['x'],eqbank,bg,ss['amp_prior'],d['U']+sv*d['eta'],ss['sensor_variances'][si]),truth,ni,co,k)
      record('causal_equivalent',f'X_VY_{sv}',evidence(d['x'],eqbank,bg,ss['amp_prior'],d['Y']+sv*d['eta'],ss['sensor_variances'][si],sensor='y'),truth,ni,co,k)
    active=draw(N,sigma,k,SEEDS['intervention'][co][ni],mode='active')
    for truth,m in enumerate(LABELS):
     d=active[m];hidden=active_evidence(d['x'],abank,bg);known=active_evidence(d['x'],abank,bg,d['protocol']);record('main_active','X_input_hidden',hidden,truth,ni,co,k);record('main_active','X_input_known',known,truth,ni,co,k)
     if truth<2:
      record('causal_active','X_input_hidden',hidden[:,:2],truth,ni,co,k);record('causal_active','X_input_known',known[:,:2],truth,ni,co,k)
    # Save all underlying X / hidden answers for independent replay; truth never passed to inference.
    np.savez_compressed(OUT/f'trajectories_n{ni}_c{co}_k{k}.npz',**{f'{mode}_{m}_{field}':v for mode,dd in [('passive',data),('active',active)] for m,d in dd.items() for field,v in d.items() if isinstance(v,np.ndarray)},equiv_x=a['x'],equiv_U=a['U'],equiv_Y_step=a['Y'],equiv_Y_smooth=b['Y'])
    log(f'Test noise={sigma}, cohort={co}, lag={k} complete; {len(records)} classifications')
 dump(OUT/'classifications.json',records);dump(OUT/'warnings.json',warn);dump(OUT/'latent_estimates.json',latent)

if __name__=='__main__':
 st=time.time();settings=prepare() if not (OUT/'trained_settings.json').exists() else json.loads((OUT/'trained_settings.json').read_text());test(settings);log(f'ALL COMPLETE {time.time()-st:.1f}s')

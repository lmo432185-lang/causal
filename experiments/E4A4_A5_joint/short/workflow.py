import copy,json,gzip,time,hashlib,resource,numpy as np
from pathlib import Path
from settings import *
from model import pack,unpack,PARAMS,em
from extension import proposals,fit_candidate,init_model,train
from inference import point_score,validate
from producer import generate,digest
from learn import fit_all

ROUTES=['direct','path','residual']
def old_bundle(j,x):
 import subprocess
 if j['index']<20:
  path='experiments/E4A_independent_state/results/'+j['case']+'/model_1000.json'
  data=subprocess.check_output(['git','show','38cda63b1ee470cdc53efa5abd86b2208028d921:'+path],cwd='/workspace/causal')
  return unpack(json.loads(data)),hashlib.sha256(data).hexdigest()
 b=fit_all(x[:20],x[20:40],x[40:50],j['seed']);return b,None

def fit_job(j):
 p=OUT/(j['case']+'.json.gz')
 if p.exists():return j['case'],'cached'
 x=generate(j['index'],j['seed'],0,70);hashes={n:digest(a) for n,a in [('R',x[:20]),('D',x[20:40]),('selection',x[40:50])]};assert hashes==json.loads((ROOT/'TRAINING_DATA_FROZEN.json').read_text())['hashes'][j['case']];b,oldhash=old_bundle(j,x);v=x[40:50];models={};meta={};plans={};candidates={};bases={};tic=time.process_time()
 # Legacy complete mixture, same algorithm, evaluated on NEW tests only.
 l_ids=[];l_prior=[]
 from scipy.special import logsumexp
 for bank,line in [('R','recursive'),('D','witness')]:
  kinds=[k for k,m in b[line]['models'].items() if m is not None];ev=np.array([b[line]['log_evidence'][k] for k in kinds]);ev-=logsumexp(ev)
  for k,pr in zip(kinds,ev):
   key='oldM4_'+bank+'_'+k;models[key]=b[line]['models'][k];l_ids.append(key);l_prior.append(float(pr-np.log(2)))
 plans['J0']=dict(ids=l_ids,prior=l_prior)
 m_ids=[];m_prior=[]
 for bank,line in [('R','direct'),('D','witness')]:
  kinds=[k for k,m in b[line]['models'].items() if m is not None];ev=np.array([b[line]['log_evidence'][k] for k in kinds]);ev-=logsumexp(ev)
  for k,pr in zip(kinds,ev):
   key='oldM3_'+bank+'_'+k;models[key]=b[line]['models'][k];m_ids.append(key);m_prior.append(float(pr-np.log(2)))
 plans['M3']=dict(ids=m_ids,prior=m_prior);plans['M1']=dict(ids=m_ids,prior=m_prior,update=False)
 for bank,line,xx in [('R','direct',x[:20]),('D','witness',x[20:40])]:
  oldmodels=b[line]['models'];n=xx.size;single=['factor1','memory2','memory3'];parent=max(single,key=lambda k:oldmodels[k]['train_mean_loglik']*n-.5*PARAMS[k]*np.log(n));base=oldmodels[parent];bases[bank]=dict(parent=parent,model=base);pr=proposals(xx,base,j['seed']+len(bank)*17);meta[bank]=dict(path_valid=pr['path_valid'],residual_r=pr['residual_r'],cpu={},failures={});keys={}
  for kind,m in oldmodels.items():
   if m is not None:models[bank+'_old_'+kind]=m
  none=bank+'_none';models[none]=base;mem,md=fit_candidate(xx,base,'obs','direct',pr,j['seed']);memkey=bank+'_obs'
  if mem is not None:models[memkey]=mem
  meta[bank]['cpu']['obs']=md
  for route in ROUTES:
   keys[route]=[]
   for mode in ['latent','both']:
    key=bank+'_'+route+'_'+mode;m,diag=fit_candidate(xx,base,mode,route,pr,j['seed']+101)
    meta[bank]['cpu'][route+'_'+mode]=diag
    if m is not None:models[key]=m;keys[route].append(key)
  for route in ['direct','joint']:
   key=bank+'_'+route+'_dense';m,diag=fit_candidate(xx,base,'dense',route,pr,j['seed']+211);meta[bank]['cpu'][route+'_dense']=diag
   if m is not None:models[key]=m
  score={key:point_score(v,m) for key,m in models.items() if key.startswith(bank+'_')};common=[none]+([memkey] if memkey in models else [])
  direct_keys=[bank+'_old_'+kind for kind in oldmodels if oldmodels[kind] is not None]+common+keys['direct']+([bank+'_direct_dense'] if bank+'_direct_dense' in models else [])
  path_old=b['recursive']['models']['factor2'] if bank=='R' else None
  if path_old is not None:key=bank+'_original_ffbs';models[key]=path_old;score[key]=point_score(v,path_old);common_path=common+[key]
  else:common_path=common
  for name,ks in [('R0',direct_keys),('R1',common_path+keys['path']),('R2',common+keys['residual']),('R3',[bank+'_joint_dense'] if bank+'_joint_dense' in models else [])]:
   winner=min(ks,key=lambda k:score[k]) if ks else None;candidates.setdefault(name,{})[bank]=dict(best=winner,novel=min(keys['path' if name=='R1' else 'residual'],key=lambda k:score[k]) if name in ['R1','R2'] and keys['path' if name=='R1' else 'residual'] else winner if name=='R3' else None)
  # Same finite prior size and bootstrap training samples for single/dual-line groups.
  particle_ids={'A':[],'B':[],'E':[]}
  for st,mode in enumerate(['none','obs','latent','dense']):
   for rep in range(2):
    rng=np.random.default_rng(j['seed']+st*779+rep*371+ord(bank)*1111);ix=rng.integers(0,len(xx),len(xx));xb=xx[ix];bp=em(xb/base['scale'],copy.deepcopy(base),maxiter=BOOT_ITERS);bp['scale']=base['scale'];pp=proposals(xb,bp,j['seed']+st*37+rep)
    for alg,route in [('A','direct'),('B','joint' if mode=='dense' else 'residual')]:
     key=f'{bank}_particle_{alg}_{st}_{rep}'
     try:
      mm=bp if mode=='none' else train(xb,init_model(bp,mode,route,pp,xb,j['seed']+st,rep),maxiter=BOOT_ITERS)
      if mm is None:raise ValueError('invalid particle')
      models[key]=mm;particle_ids[alg].append(key)
      if (alg=='A' and rep==0) or (alg=='B' and rep==1):particle_ids['E'].append(key)
     except (ValueError,AssertionError,np.linalg.LinAlgError,FloatingPointError) as ex:meta[bank]['failures'][key]=type(ex).__name__
  for method,ids in particle_ids.items():plans.setdefault(method,dict(ids=[],prior=[]));plans[method]['ids'].extend(ids)
  # Canonical physical-state uncertainty: bootstrap parameters only, no physical labels.
  params=[]
  for rep in range(4):
   rng=np.random.default_rng(j['seed']+rep*271+ord(bank)*571);ix=rng.integers(0,len(xx),len(xx));m0=oldmodels['factor2'];mm=em(xx[ix]/m0['scale'],copy.deepcopy(m0),maxiter=BOOT_ITERS);mm['scale']=m0['scale'];params.append(mm)
  bases[bank]['canonical_bootstrap']=params;bases[bank]['canonical_point']=oldmodels['factor2']
 for name in ['R0','R1','R2','R3']:
  ids=[candidates[name][z]['best'] for z in ['R','D'] if candidates[name][z]['best'] is not None];plans[name]=dict(ids=ids,prior=[-np.log(len(ids))]*len(ids))
 plans['C']=dict(ids=[candidates[z][bank]['best'] for bank in ['R','D'] for z in ['R0','R2'] if candidates[z][bank]['best'] is not None]);plans['D']=plans['R3'];plans['J1']=plans['D'];plans['J2']=plans['E']
 val={}
 for name in ['A','B','C','D','E']:
  ids=plans[name]['ids']
  if not ids:plans[name]['prior']=[];val[name]=np.inf;continue
  score,post=validate(v,models,ids);plans[name]['prior']=post.tolist();val[name]=score
 plans['J2']=plans['E'];plans['BestSingle']=plans['A' if val['A']<=val['B'] else 'B'];plans['BestSingle']['selected_algorithm']='A' if val['A']<=val['B'] else 'B'
 # Parameter particles receive selection likelihood once, not squared across routes.
 for name in ['R0','R1','R2','R3','J1']:
  if plans[name]['ids']:_,post=validate(v,models,plans[name]['ids']);plans[name]['prior']=post.tolist()
 result=dict(case=j['case'],seed=j['seed'],index=j['index'],models=models,plans=plans,candidates=candidates,bases=bases,diagnostics=meta,validation=val,old_model_sha256=oldhash,training_hashes=hashes,cpu=time.process_time()-tic,peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 p.parent.mkdir(exist_ok=True);tmp=p.with_suffix('.tmp')
 with gzip.open(tmp,'wt') as f:json.dump(pack(result),f)
 tmp.replace(p);print('FITTED',j['case'],round(result['cpu'],2),flush=True);return j['case'],result['cpu']

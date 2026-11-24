import json,gzip,csv,time,hashlib,numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
from settings import *
from producer import generate,digest
from model import unpack
from inference import mix
from predict import scores,component,history_scores
METHODS=['M1','M3','J0','R0','R1','R2','R3','A','B','C','D','E','BestSingle','J1','J2','R1_original','J1_param','J2_point','StrongestSingle','Xhistory']

def work(j,batch):
 with gzip.open(OUT/(j['case']+'.json.gz'),'rt') as f:b=unpack(json.load(f))
 b['plans'].update(json.loads((OUT/'extra_plans.json').read_text())[j['case']]);x=generate(j['index'],j['seed'],11 if batch=='A' else 12,NTEST);cache={};computed={};rows=[];controls=[];traces={}
 for name in METHODS:
  plan=b['plans']['J0' if name=='R1_original' else name];ids=plan['ids'];pr=plan.get('prior');update=plan.get('update',True)
  if plan.get('type')=='ARhistory':
   sc=history_scores(x,plan['parameters']);sc['particle_ESS']=np.full(NTEST,np.nan);sc['model_entropy']=np.full(NTEST,np.nan)
   for i in range(NTEST):rows.append(dict(case=j['case'],seed=j['seed'],batch=batch,method=name,trajectory=i,available=True,**{k:format(float(v[i]),'.12g') for k,v in sc.items()}))
   continue
  if not ids:
   for i in range(NTEST):rows.append(dict(case=j['case'],seed=j['seed'],batch=batch,method=name,trajectory=i,available=False,NLL=np.nan,NLL_h1=np.nan,NLL_h2=np.nan,NLL_h5=np.nan,NLL_h10=np.nan,NLL_h20=np.nan,CRPS=np.nan,cover90=np.nan,width90=np.nan,MSE=np.nan,particle_ESS=np.nan,model_entropy=np.nan))
   continue
  signature=(tuple(ids),tuple(pr or []),update)
  if signature not in computed:
   r,post=mix(x,b['models'],ids,pr,cache,update=update);sc=scores(x,r);sc['particle_ESS']=np.mean(1/np.sum(r['weights'][:,200:]**2,2),1);sc['model_entropy']=np.mean(-np.sum(r['weights'][:,200:]*np.log(np.maximum(r['weights'][:,200:],1e-300)),2),1);computed[signature]=(sc,r)
  sc,r=computed[signature]
  for i in range(NTEST):rows.append(dict(case=j['case'],seed=j['seed'],batch=batch,method=name,trajectory=i,available=True,**{k:format(float(v[i]),'.12g') for k,v in sc.items()}))
  if j['seed']==4011 and name in ['J0','J1','J2','A','B']:
   traces['X']=x[0];traces['origins']=np.array(origins(5000));traces[name+'_weights']=r['weights'][0,np.array(origins(5000))];traces[name+'_forecast']=np.sum(r['origin_weights'][0,:,:,None]*r['mean'][0],1)
   if name=='J1':
    for k,p in enumerate(r['parts']):traces[f'J1_state_{k}_mean']=p['post']['mean'][0];traces[f'J1_state_{k}_cov']=p['post']['cov']
 for bank in ['R','D']:
  for kind in ['none','obs','direct_latent','direct_both','direct_dense','path_latent','path_both','residual_latent','residual_both','joint_dense']:
   key=bank+'_'+kind
   if key not in b['models']:controls.append(dict(case=j['case'],seed=j['seed'],batch=batch,bank=bank,control=kind,trajectory=-1,available=False,NLL=np.nan,CRPS=np.nan,cover90=np.nan,width90=np.nan,MSE=np.nan));continue
   r,_=mix(x,b['models'],[key],cache=cache);sc=scores(x,r)
   for i in range(NTEST):controls.append(dict(case=j['case'],seed=j['seed'],batch=batch,bank=bank,control=kind,trajectory=i,available=True,**{k:format(float(sc[k][i]),'.12g') for k in ['NLL','CRPS','cover90','width90','MSE']}))
 if traces:np.savez_compressed(OUT/(j['case']+'_example_'+batch+'.npz'),**traces)
 print('TEST',batch,j['case'],flush=True);return rows,controls,digest(x)

def main():
 for label in ['CODE_REGISTERED','PREDICTION_REGISTERED','CANDIDATES_FROZEN','FINAL_MODELS_FROZEN']:
  for f,h in json.loads((ROOT/(label+'.json')).read_text()).get('files',{}).items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
 js=json.loads((ROOT/'jobs.json').read_text())
 for batch in ['A','B']:
  path=OUT/f'test_{batch}.csv.gz';cp=OUT/f'controls_{batch}.csv.gz';assert not path.exists();hashes={};count=0
  with gzip.open(path,'wt') as out,gzip.open(cp,'wt') as cs:
   writer=None;cw=None
   with ProcessPoolExecutor(max_workers=3) as pool:
    fs={pool.submit(work,j,batch):j for j in js}
    for f in as_completed(fs):
     a,c,h=f.result();hashes[fs[f]['case']]=h
     if writer is None:writer=csv.DictWriter(out,fieldnames=a[0]);writer.writeheader();cw=csv.DictWriter(cs,fieldnames=c[0]);cw.writeheader()
     writer.writerows(a);cw.writerows(c);count+=len(a)
  files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,cp]};files.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*_example_'+batch+'.npz')});(ROOT/('TEST_'+batch+'_FROZEN.json')).write_text(json.dumps(dict(time=time.time(),files=files,input_hashes=hashes,rows=count,truth_not_opened=True),indent=2));print('TEST FROZEN',batch,count,flush=True)
if __name__=='__main__':main()

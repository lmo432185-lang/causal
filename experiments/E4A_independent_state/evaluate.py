"""Frozen prediction policies; scores always precede use of actual future observations."""
import json,time,hashlib,gzip,csv,numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
from settings import *
from producer import generate,digest
from model import unpack
from predict import infer,scores,history_scores,component,targets,logpdf

def work(j,batch):
 xlong=generate(j['index'],j['seed'],1 if batch=='A' else 2,NTEST);rows=[];extras=[];trace={}
 for length in LENGTHS:
  x=xlong[:,:length];b=unpack(json.loads((OUT/j['case']/f'model_{length}.json').read_text()));cache={}
  for group in ['M1','M2','M3','M4']:
   r=infer(x,b,group,cache);sc=scores(x,r);counts=np.array([b[line]['models'][kind]['factor_count'] for line,kind in r['ids']]);dims=np.array([b[line]['models'][kind]['state_dimension'] for line,kind in r['ids']]);sc['working_factor_count']=(r['weights'][:,200:]@counts).mean(1);sc['Markov_dimension']=(r['weights'][:,200:]@dims).mean(1);sc['model_entropy']=(-np.sum(r['weights'][:,200:]*np.log(np.maximum(r['weights'][:,200:],1e-300)),2)).mean(1);ambiguity=(r['weights'][:,:,counts==2].sum(2));sc['two_mode_alias_mass']=ambiguity[:,200:].mean(1)
   for i in range(NTEST):rows.append(dict(case=j['case'],seed=j['seed'],length=length,batch=batch,group=group,trajectory=i,**{k:float(v[i]) for k,v in sc.items()}))
   if length==5000 and j['seed']==4011:
    trace['X']=x[0];trace['origins']=np.array(origins(length));trace[group+'_weights']=r['weights'][0];trace[group+'_forecast_mean']=np.sum(r['origin_weights'][0,:,:,None]*r['mean'][0],1);trace[group+'_ids']=np.array([KINDS.index(kind)+(0 if line in ['direct','recursive'] else len(KINDS)) for line,kind in r['ids']]);
    if group=='M4':
     for ci,c in enumerate(r['parts']):
      trace[f'component_{ci}_mean']=c['post']['mean'][0];trace[f'component_{ci}_cov']=c['post']['cov']
  sc=history_scores(x,b['history'])
  for i in range(NTEST):rows.append(dict(case=j['case'],seed=j['seed'],length=length,batch=batch,group='Xhistory',trajectory=i,**{k:float(v[i]) for k,v in sc.items()},working_factor_count=np.nan,Markov_dimension=np.nan,model_entropy=np.nan,two_mode_alias_mass=np.nan))
  # Fixed per-family Gaussian scores, including failed proposals; no success filtering.
  y=targets(x)
  for line in ['direct','recursive','witness']:
   for kind in KINDS:
    m=b[line]['models'][kind]
    if m is None:continue
    key=(line,kind,True)
    if key not in cache:cache[key]=component(x,m)
    c=cache[key];v=-logpdf(c['mean'],c['cov'],y).mean(1)/20
    for i in range(NTEST):extras.append(dict(case=j['case'],seed=j['seed'],length=length,batch=batch,line=line,kind=kind,trajectory=i,NLL=float(v[i])))
 if trace:np.savez_compressed(OUT/j['case']/f'example_{batch}.npz',**trace)
 print('TEST',batch,j['case'],flush=True);return rows,extras,digest(xlong)

def main():
 for label in ['CODE_REGISTERED','MODELS_FROZEN','DECISIONS_FROZEN']:
  for f,h in json.loads((ROOT/(label+'.json')).read_text())['files'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
 js=json.loads((ROOT/'jobs.json').read_text())
 for batch in ['A','B']:
  path=OUT/f'test_{batch}.csv.gz';extra=OUT/f'family_{batch}.csv.gz';assert not path.exists() and not extra.exists();count=0;hashes={}
  with gzip.open(path,'wt') as stream,gzip.open(extra,'wt') as ef:
   writer=None;ew=None
   with ProcessPoolExecutor(max_workers=3) as pool:
    fs={pool.submit(work,j,batch):j for j in js}
    for f in as_completed(fs):
     rows,ex,h=f.result();hashes[fs[f]['case']]=h
     if writer is None:writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();ew=csv.DictWriter(ef,fieldnames=ex[0]);ew.writeheader()
     writer.writerows(rows);ew.writerows(ex);count+=len(rows)
  files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,extra]};files.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob(f'*/example_{batch}.npz')});(ROOT/f'TEST_{batch}_FROZEN.json').write_text(json.dumps(dict(time=time.time(),files=files,rows=count,input_hashes=hashes,true_states_not_read=True),indent=2));print('TEST FROZEN',batch,count,flush=True)
if __name__=='__main__':main()

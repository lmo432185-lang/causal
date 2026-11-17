"""X-only workers. No imports of private generator configuration or truth loader."""
import json,time,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import numpy as np
from scipy.special import softmax
from settings import ROOT,OUT,NSTART,MAXITER,START,CONF
from ssm import fit,estep,mirror,serialize
from equivalence import counterpart

def fit_job(job):
 begin=time.time();case=job['case'];dest=OUT/case;dest.mkdir(exist_ok=True)
 if (dest/'LOCAL_X_FREEZE.json').exists():
  fr=json.loads((dest/'LOCAL_X_FREEZE.json').read_text());assert hashlib.sha256((dest/'selected_parameters.json').read_bytes()).hexdigest()==fr['parameter_sha256'];assert hashlib.sha256((dest/'X_only_decisions.json').read_bytes()).hexdigest()==fr['decision_sha256'];print(case,'cached complete; unchanged',flush=True);return case
 x=np.load(ROOT/job['X_path']);mu=float(x[:600].mean());sd=float(x[:600].std());y=(x-mu)/sd;tr,va,te=y[:600],y[600:800],y[800:];allfits={};best={};records=[]
 for ki,kind in enumerate(['G1','G2','G3','single']):
  cached=dest/f'{kind}_initializations.json'
  if cached.exists():
   starts=json.loads(cached.read_text());records.extend([dict(case=case,kind=kind,init=a['init'],delay=a['parameters']['delay'],iterations=a['fit']['iterations'],stop=a['fit']['stop'],validation_score=a['validation_penalized']) for a in starts])
  else:
   starts=[]
   for init in range(NSTART):
    delay=1 if kind!='G3' or init<2 else 2;optseed=int(np.random.SeedSequence([981000,job['seed'],int(case[4:7]),ki,init]).generate_state(1)[0]);p,log=fit(tr,kind,delay,optseed,maxiter=MAXITER);val=estep(va,p,stats=False)['trajectory_ll'];k=5 if kind=='single' else 8 if kind=='G3' else 7;score=float(val.sum()-.5*k*np.log(200*900)-(np.log(2) if kind=='G3' else 0));starts.append(dict(parameters=serialize(p),fit=log,validation_total=float(val.sum()),validation_penalized=score,init=init));records.append(dict(case=case,kind=kind,init=init,delay=delay,iterations=log['iterations'],stop=log['stop'],validation_score=score))
  win=max(starts,key=lambda q:q['validation_penalized']);from ssm import restore
  p=restore(win['parameters']);best[kind]=p;allfits[kind]=starts
  (dest/f'{kind}_initializations.json').write_text(json.dumps(starts,indent=2))
 best['G4']=mirror(best['G2']);allfits['G4']=[{**a,'parameters':serialize(mirror(restore(a['parameters']))),'fit':{**a['fit'],'source':'exact EM-equivalent mirror of G2'}} for a in allfits['G2']];(dest/'G4_initializations.json').write_text(json.dumps(allfits['G4'],indent=2))
 chosen={k:serialize(v) for k,v in best.items()};(dest/'selected_parameters.json').write_text(json.dumps(dict(mu=mu,sd=sd,models=chosen),indent=2));modelhash=hashlib.sha256((dest/'selected_parameters.json').read_bytes()).hexdigest();llva={k:estep(va,p,stats=False)['trajectory_ll'] for k,p in best.items()};top=['G1','G2','G3','G4'];df=np.array([7,7,8,7]);vscore=np.array([llva[k].sum()-.5*df[i]*np.log(200*900)-(np.log(2) if k=='G3' else 0) for i,k in enumerate(top)]);winner=int(vscore.argmax());compatible=[]
 rng=np.random.default_rng(982000+job['seed']+int(case[4:7]));vbest=llva[top[winner]]
 for k in top:
  delta=vbest-llva[k];bs=delta[rng.integers(0,200,(1000,200))].mean(1);compatible.append(bool(np.quantile(bs,.025)<=0 or delta.mean()/900<1e-4))
 compatible[winner]=True;alts={k:counterpart(best[k]) for k in top};lltest={k:estep(te,p,stats=False)['trajectory_ll'] for k,p in best.items()};llmatrix=np.stack([lltest[k] for k in top],1);mirror_error=float(np.max(abs(llmatrix[:,1]-llmatrix[:,3])));assert mirror_error<1e-5;llmatrix[:,[1,3]]=llmatrix[:,[1,3]].mean(1)[:,None];scores=llmatrix-.5*df[None,:]*np.log(900);scores[:,2]-=np.log(2);post=softmax(scores,axis=1);pred=post.argmax(1);conf=post.max(1);naive=conf>=CONF;guard=[];ambiguities=[];degenerate={}
 for k,p in best.items():
  if k=='single':continue
  a=p['A'];vals=[a[2,0],a[2,1]] if k=='G1' else [a[1,0],a[2,1]] if k=='G2' else [a[0,1],a[2,0]] if k=='G4' else [a[1,0 if p['delay']==1 else 3],a[2,0],a[2,1]];degenerate[k]=bool(min(vals)<1e-3)
 for i,j in enumerate(pred):
  family=top[j];ambiguity={family}|{v['label'] for v in alts[family]}|{k for k,c in zip(top,compatible) if c};amb=sorted(ambiguity);ambiguities.append(amb);guard.append(bool(naive[i] and len(amb)==1 and not degenerate[family]))
 decisions=dict(case=case,topologies=top,posterior=post.tolist(),prediction=[top[j] for j in pred],naive_declared=naive.tolist(),guarded_declared=guard,compatible_validation=compatible,ambiguities=ambiguities,degenerate=degenerate,estimated_equivalent_alternatives=alts,mirror_loglik_error=mirror_error,single_test_LL=lltest['single'].tolist(),test_LL=llmatrix.tolist(),selected_by_validation=top[winner]);(dest/'X_only_decisions.json').write_text(json.dumps(decisions,indent=2));(dest/'LOCAL_X_FREEZE.json').write_text(json.dumps(dict(time=time.time(),parameter_sha256=modelhash,decision_sha256=hashlib.sha256((dest/'X_only_decisions.json').read_bytes()).hexdigest(),no_truth_read=True),indent=2));(dest/'fit_summary.json').write_text(json.dumps(records,indent=2));print(case,'XONLY complete',round(time.time()-begin,1),'sec',flush=True);return case

if __name__=='__main__':
 jobs=json.loads((ROOT/'public_jobs.json').read_text());done=[]
 with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool:
  for f in as_completed([pool.submit(fit_job,j) for j in jobs]):done.append(f.result());print('PROGRESS',len(done),'/',len(jobs),flush=True)
 manifest={}
 for j in jobs:
  for name in ['selected_parameters.json','X_only_decisions.json','LOCAL_X_FREEZE.json']:
   p=OUT/j['case']/name;manifest[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 (ROOT/'ALL_X_ONLY_FROZEN.json').write_text(json.dumps(dict(time=time.time(),files=manifest,case_count=len(done),hidden_truth_not_loaded_by_worker=True),indent=2));print('ALL X-ONLY FROZEN',flush=True)

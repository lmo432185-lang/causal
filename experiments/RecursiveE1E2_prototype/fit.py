"""Discovery on independent X splits. No confirmation/test file or hidden-state reads."""
import json,time,hashlib
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from settings import *
from core import *
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(pack(v),indent=2))

def candidate_feature(q,j):return np.concatenate([q['current'],q['W1'],q['A'][:,:,j:j+1]],-1)
def initial_feature(q):return np.concatenate([q['current'],q['W1']],-1)
def features_for(q,model):
 j=model['candidate_index'];r=model['rank_B'];axis=model['B_axis']
 return dict(current=q['current'],initial=initial_feature(q),long=np.concatenate([q['current'],q['W1long']],-1),recursive=candidate_feature(q,j),witness=np.concatenate([q['current'],q['W1'],q['B'][:,:,:r]@axis],-1),direct2=np.concatenate([q['current'],q['match'][:,:,:2]],-1),direct_auto=np.concatenate([q['current'],q['match'][:,:,:model['rank_match']]],-1),full_X=np.concatenate([q['current'],q['past']],-1),full_W1=np.concatenate([q['current'],q['W1'],q['Whistory']],-1),duplicate=np.concatenate([q['current'],q['W1'],2*q['W1']],-1),W1_parent=q['W1'],W1_split=np.concatenate([q['W1'],q['A'][:,:,j:j+1]],-1))

def fit_case(job):
 case=job['case'];d=OUT/case;d.mkdir(parents=True,exist_ok=True);data=DATA/case
 x0=np.load(data/'initial_X.npy');xa=np.load(data/'recursive_X.npy');xb=np.load(data/'direct_X.npy');xv=np.load(data/'selection_X.npy');c,p,y=base_arrays(x0,P);short=learn(c,p,y);c,p,y=base_arrays(x0,FULL);long=learn(c,p,y);c,p,y=base_arrays(xb,FULL);B=learn(c,p,y);c,p,y=base_arrays(np.concatenate([x0,xa]),FULL);match=learn(c,p,y)
 wa=wseries(xa,short);cur,past,future=recursive_arrays(wa);rec=learn(cur,past,future)
 q=feature_bundle(xa,short,long,rec,B,match);v=feature_bundle(xv,short,long,rec,B,match);rankinfo={};rankheads={}
 for label,key in [('initial','W1'),('B','B'),('match','match')]:
  trainstates=apply(short,q['current'],base_arrays(xa,P)[1]) if label=='initial' else q[key];valstates=apply(short,v['current'],base_arrays(xv,P)[1]) if label=='initial' else v[key];heads=[];scores=[]
  for k in range(KMAX+1):
   head=fit_head(np.concatenate([q['current'],trainstates[:,:,:k]],-1),q['future']);heads.append(head);scores.append(nll_series(head,np.concatenate([v['current'],valstates[:,:,:k]],-1),v['future']))
  rank,info=choose_rank(scores);rankinfo[label]=dict(selected=rank,**info);rankheads[label]=heads
 basehead=fit_head(initial_feature(q),q['future']);baseline=nll_series(basehead,initial_feature(v),v['future']).mean();Aheads=[];scores=[]
 for j in range(KMAX):
  head=fit_head(candidate_feature(q,j),q['future']);Aheads.append(head);scores.append(float(nll_series(head,candidate_feature(v,j),v['future']).mean()))
 j=int(np.argmin(scores));r=rankinfo['B']['selected'];baseq=affine(initial_feature(q));basev=affine(initial_feature(v));rawAq=q['A'][:,:,j:j+1];rawAv=v['A'][:,:,j:j+1];areg=regfit(baseq,rawAq);Aresq=rawAq-baseq@areg;Aresv=rawAv-basev@areg
 if r:
  breg=regfit(baseq,q['B'][:,:,:r]);Bresv=v['B'][:,:,:r]-basev@breg;axis=regfit(Bresv,Aresv)
 else:breg=np.zeros((3,0));axis=np.zeros((0,1))
 model=dict(case=case,seed=job['seed'],short=short,long=long,recursive_encoder=rec,B=B,match=match,candidate_index=j,rank_B=r,rank_match=rankinfo['match']['selected'],rank_initial=rankinfo['initial']['selected'],B_axis=axis,A_base_regression=areg,B_base_regression=breg,rank_selection=rankinfo,candidate_selection=dict(scores=scores,baseline=baseline,selected=j,validation_gain=baseline-scores[j]),raw_filter=reconstruct_raw_filter(short,rec,j))
 fq=features_for(q,model);heads={}
 for name,f in fq.items():heads[name]=fit_head(f,q['Wfuture'] if name.startswith('W1_') else q['future'])
 # Strong full-history baseline gets the same union of400 trajectories as the recursive encoders.
 union=np.concatenate([x0,xa]);uc,up,uy=base_arrays(union,FULL);heads['full_X']=fit_head(np.concatenate([uc,up],-1),uy)
 # Independent fixed random features, not shuffled future observations.
 rng=np.random.default_rng(job['seed']+400000);random=rng.normal(size=(*q['current'].shape[:2],1));heads['random']=fit_head(np.concatenate([fq['initial'],random],-1),q['future']);model['heads']=heads
 # Disclose the exact linear re-expression in raw X; this is algebra, NOT independent-path evidence.
 err=float(np.max(abs(apply_raw_filter(xv,model['raw_filter'])-v['A'][:,:,j])));assert err<1e-9;model['raw_X_reexpression_max_error_selection']=err
 save(d/'model.json',model);save(d/'LOCAL_MODEL_FREEZE.json',dict(time=time.time(),model_sha256=sha(d/'model.json'),no_confirmation_or_test_loaded=True));print('FITTED',case,'initial_rank',model['rank_initial'],'independent_rank',r,'candidate',j,flush=True)

def main():
 jobs=json.loads((ROOT/'public_jobs.json').read_text())
 with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool:
  for task in as_completed([pool.submit(fit_case,j) for j in jobs]):task.result()
 files={}
 for j in json.loads((ROOT/'public_jobs.json').read_text()):
  for f in ['model.json','LOCAL_MODEL_FREEZE.json']:p=OUT/j['case']/f;files[str(p.relative_to(ROOT))]=sha(p)
 save(ROOT/'MODELS_FROZEN.json',dict(time=time.time(),files=files,model_count=15))
if __name__=='__main__':main()

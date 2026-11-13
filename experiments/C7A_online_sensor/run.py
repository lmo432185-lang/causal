import json,hashlib,time
from datetime import datetime,timezone
import numpy as np,pandas as pd
from scipy.stats import binom
from scipy.special import logsumexp
from config import *
from generate import draw,normal,equivalent_pair
from mechanisms import library
from training_inference import evidence
from online import stream,alarm,commit
from source_A import fit_normal
from legacy_stream import reference_scores

def log(s):print(datetime.now(timezone.utc).isoformat(),s,flush=True)
def dump(p,a):p.write_text(json.dumps(a,indent=2))
def tolerance(score):
 maxima=np.minimum(score[:,1:],score[:,:-1]).max(1);index=int(binom.ppf(1-.05/21,len(maxima),.95));return float(np.sort(maxima)[min(index,len(maxima)-1)])
def allscores(x,eta,bank,ss,classify=False,U=None,Y=None):
 bg=ss['bg'];amp=ss['amp_prior'];vv=ss['sensor_variance'];out={}
 for arm,sensor,V in [('X','u',None),('X_U','u',eta if U is None else U+eta),('X_Y','y',eta if Y is None else Y+eta)]:out[arm]=stream(x,bank,bg,amp,V,vv,sensor,classify)
 return out

def prepare():
 dump(ROOT/'PROTOCOL.json',PROTOCOL);settings=[]
 for ni,sigma in enumerate(SIGMAS):
  bg=fit_normal(normal(N_NORMAL,sigma,SEEDS['normal_train'][ni]));bank=library(bg,steps=T+1);eq={'A':bank['A_first'],'E':dict(bank['A_first'],y=bank['A_first']['u'])};tr=draw(N_TRAIN_PER_CLASS,sigma,1,SEEDS['unlabelled_train'][ni],train=True);pooled=np.concatenate([tr[m]['x'] for m in LABELS]);np.random.default_rng(306000+ni).shuffle(pooled)
  vals=[float(logsumexp(evidence(pooled,bank,bg,a),axis=1).sum()) for a in AMP_PRIOR_CANDIDATES];amp=AMP_PRIOR_CANDIDATES[int(np.argmax(vals))];rng=np.random.default_rng(SEEDS['sensor_train'][ni]);vv=float(np.mean((VSTD*rng.normal(size=(N_SENSOR_NORMAL,T)))**2));ss=dict(bg=bg,amp_prior=amp,training_evidence=vals,sensor_variance=vv)
  for split,nn in [('calibration',N_CAL),('null',N_NULL)]:
   x=normal(nn,sigma,SEEDS[split][ni]);eta=VSTD*np.random.default_rng(SEEDS[split][ni]+100).normal(size=(nn,T));lists={a:[] for a in ['X','X_U','X_Y','EQ_X','EQ_X_U','EQ_X_Y','failure_reference']}
   for start in range(0,nn,200):
    out=allscores(x[start:start+200],eta[start:start+200],bank,ss)
    for arm,(sc,_) in out.items():lists[arm].append(sc)
    for arm,(sc,_) in allscores(x[start:start+200],eta[start:start+200],eq,ss).items():lists['EQ_'+arm].append(sc)
    lists['failure_reference'].append(reference_scores(x[start:start+200],bg)['M4_E'])
   allsc={a:np.concatenate(v) for a,v in lists.items()}
   if split=='calibration':ss['thresholds']={a:tolerance(sc) for a,sc in allsc.items()}
   else:ss['null_audit']={a:dict(n=nn,count=int(np.isfinite(alarm(sc,ss['thresholds'][a])).sum()),fpr=float(np.isfinite(alarm(sc,ss['thresholds'][a])).mean())) for a,sc in allsc.items()}
   np.savez_compressed(OUT/f'{split}_scores_{ni}.npz',**allsc);log(f'{split} complete noise={sigma}')
  settings.append(ss);dump(OUT/'trained_settings.json',settings)
 return settings

rows=[];curves=[];postsave={}
def evaluate(d,bank,ss,truth,ni,co,k,family):
 out=allscores(d['x'],VSTD*d['eta'],bank,ss,True,d['U'],d['Y']);ft=alarm(reference_scores(d['x'],ss['bg'])['M4_E'],ss['thresholds']['failure_reference']);times=np.arange(1,T//BLOCK+1)*CLOCK
 for arm,(sc,pp) in out.items():
  at,it,pred=commit(sc,pp,ss['thresholds'][('EQ_' if family=='passive_equivalent_pair' else '')+arm]);instant=pp.argmax(2);conf=pp.max(2)>=ABSTAIN;correct=(instant==truth)&conf;streak=correct[:,1:]&correct[:,:-1]&(times[None,1:]>=at[:,None]);h=streak.any(1);oracle=np.where(h,(streak.argmax(1)+2)*CLOCK,np.nan)
  valid=np.isfinite(at)&(at>=10);icorrect=(pred==truth)&np.isfinite(it)&(it>=10)&valid;wrong=(pred>=0)&~icorrect
  for i in range(N):
   r=dict(family=family,arm=arm,noise=SIGMAS[ni],cohort=co,kappa=k,trajectory=i,truth=truth,alarm=at[i],id_time=it[i],id_class=int(pred[i]),early_alarm=int(np.isfinite(at[i]) and at[i]<10),detected=int(valid[i]),delay=at[i]-10 if valid[i] else np.nan,correct_commit=int(icorrect[i]),wrong_commit=int(wrong[i]),abstain=int(pred[i]<0),id_delay=it[i]-10 if icorrect[i] else np.nan,oracle_first_correct_diagnostic=oracle[i],failure_reference=ft[i])
   for horizon in [1.,2.,5.]:r[f'detect_by_{horizon}']=int(valid[i] and at[i]<=10+horizon);r[f'correct_id_by_{horizon}']=int(icorrect[i] and it[i]<=10+horizon)
   for margin in [.5,1.,2.]:
    r[f'event_cover_{margin}']=int(valid[i] and np.isfinite(ft[i]) and ft[i]>=10 and ft[i]-at[i]>=margin);r[f'id_cover_{margin}']=int(icorrect[i] and np.isfinite(ft[i]) and ft[i]>=10 and ft[i]-it[i]>=margin)
   rows.append(r)
  for clock in [10.,10.5,11.,12.,13.,15.,20.,25.]:
   u=int(clock/CLOCK)-1;committed=(it<=clock)&np.isfinite(it);good=icorrect&committed;curves.append(dict(family=family,arm=arm,noise=SIGMAS[ni],cohort=co,kappa=k,truth=truth,time=clock,instant_accuracy=float((instant[:,u]==truth).mean()),instant_reject=float((~conf[:,u]).mean()),correct_first_commit_coverage=float(good.mean()),commit_coverage=float(committed.mean()),event_coverage=float((valid&(at<=clock)).mean())))
  key=f'{family}_n{ni}_c{co}_k{k}_truth{truth}_{arm}';postsave[key+'_scores']=sc;postsave[key+'_posterior']=pp.astype(np.float32)

def test(settings):
 names=['config.py','generate.py','mechanisms.py','online.py','run.py','training_inference.py','PROTOCOL.json','results/trained_settings.json'];dump(OUT/'TEST_START.json',dict(time=datetime.now(timezone.utc).isoformat(),sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in names}))
 for ni,sigma in enumerate(SIGMAS):
  ss=settings[ni];bank=library(ss['bg'],steps=T+1);eq={'A':bank['A_first'],'E':dict(bank['A_first'],y=bank['A_first']['u'])}
  for co in COHORTS:
   for k in KTEST:
    dd=draw(N,sigma,k,SEEDS['passive'][co][ni])
    for truth,m in enumerate(LABELS):evaluate(dd[m],bank,ss,truth,ni,co,k,'four_mechanisms')
    aa,bb=equivalent_pair(N,sigma,k,SEEDS['equivalent'][co][ni])
    for truth,d in enumerate([aa,bb]):evaluate(d,eq,ss,truth,ni,co,k,'passive_equivalent_pair')
    np.savez_compressed(OUT/f'online_outputs_n{ni}_c{co}_k{k}.npz',**postsave);postsave.clear();log(f'Test complete noise={sigma},cohort={co},kappa={k}')
 pd.DataFrame(rows).to_csv(OUT/'trajectory_metrics.csv.gz',index=False);pd.DataFrame(curves).to_csv(OUT/'time_curves.csv',index=False)

if __name__=='__main__':
 st=time.time();ss=prepare() if not (OUT/'trained_settings.json').exists() else json.loads((OUT/'trained_settings.json').read_text());test(ss);log(f'ALL COMPLETE {time.time()-st:.1f}s')

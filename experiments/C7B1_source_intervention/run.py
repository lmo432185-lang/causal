import json,hashlib,time
from datetime import datetime,timezone
import numpy as np,pandas as pd
from scipy.stats import binom
from config import *
from physics import *
from inference import stream,commit,anomaly_alarm

def log(s):print(datetime.now(timezone.utc).isoformat(),s,flush=True)
def dump(p,x):p.write_text(json.dumps(x,indent=2))
def fit_normal(x):
 xx=x[:,:-1].ravel();target=x[:,1:].ravel();coef=np.linalg.lstsq(np.c_[xx,np.ones(len(xx))],target,rcond=None)[0];v=np.mean((target-coef[0]*xx-coef[1])**2);return dict(q=float(coef[0]),c=float(coef[1]),var=float(v))
def threshold(score):
 v=np.minimum(score[:,1:],score[:,:-1]).max(1);idx=int(binom.ppf(1-.05/6,len(v),.95));return float(np.sort(v)[idx])

def nominal_batch(n,sigma,seed):
 rng=np.random.default_rng(seed+19);protocol=rng.integers(0,4,n);truth=rng.integers(0,2,n);tau=rng.choice(KTEST,n);cmd=make_commands(n,seed);names=INPUTS[1:]+['blank'];I=np.stack([cmd[names[p]][i] for i,p in enumerate(protocol)]);d=simulate(I,sigma,tau,truth,seed);return d,protocol

def prepare():
 dump(ROOT/'PROTOCOL.json',PROTOCOL);settings=[]
 for ni,sigma in enumerate(SIGMAS):
  bg=fit_normal(healthy(N_NORMAL,sigma,SEEDS['healthy'][ni]));rng=np.random.default_rng(SEEDS['sensor'][ni]);vv=float(np.mean((VSTD*rng.normal(size=(N_SENSOR,T)))**2));ss=dict(bg=bg,sensor_variance=vv)
  for split,nn in [('calibration',N_CAL),('null',N_NULL)]:
   pieces={arm:[] for arm in ARMS};ids=[];blank_commits={arm:0 for arm in ARMS}
   for offset in range(0,nn,100):
    d,pr=nominal_batch(100,sigma,SEEDS[split][ni]+10000*offset);ids.extend(pr)
    for arm in ARMS:
     p,z=stream(d['x'],d['command'],bg,None if arm=='X' else d['U']+VSTD*d['eta'],None if arm=='X' else vv);pieces[arm].append(z);mask=pr==3;_,pred=commit(p);blank_commits[arm]+=int(np.sum(pred[mask]>=0))
   scores={a:np.concatenate(v) for a,v in pieces.items()};np.savez_compressed(OUT/f'{split}_scores_{ni}.npz',**scores,protocol=np.array(ids))
   if split=='calibration':ss['thresholds']={a:threshold(z) for a,z in scores.items()}
   else:
    ss['nominal_audit']={}
    for a,z in scores.items():
     flag=np.isfinite(anomaly_alarm(z,ss['thresholds'][a]));ss['nominal_audit'][a]=dict(n=nn,count=int(flag.sum()),fpr=float(flag.mean()),by_input={names:float(flag[np.array(ids)==k].mean()) for k,names in enumerate(INPUTS[1:]+['blank'])},blank_false_identity_commits=blank_commits[a])
   log(f'{split} complete noise={sigma}')
  settings.append(ss);dump(OUT/'trained_settings.json',settings)
 return settings

rows=[];curve=[]
def test(settings):
 files=['config.py','physics.py','inference.py','run.py','PROTOCOL.json','results/trained_settings.json'];dump(OUT/'TEST_START.json',dict(time=datetime.now(timezone.utc).isoformat(),sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}))
 for ni,sigma in enumerate(SIGMAS):
  ss=settings[ni]
  for co in COHORTS:
   seed=SEEDS['test'][co][ni];cmds=make_commands(N,seed)
   for k in KTEST:
    saved={};example={}
    for key in INPUTS:
     I=cmds['blank'] if key=='I0_passive' else cmds[key];dose=np.cumsum(I*I,axis=1)*DT
     for truth in [0,1]:
      d=simulate(I,sigma,np.full(N,k),truth,seed,passive=key=='I0_passive');example[f'{key}_truth{truth}_x']=d['x'][:10];example[f'{key}_truth{truth}_U']=d['U'][:10];example[f'{key}_truth{truth}_command']=I[:10]
      for arm in ARMS:
       p,z=stream(d['x'],I,ss['bg'],None if arm=='X' else d['U']+VSTD*d['eta'],None if arm=='X' else ss['sensor_variance'],passive=key=='I0_passive');it,pred=commit(p);correct=(pred==truth)&np.isfinite(it);wrong=(pred>=0)&~correct;hard=p.argmax(2);conf=p.max(2)>=CONF;fault=np.full(N,np.nan) if z is None else anomaly_alarm(z,ss['thresholds'][arm]);times=np.arange(1,T//BLOCK+1)*CLOCK
       for i in range(N):
        dose_now=float(dose[i,min(T-1,int(round(it[i]/DT))-1)]) if np.isfinite(it[i]) else np.nan
        r=dict(input=key,arm=arm,noise=sigma,cohort=co,kappa=k,trajectory=i,truth=truth,first_id_time=it[i],first_id_class=int(pred[i]),correct_first=int(correct[i]),wrong_first=int(wrong[i]),refuse=int(pred[i]<0),delay=it[i]-START if correct[i] else np.nan,final_hard_accuracy=int(hard[i,-1]==truth),energy_total=float(dose[i,-1]),energy_at_commit=dose_now,nominal_false_anomaly=None if z is None else int(np.isfinite(fault[i])))
        for h in [.5,1.,2.,4.,8.,15.]:r[f'correct_by_{h}']=int(correct[i] and it[i]<=START+h)
        rows.append(r)
       for clock in [10.,10.5,11.,12.,14.,18.,25.]:
        u=int(clock/CLOCK)-1;curve.append(dict(input=key,arm=arm,noise=sigma,cohort=co,kappa=k,truth=truth,time=clock,hard_accuracy=float((hard[:,u]==truth).mean()),instant_refusal=float((~conf[:,u]).mean()),correct_first_coverage=float((correct&(it<=clock)).mean()),wrong_first_coverage=float((wrong&(it<=clock)).mean()),committed=float((np.isfinite(it)&(it<=clock)).mean())))
       stem=f'{key}_truth{truth}_{arm}';saved[stem+'_posterior']=p.astype(np.float32)
       if z is not None:saved[stem+'_nominal_score']=z
    np.savez_compressed(OUT/f'online_outputs_n{ni}_c{co}_k{k}.npz',**saved);np.savez_compressed(OUT/f'examples_n{ni}_c{co}_k{k}.npz',**example);log(f'Test noise={sigma},cohort={co},kappa={k} complete')
 pd.DataFrame(rows).to_csv(OUT/'trajectory_metrics.csv.gz',index=False);pd.DataFrame(curve).to_csv(OUT/'time_curves.csv',index=False)

if __name__=='__main__':
 start=time.time();s=prepare() if not (OUT/'trained_settings.json').exists() else json.loads((OUT/'trained_settings.json').read_text());test(s);log(f'ALL COMPLETE {time.time()-start:.1f}s')

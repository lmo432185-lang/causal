import json,hashlib,time
from datetime import datetime,timezone
import numpy as np,pandas as pd
from scipy.stats import binom
from config import *
from physics import *
from inference import stream,commit,anomaly_alarm

def log(s):print(datetime.now(timezone.utc).isoformat(),s,flush=True)
def dump(p,x):p.write_text(json.dumps(x,indent=2))
def digest(files):return {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
def seed_for(base,program,batch):return int(np.random.SeedSequence([base,IDS.index(program),batch,51]).generate_state(1)[0])
def fit_normal(x):
 xx=x[:,:-1].ravel();y=x[:,1:].ravel();co=np.linalg.lstsq(np.c_[xx,np.ones(len(xx))],y,rcond=None)[0];v=np.mean((y-co[0]*xx-co[1])**2);return dict(q=float(co[0]),c=float(co[1]),var=float(v))
def threshold(v):
 index=int(binom.ppf(1-.05/(len(PROGRAMS)*3),len(v),.95));return float(np.sort(v)[index])

def prepare():
 s=[]
 for ni,sigma in enumerate(SIGMAS):
  bg=fit_normal(healthy(N_NORMAL,sigma,SEEDS['healthy'][ni]));thresholds={};audit={};maxima={}
  for pi,program in enumerate(PROGRAMS):
   name=program['id']
   for split,nn in [('calibration',N_CAL),('null',N_NULL)]:
    vs=[]
    for batch in range(nn//100):
     seed=seed_for(SEEDS[split][ni],name,batch);rng=np.random.default_rng(np.random.SeedSequence([seed,19]));truth=rng.integers(0,2,100);tau=rng.choice(KTEST,100);I=command(100,program,seed);d=simulate(I,sigma,tau,truth,seed);_,z=stream(d['x'],I,bg);vs.append(np.minimum(z[:,1:],z[:,:-1]).max(1))
    v=np.concatenate(vs);maxima[f'{split}_{name}']=v
    if split=='calibration':thresholds[name]=threshold(v)
    else:audit[name]=dict(n=nn,count=int(np.sum(v>thresholds[name])),fpr=float(np.mean(v>thresholds[name])))
   if (pi+1)%6==0:log(f'Nominal calibration/audit noise={sigma} programs={pi+1}/{len(PROGRAMS)}')
  s.append(dict(bg=bg,thresholds=thresholds,nominal_audit=audit));dump(OUT/'trained_settings.json',s);np.savez_compressed(OUT/f'nominal_maxima_n{ni}.npz',**maxima);log(f'Nominal complete noise={sigma}')
 return s

def evaluate(program,ni,co,k,truth,n,seed,ss,split,save=False):
 I=command(n,program,seed);d=simulate(I,SIGMAS[ni],np.full(n,k),truth,seed);p,z=stream(d['x'],I,ss['bg']);it,pred=commit(p);good=(pred==truth)&np.isfinite(it);wrong=(pred>=0)&~good;refuse=pred<0;flag=np.isfinite(anomaly_alarm(z,ss['thresholds'][program['id']]));dose=np.cumsum(I*I,axis=1)*DT;peak=I.max(1);records=[]
 for i in range(n):
  delay=float(it[i]-START) if good[i] else np.nan;tcap=delay if good[i] else 15.;E=float(dose[i,-1]);loss=.15*E/4+.20*tcap/15+3*wrong[i]+refuse[i]
  r=dict(split=split,program=program['id'],primary=program['primary'],noise=SIGMAS[ni],cohort=co,kappa=k,trajectory=i,truth=truth,first_time=it[i],first_class=int(pred[i]),correct=int(good[i]),wrong=int(wrong[i]),refuse=int(refuse[i]),delay=delay,tcap=tcap,energy=E,peak=float(peak[i]),energy_at_commit=float(dose[i,min(T-1,int(round(it[i]/DT))-1)]) if np.isfinite(it[i]) else np.nan,J=float(loss),final_hard_correct=int(p[i,-1].argmax()==truth),nominal_false_anomaly=int(flag[i]))
  for h in [.5,1.,2.,4.,8.,15.]:r[f'cover_{h}']=int(good[i] and it[i]<=START+h)
  records.append(r)
 if save:
  tag=f'{split}_{program["id"]}_n{ni}_c{co}_k{k}_truth{truth}';np.savez_compressed(OUT/(tag+'.npz'),posterior=p.astype(np.float32),nominal_score=z,command=I[:10],example_x=d['x'][:10])
 return records

def summary(d):
 return d.groupby('program').agg(J=('J','mean'),energy=('energy','mean'),peak=('peak','max'),tcap=('tcap','mean'),wrong=('wrong','mean'),refuse=('refuse','mean'),cover=('cover_4.0','mean')).reset_index()
def pareto(tab):
 cols=['energy','peak','tcap','wrong','refuse'];vals=tab[cols].to_numpy();return [tab.iloc[i].program for i in range(len(tab)) if not np.any(np.all(vals<=vals[i]+1e-12,axis=1)&np.any(vals<vals[i]-1e-12,axis=1))]

def select(settings):
 records=[]
 for pi,program in enumerate(PROGRAMS):
  for ni in range(3):
   seed=SEEDS['validation'][ni]
   for k in KTEST:
    for truth in [0,1]:records.extend(evaluate(program,ni,0,k,truth,NVAL,seed,settings[ni],'validation'))
  if (pi+1)%6==0:log(f'Validation programs={pi+1}/{len(PROGRAMS)}')
 d=pd.DataFrame(records);d.to_csv(OUT/'validation_metrics.csv.gz',index=False);tab=summary(d);tab.to_csv(OUT/'validation_summary.csv',index=False);eligible=tab[tab.program.isin(PRIMARY)&(tab.wrong<=.02)];baseline=float(tab.set_index('program').loc[REFERENCE,'cover']);retain=eligible[eligible.cover>=.8*baseline];high=summary(d[np.isclose(d.noise,.7)]);robust=high[high.program.isin(PRIMARY)&(high.wrong<=.02)]
 picks=dict(bestJ=None if eligible.empty else eligible.sort_values(['J','program']).iloc[0].program,fastest=None if retain.empty else retain.sort_values(['tcap','J','program']).iloc[0].program,lowest_energy=None if retain.empty else retain.sort_values(['energy','J','program']).iloc[0].program,high_noise=None if robust.empty else robust.sort_values(['cover','J','program'],ascending=[False,True,True]).iloc[0].program)
 result=dict(time=datetime.now(timezone.utc).isoformat(),reference=REFERENCE,policies=picks,validation_reference_coverage=baseline,coverage_retention_fraction=.8,pareto=pareto(tab[tab.program.isin(PRIMARY)]),protocol_hash=hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest());dump(OUT/'SELECTION.json',result);log(f'FROZEN POLICIES {picks}');return result

def test(settings,selection):
 files=['config.py','physics.py','inference.py','run.py','PROTOCOL.json','PROTOCOL_REGISTERED.json','results/trained_settings.json','results/SELECTION.json'];dump(OUT/'TEST_START.json',dict(time=datetime.now(timezone.utc).isoformat(),sha256=digest(files)))
 save_ids={REFERENCE,selection['policies']['bestJ']};records=[]
 for pi,program in enumerate(PROGRAMS):
  for ni in range(3):
   for co in COHORTS:
    seed=SEEDS['test'][co][ni]
    for k in KTEST:
     for truth in [0,1]:records.extend(evaluate(program,ni,co,k,truth,N,seed,settings[ni],'test',save=program['id'] in save_ids))
  if (pi+1)%3==0:log(f'Test programs={pi+1}/{len(PROGRAMS)}')
 pd.DataFrame(records).to_csv(OUT/'test_metrics.csv.gz',index=False)
 stress=[];roles=set(selection['policies'].values())|{REFERENCE};roles.discard(None)
 for name in sorted(roles):
  program=PROGRAMS[IDS.index(name)]
  for ni in range(3):
   for co in COHORTS:
    for k in STRESS:
     for truth in [0,1]:stress.extend(evaluate(program,ni,co,k,truth,N,SEEDS['stress'][co][ni],settings[ni],'stress'))
 pd.DataFrame(stress).to_csv(OUT/'stress_metrics.csv.gz',index=False);log('Stress complete')

if __name__=='__main__':
 begin=time.time();ss=prepare();selection=select(ss);test(ss,selection);log(f'ALL COMPLETE {time.time()-begin:.1f}s')

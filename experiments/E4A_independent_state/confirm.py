"""Two independent confirmation blocks; no final tests or true states read."""
import json,time,hashlib,numpy as np,pandas as pd
from scipy.stats import ttest_1samp
from concurrent.futures import ProcessPoolExecutor,as_completed
from settings import *
from producer import generate
from model import unpack,two_roots,acov
from predict import component,targets,logpdf

def holm(p):
 p=np.nan_to_num(np.asarray(p),nan=1.);ix=np.argsort(p);out=np.empty_like(p);out[ix]=np.minimum(1,np.maximum.accumulate(p[ix]*(len(p)-np.arange(len(p)))));return out

def prediction_parts(x,m):
 c=component(x,m);return c,-logpdf(c['mean'],c['cov'],targets(x)).mean(1)/H

def functional(x,b):
 parent=b['parent'];m=b['recursive']['models']['factor2'];w=b['witness']['models']['factor2']
 if m is None:return None
 rec=component(x,m);wit=component(x,w);base=component(x,b['direct']['models'][parent]);o=np.array(origins(x.shape[1]));B=np.c_[np.ones(len(x)*len(o)),x[:,o].ravel(),base['mean'][:,:,9].ravel()];a=(rec['mean'][:,:,9]-base['mean'][:,:,9]).ravel();d=(wit['mean'][:,:,9]-base['mean'][:,:,9]).ravel();return B,a,d

def work(j):
 full=generate(j['index'],j['seed'],0,70);xa=full[50:60];xb=full[60:70];rows=[];checks=[]
 for length in LENGTHS:
  b=unpack(json.loads((OUT/j['case']/f'model_{length}.json').read_text()));a=xa[:,:length];z=xb[:,:length]
  for line in ['recursive','witness']:
   bank=b[line];candidate=bank['models']['factor2'];parent=b['parent'] if line=='recursive' else min(['factor1','memory2','memory3'],key=lambda k:bank['selection_scores'][k].mean());old=bank['models'][parent]
   for block,x in [('A',a),('B',z)]:
    if candidate is None:gain=-np.inf;p=1.
    else:
     _,s0=prediction_parts(x,old);_,s1=prediction_parts(x,candidate);delta=s0-s1;gain=float(delta.mean());p=float(ttest_1samp(delta,.005,alternative='greater').pvalue)
    rows.append(dict(case=j['case'],seed=j['seed'],length=length,line=line,block=block,parent=parent,gain=gain,p=p))
  m=b['recursive']['models']['factor2'];w=b['witness']['models']['factor2'];match=False;pole_error=np.nan;gap=np.nan;response=np.nan;corr=np.nan;fraction=np.nan
  if m is not None:
   poles=np.diag(m['F']);wp=np.diag(w['F']);pole_error=float(np.max(abs(poles-wp)));gap=float(abs(poles[0]-poles[1]));profiles=poles[:,None]**np.arange(1,21);wprofiles=wp[:,None]**np.arange(1,21);response=float(min(np.corrcoef(profiles[i],wprofiles[i])[0,1] for i in range(2)));fa=functional(a,b);fb=functional(z,b)
   B,u,v=fa;Bt,ut,vt=fb;ur=ut-Bt@np.linalg.lstsq(B,u,rcond=1e-8)[0];vr=vt-Bt@np.linalg.lstsq(B,v,rcond=1e-8)[0];corr=float(abs(np.corrcoef(ur,vr)[0,1]));fraction=float(np.var(ur)/max(np.var(ut),1e-16));match=bool(gap>=.05 and pole_error<=.05 and response>=.95)
  halves=[]
  for xx in [full[:20,:length//2],full[:20,length//2:length]]:
   pp,valid=two_roots(acov(xx,50));halves.append(dict(poles=pp.tolist(),valid=valid))
  checks.append(dict(case=j['case'],seed=j['seed'],length=length,proposal_valid=m is not None,pole_match=match,pole_error=pole_error,pole_gap=gap,response_corr=response,conditional_effect_corr=corr,residual_fraction=fraction,early_spectral_valid=halves[0]['valid'],late_spectral_valid=halves[1]['valid'],early_late_pole_error=float(np.max(abs(np.array(halves[0]['poles'])-halves[1]['poles'])))))
 print('CONFIRMED',j['case'],flush=True);return rows,checks

def main():
 for f,h in json.loads((ROOT/'MODELS_FROZEN.json').read_text())['files'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
 rows=[];checks=[]
 with ProcessPoolExecutor(max_workers=3) as pool:
  for f in as_completed([pool.submit(work,j) for j in json.loads((ROOT/'jobs.json').read_text())]):a,b=f.result();rows.extend(a);checks.extend(b)
 df=pd.DataFrame(rows);assert len(df)==3200;df['p_holm']=holm(df.p);df['passed']=(df.gain>=.005)&(df.p_holm<.05);df.to_csv(OUT/'confirmation_tests.csv',index=False);c=pd.DataFrame(checks);support=df.groupby(['case','length']).passed.all().rename('four_tests_passed').reset_index();c=c.merge(support,on=['case','length']);c['novel_predictive_support']=c.four_tests_passed&c.pole_match;c.to_csv(OUT/'confirmation_checks.csv',index=False);files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'confirmation_tests.csv',OUT/'confirmation_checks.csv']};(ROOT/'DECISIONS_FROZEN.json').write_text(json.dumps(dict(time=time.time(),files=files),indent=2));print('CONFIRMATION FROZEN',flush=True)
if __name__=='__main__':main()

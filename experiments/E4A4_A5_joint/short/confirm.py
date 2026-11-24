import json,gzip,time,hashlib,numpy as np,pandas as pd
from scipy.stats import ttest_1samp
from scipy.linalg import solve_discrete_lyapunov
from concurrent.futures import ProcessPoolExecutor,as_completed
from settings import *
from model import unpack
from producer import generate
from predict import component,targets,logpdf

def holm(p):
 p=np.nan_to_num(np.asarray(p),nan=1.);ix=np.argsort(p);out=np.empty_like(p);out[ix]=np.minimum(1,np.maximum.accumulate(p[ix]*(len(p)-np.arange(len(p)))));return out

def density_scores(x,m):
 c=component(x,m);return -logpdf(c['mean'],c['cov'],targets(x)).mean(1)/H

def profile(m):
 F,Q,C,R=m['F'],m['Q'],m['C'],m['R'];d=len(C);A=np.zeros((d+1,d+1));A[:d,:d]=F;A[d,:d]=C@F;A[d,d]=m.get('a',0);V=np.zeros_like(A);V[:d,:d]=Q;V[:d,d]=Q@C;V[d,:d]=C@Q;V[d,d]=C@Q@C+R;P=solve_discrete_lyapunov(A,V);v=P[:,-1];cur=np.eye(d+1);g=[]
 for h in range(20):cur=cur@A;g.append((cur@v)[-1])
 return np.array(g)/max(P[-1,-1],1e-12)

def work(j):
 with gzip.open(OUT/(j['case']+'.json.gz'),'rt') as f:b=unpack(json.load(f))
 x=generate(j['index'],j['seed'],10,20);rows=[];matches=[]
 for route in ['R1','R2','R3']:
  m=[]
  for bank in ['R','D']:
   key=b['candidates'][route][bank]['novel'];m.append(None if key is None else b['models'][key]);base=b['models'][b['candidates']['R0'][bank]['best']]
   for block,xx in [('A',x[:10]),('B',x[10:])]:
    if key is None:gain=-np.inf;p=1.
    else:
     delta=density_scores(xx,base)-density_scores(xx,b['models'][key]);gain=float(delta.mean());p=float(ttest_1samp(delta,.005,alternative='greater').pvalue)
    rows.append(dict(case=j['case'],seed=j['seed'],route=route,bank=bank,block=block,gain=gain,p=p,available=key is not None))
  if any(v is None for v in m):match=False;error=np.nan;corr=np.nan
  else:
   a,z=map(profile,m);error=float(np.linalg.norm(a-z)/max(np.linalg.norm(a),np.linalg.norm(z),1e-12));corr=float(np.corrcoef(a,z)[0,1]);match=bool(error<=.1 and corr>=.95)
  matches.append(dict(case=j['case'],seed=j['seed'],route=route,function_match=match,relative_response_error=error,response_corr=corr))
 print('CONFIRM',j['case'],flush=True);return rows,matches

def main():
 for f,h in json.loads((ROOT/'CANDIDATES_FROZEN.json').read_text())['files'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
 rows=[];matches=[]
 with ProcessPoolExecutor(max_workers=3) as pool:
  for f in as_completed([pool.submit(work,j) for j in json.loads((ROOT/'jobs.json').read_text())]):a,b=f.result();rows.extend(a);matches.extend(b)
 d=pd.DataFrame(rows);assert len(d)==5760;d['p_holm']=holm(d.p);d['passed']=(d.gain>=.005)&(d.p_holm<.05);d.to_csv(OUT/'confirmation.csv',index=False);m=pd.DataFrame(matches);q=d.groupby(['case','route']).passed.all().rename('four_tests_passed').reset_index();m=m.merge(q,on=['case','route']);m['supported']=m.four_tests_passed&m.function_match;m.to_csv(OUT/'confirmation_support.csv',index=False)
 files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'confirmation.csv',OUT/'confirmation_support.csv']};(ROOT/'FINAL_MODELS_FROZEN.json').write_text(json.dumps(dict(time=time.time(),files=files,candidate_manifest_sha256=hashlib.sha256((ROOT/'CANDIDATES_FROZEN.json').read_bytes()).hexdigest(),test_not_opened=True,no_candidate_retuning=True),indent=2));print('CONFIRMATION AND FINAL MODELS FROZEN',flush=True)
if __name__=='__main__':main()

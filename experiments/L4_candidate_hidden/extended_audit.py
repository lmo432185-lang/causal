"""Supplementary fitted-law closure, registered before any hidden answers opened.
Mixed relative delay2 can be absorbed into unknown X process noise in some cases.
No model fit, score, threshold or primary decision is changed.
"""
import json,time,hashlib
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from settings import ROOT,OUT
from ssm import restore,serialize,mirror
from equivalence import kernel

def delayed_counterparts(p):
 if p['kind']!='G3' or p['delay']!=2:return []
 A=p['A'];a,c,b=A[0,0],A[1,1],A[2,2];alpha=A[1,3];g1,g2=A[2,0],A[2,1]
 if a*c<1e-8:return []
 shift=g1*g2*alpha/(a*c);d0=(1+a*a)*(1+c*c)+2*a*c;d1=-2*(a*(1+c*c)+c*(1+a*a));f0=g1*g1*(1+c*c)+g2*g2*(1+a*a+alpha*alpha);f1=-2*(g1*g1*c+g2*g2*a+c*g1*g2*alpha);new0=f0-shift*d0;L=-(f1-shift*d1)/2;qx=p['Q'][2,2]+shift;out=[]
 if not(1e-5<=qx<=5):return []
 for source,middle in [(a,c),(c,a)]:
  if source<1e-8:continue
  gain2=L/source
  if gain2<=0:continue
  alpha2=new0/gain2-1-source*source
  if alpha2<0:continue
  gain=np.sqrt(gain2);aa=np.sqrt(alpha2)
  if gain>3 or aa>3:continue
  F=np.array([[source,0,0],[aa,middle,0],[0,gain,b]]);Q=np.diag([1.,1.,qx]);q={**p,'kind':'G2','delay':1,'A':F,'Q':Q,'P0':solve_discrete_lyapunov(F,Q)};err=float(np.max(abs(kernel(q)-kernel(p)))/max(np.max(abs(kernel(p))),1e-12))
  if err<1e-8:out.extend([dict(label='G2',error=err,parameters=serialize(q)),dict(label='G4',error=err,parameters=serialize(mirror(q)))]);break
 return out

if __name__=='__main__':
 while not (ROOT/'ALL_X_ONLY_FROZEN.json').exists():time.sleep(5)
 gate=json.loads((ROOT/'ALL_X_ONLY_FROZEN.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert all(sha(ROOT/f)==h for f,h in gate['files'].items());jobs=json.loads((ROOT/'public_jobs.json').read_text());manifest={}
 for job in jobs:
  d=OUT/job['case'];p=json.loads((d/'selected_parameters.json').read_text())['models'];a={k:delayed_counterparts(restore(v)) for k,v in p.items() if k!='single'};dec=json.loads((d/'X_only_decisions.json').read_text());guard=[]
  for i,k in enumerate(dec['prediction']):guard.append(bool(dec['guarded_declared'][i] and not a[k]))
  f=d/'supplementary_law_audit.json';f.write_text(json.dumps(dict(case=job['case'],additional_alternatives=a,supplementary_guarded_declared=guard,primary_decisions_unchanged=True),indent=2));manifest[str(f.relative_to(ROOT))]=sha(f)
 (ROOT/'ALL_SUPPLEMENTARY_AUDIT_FROZEN.json').write_text(json.dumps(dict(time=time.time(),files=manifest,truth_not_opened=True),indent=2));print('SUPPLEMENTARY AUDIT FROZEN before truth evaluation',flush=True)

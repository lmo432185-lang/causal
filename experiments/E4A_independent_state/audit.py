"""Read-only integrity, online causality, observational equivalence and replay checks."""
import json,time,hashlib,numpy as np,pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from settings import *
from producer import generate,matrices,SYSTEMS,arma_alias
from model import unpack
from predict import component,infer,scores
import native

def autov(m,n=100):
 A=np.eye(len(m['C']));g=[]
 for k in range(n):g.append(m['C']@A@m['P']@m['C']+(m['R'] if k==0 else 0));A=A@m['F']
 return np.array(g)*m.get('scale',1.)**2

def main():
 audits={};dates={}
 for label in ['PROTOCOL_REGISTERED','CODE_REGISTERED','ANALYSIS_REGISTERED','POSTCHECK_REGISTERED','MODELS_FROZEN','DECISIONS_FROZEN','TEST_A_FROZEN','TEST_B_FROZEN']:
  a=json.loads((ROOT/(label+'.json')).read_text());dates[label]=a['time'];assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in a['files'].items());audits[label]=len(a['files'])
 assert dates['PROTOCOL_REGISTERED']<dates['CODE_REGISTERED']<dates['MODELS_FROZEN']<dates['DECISIONS_FROZEN']<dates['TEST_A_FROZEN']<dates['TEST_B_FROZEN'];assert dates['ANALYSIS_REGISTERED']<dates['DECISIONS_FROZEN'];assert dates['POSTCHECK_REGISTERED']<dates['TEST_A_FROZEN']
 checks=[];equiv=[];replays=[];stability=[]
 tables={batch:pd.read_csv(OUT/f'test_{batch}.csv.gz') for batch in ['A','B']}
 for batch,d in tables.items():assert len(d)==80000 and not d.duplicated(['case','length','group','trajectory']).any()
 for i,meta in enumerate(SYSTEMS):
  case=meta['id']+'_4011';b=unpack(json.loads((OUT/case/'model_5000.json').read_text()));x=generate(i,4011,1,NTEST)[:1];cache={}
  for g in ['M1','M2','M3','M4']:
   r=infer(x,b,g,cache);ss=scores(x,r);stored=tables['A'].query('case==@case and length==5000 and trajectory==0 and group==@g').iloc[0];error=max(abs(ss[k][0]-stored[k]) for k in ['NLL','CRPS','MSE','cover90','width90']);assert error<1e-8;replays.append(dict(case=case,group=g,max_error=error));assert np.max(abs(r['weights'].sum(2)-1))<1e-10
  m=b['direct']['models']['factor2'];c=component(x,m);o=np.array(origins(x.shape[1]));v=c['cov'][:,0,0];l=-.5*(np.log(2*np.pi*v)+(x[:,o+1]-c['mean'][:,:,0])**2/v);err=np.max(abs(l-c['post']['ll'][:,o+1]));assert err<1e-7
  xp=x.copy();xp[:,1300:]+=17.;r0=infer(x,b,'M4',{});r1=infer(xp,b,'M4',{});early=np.array(origins(5000))<1300;future_error=max(np.max(abs(r0['mean'][:,early]-r1['mean'][:,early])),np.max(abs(r0['weights'][:,:1300]-r1['weights'][:,:1300])));assert future_error<1e-10;checks.append(dict(case=case,issued_density_error=err,future_perturbation_error=future_error))
  a=arma_alias(m);ca=component(x,a);e1=np.max(abs(autov(m)-autov(a)));e2=np.max(abs(c['mean']-ca['mean']));e3=np.max(abs(c['cov']-ca['cov']));e4=np.max(abs(c['post']['ll']-ca['post']['ll']));assert max(e1,e2,e3,e4)<1e-6;equiv.append(dict(case=case,type='fitted_two_AR1_vs_one_ARMA21',autocov_error=e1,mean_error=e2,cov_error=e3,loglik_error=e4,equal_prior_weight=.5))
  if meta['family'] in ['C','E']:
   F,Q,C,R=matrices(meta);p=solve_discrete_lyapunov(F,Q);truth=dict(F=F,Q=Q,C=C,R=R,P=p,a=0.,scale=1.,kind='factor2');original=native.run(x,truth)
   if meta['family']=='E':alt=arma_alias(truth);name='true_two_AR1_vs_one_ARMA21'
   else:
    pp=np.diag(p@C);qq=pp-F@pp@F.T;assert np.linalg.eigvalsh(qq).min()>0;alt=dict(truth,P=pp,Q=qq);name='correlated_vs_independent_same_X'
   post=native.run(x,alt);error=np.max(abs(autov(truth)-autov(alt)));lle=np.max(abs(original['ll']-post['ll']));assert max(error,lle)<1e-7;equiv.append(dict(case=case,type=name,autocov_error=error,loglik_error=lle,latent_cov_difference=float(np.max(abs(original['cov'][200:]-post['cov'][200:]))),equal_prior_weight=.5))
 for p in OUT.glob('*/model_*.json'):
  b=unpack(json.loads(p.read_text()))
  for line in ['direct','recursive','witness']:
   for kind,m in b[line]['models'].items():
    if m is None:continue
    radius=max(abs(np.linalg.eigvals(m['F']))) if len(m['F']) else abs(m['a']);mineig=np.linalg.eigvalsh(m['P']).min() if len(m['P']) else 0;assert radius<1 and mineig>=-1e-8;stability.append(dict(case=p.parent.name,length=b.get('length',int(p.stem.split('_')[-1])),line=line,kind=kind,radius=radius,minimum_P_eigenvalue=mineig,iterations=m.get('iterations',m.get('n_iter',np.nan))))
 pd.DataFrame(checks).to_csv(OUT/'online_audit.csv',index=False);pd.DataFrame(equiv).to_csv(OUT/'equivalence_audit.csv',index=False);pd.DataFrame(replays).to_csv(OUT/'replay_audit.csv',index=False);pd.DataFrame(stability).to_csv(OUT/'stability_audit.csv',index=False);(ROOT/'AUDIT_COMPLETE.json').write_text(json.dumps(dict(time=time.time(),all_passed=True,frozen_file_counts=audits,freeze_times=dates,replayed_predictions=80,final_test_rows_per_batch=80000,interpretation='equivalence does not identify physical factor count'),indent=2));print('ALL AUDITS PASSED',flush=True)
if __name__=='__main__':main()

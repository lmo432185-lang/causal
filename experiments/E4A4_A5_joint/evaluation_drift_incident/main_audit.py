import json,gzip,time,hashlib,numpy as np,pandas as pd
from scipy.special import logsumexp
from scipy.linalg import solve_discrete_lyapunov
from settings import *
from producer import generate,digest,SYSTEMS,matrices,arma_alias
from model import unpack
from predict import component,scores,logpdf,targets,history_scores
from inference import mix
from probability import correlated_alias

def autov(m,n=60):
 F,Q,C,R=m['F'],m['Q'],m['C'],m['R'];d=len(C)
 if m.get('a',0)!=0:
  A=np.zeros((d+1,d+1));A[:d,:d]=F;A[d,:d]=C@F;A[d,d]=m['a'];V=np.zeros_like(A);V[:d,:d]=Q;V[:d,d]=Q@C;V[d,:d]=C@Q;V[d,d]=C@Q@C+R;P=solve_discrete_lyapunov(A,V);C=np.r_[np.zeros(d),1.];F=A
 else:P=m['P']
 A=np.eye(len(C));ans=[]
 for k in range(n):ans.append(C@A@P@C+(R if k==0 and m.get('a',0)==0 else 0));A=A@F
 return np.array(ans)*m['scale']**2

def main():
 dates={};counts={}
 labels=['PROTOCOL_REGISTERED','CODE_REGISTERED','PREDICTION_REGISTERED','POSTTEST_REGISTERED','BAYES_REGISTERED','CANDIDATES_FROZEN','CALIBRATION_SETUP_FROZEN','BAYES_FROZEN','EXTRA_PLANS_FROZEN','FINAL_MODELS_FROZEN','TEST_A_FROZEN','TEST_B_FROZEN']
 for label in labels:
  rec=json.loads((ROOT/(label+'.json')).read_text());dates[label]=rec['time'];counts[label]=len(rec.get('files',{}))
  for f,h in rec.get('files',{}).items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,(label,f)
 assert dates['PROTOCOL_REGISTERED']<dates['CODE_REGISTERED']<dates['CANDIDATES_FROZEN']<dates['FINAL_MODELS_FROZEN']<dates['TEST_A_FROZEN']<dates['TEST_B_FROZEN'];assert dates['PREDICTION_REGISTERED']<dates['FINAL_MODELS_FROZEN'];assert dates['POSTTEST_REGISTERED']<dates['TEST_A_FROZEN'];assert dates['BAYES_REGISTERED']<dates['BAYES_FROZEN']<dates['FINAL_MODELS_FROZEN']
 oldmanifest=json.loads((OLD/'MODELS_FROZEN.json').read_text())['files'];old_verified=0
 for file in OUT.glob('*.json.gz'):
  with gzip.open(file,'rt') as ff:bb=unpack(json.load(ff))
  if bb['index']<20:
   expected=oldmanifest['results/'+bb['case']+'/model_5000.json'];assert bb['old_model_sha256']==expected;old_verified+=1
 assert old_verified==400
 tests={ba:pd.read_csv(OUT/f'test_{ba}.csv.gz') for ba in ['A','B']}
 for d in tests.values():assert len(d)==480*20*21 and not d.duplicated(['case','method','trajectory']).any()
 eq=[];rotation=[];online=[];replay=[];priors=json.loads((OUT/'extra_plans.json').read_text())
 for index,meta in enumerate(SYSTEMS):
  case=meta['id']+'_4011'
  with gzip.open(OUT/(case+'.json.gz'),'rt') as f:b=unpack(json.load(f))
  b['plans'].update(priors[case]);full=generate(index,4011,11,NTEST);assert digest(full)==json.loads((ROOT/'TEST_A_FROZEN.json').read_text())['input_hashes'][case];x=full[:1];cache={}
  for method in ['R0','R1','R2','R3','A','B','C','D','E','J0','J1_param','J2_point','StrongestSingle']:
   plan=b['plans'][method]
   if plan.get('type')=='ARhistory':
    sc=history_scores(x,plan['parameters']);stored=tests['A'].query('case==@case and method==@method and trajectory==0').iloc[0];error=max(abs(float(sc[k][0])-stored[k]) for k in ['NLL','CRPS','MSE','cover90','width90']);assert all(np.isclose(float(sc[k][0]),stored[k],rtol=1e-10,atol=1e-8) for k in ['NLL','CRPS','MSE','cover90','width90']);replay.append(dict(condition=meta['id'],method=method,error=error));continue
   if not plan['ids']:continue
   r,_=mix(x,b['models'],plan['ids'],plan.get('prior'),cache);sc=scores(x,r);stored=tests['A'].query('case==@case and method==@method and trajectory==0').iloc[0];error=max(abs(float(sc[k][0])-stored[k]) for k in ['NLL','CRPS','MSE','cover90','width90']);assert all(np.isclose(float(sc[k][0]),stored[k],rtol=1e-10,atol=1e-8) for k in ['NLL','CRPS','MSE','cover90','width90']);replay.append(dict(condition=meta['id'],method=method,error=error));assert np.max(abs(r['weights'].sum(2)-1))<1e-10
  plan=b['plans']['E'];r,_=mix(x,b['models'],plan['ids'],plan['prior'],{});xp=x.copy();xp[:,1300:]+=11.;rp,_=mix(xp,b['models'],plan['ids'],plan['prior'],{});ix=np.array(origins(5000))<1300;err=max(np.max(abs(r['mean'][:,ix]-rp['mean'][:,ix])),np.max(abs(r['weights'][:,:1300]-rp['weights'][:,:1300])));assert err<1e-9
  yy=targets(x);ll=np.stack([logpdf(p['mean'][:,:,:1],p['cov'][:,:1,:1],yy[:,:,:1]) for p in r['parts']],-1);issued=logsumexp(np.log(np.maximum(r['origin_weights'],1e-300))+ll,2);nextll=np.stack([p['post']['ll'][:,np.array(origins(5000))+1] for p in r['parts']],-1);expected=logsumexp(np.log(np.maximum(r['origin_weights'],1e-300))+nextll,2);densityerr=np.max(abs(issued-expected));assert densityerr<1e-7;online.append(dict(condition=meta['id'],future_error=err,issued_density_error=densityerr))
  m=b['bases']['R']['canonical_point'];a=arma_alias(m);c=component(x,m);ca=component(x,a);error=max(np.max(abs(c['mean']-ca['mean'])),np.max(abs(c['cov']-ca['cov'])),np.max(abs(c['post']['ll']-ca['post']['ll'])),np.max(abs(autov(m)-autov(a))));assert error<1e-6
  ar=b['models']['R_old_memory2'];v=generate(index,4011,0,70)[40:50];alts=[m,a,ar];prior=np.array([component(v,z)['post']['ll'].sum() for z in alts]);prior-=logsumexp(prior);ll=np.stack([component(x,z)['post']['ll'][0] for z in alts],1);lw=np.cumsum(ll,0)+prior;odds=lw[:,0]-lw[:,1];pair=np.exp(lw[:,:2]-logsumexp(lw[:,:2],1)[:,None]);assert max(abs(odds))<1e-5;eq.append(dict(condition=meta['id'],kind='two_AR1_one_ARMA',maximum_error=error,pair_probability=float(pair[-1,0]),pair_log_odds=float(odds[-1]),AR2_autocov_difference=float(np.max(abs(autov(m)-autov(ar)))),factor_count='unresolved:1_or_2',storage_dimension=2,predictive_mode_hypothesis=2))
  if meta['family']=='E':
   F,Q,C,R=matrices(meta);truth=dict(F=F,Q=Q,C=C,R=R,P=solve_discrete_lyapunov(F,Q),a=0.,scale=1.,kind='factor2');alias=arma_alias(truth);phi=np.array([np.trace(F),-np.linalg.det(F)]);Fa=np.array([[phi[0],phi[1]],[1.,0.]]);Qa=np.diag([alias['Q'][0,0],0.]);ar2=dict(F=Fa,Q=Qa,C=np.array([1.,0.]),R=R,P=solve_discrete_lyapunov(Fa,Qa),a=0.,scale=1.,kind='memory2');eq.append(dict(condition=meta['id'],kind='true_ARMA_vs_AR2_no_MA',maximum_error=float(np.max(abs(autov(truth)-autov(alias)))),AR2_autocov_difference=float(np.max(abs(autov(truth)-autov(ar2)))),factor_count='AR2_not_generally_equal',storage_dimension=2,predictive_mode_hypothesis=2))
  for rho in [-.95,0,.95]:
   z=correlated_alias(m,rho);cz=component(x,z);error=max(np.max(abs(cz['mean']-c['mean'])),np.max(abs(cz['cov']-c['cov'])),np.max(abs(cz['post']['ll']-c['post']['ll'])));assert error<1e-6;eq.append(dict(condition=meta['id'],kind='correlated_Q_null',rho=rho,maximum_error=error,latent_cov_difference=float(np.max(abs(cz['post']['cov']-c['post']['cov'])))))
  jointkey=b['candidates']['R3']['R']['best'];rotmodels=[('diagonal',m)]+([('joint',b['models'][jointkey])] if jointkey is not None else [])
  for name,model in rotmodels:
   d=len(model['C']);rng=np.random.default_rng(index+998);T=np.linalg.qr(rng.normal(size=(d,d)))[0];z=dict(model,F=T@model['F']@T.T,Q=T@model['Q']@T.T,C=model['C']@T.T,P=T@model['P']@T.T);p0=component(x,model);p1=component(x,z);error=max(np.max(abs(p0['mean']-p1['mean'])),np.max(abs(p0['cov']-p1['cov'])),np.max(abs(p0['post']['ll']-p1['post']['ll'])));assert error<1e-6;rotation.append(dict(condition=meta['id'],model=name,dimension=d,maximum_error=error,original_offdiagonal=int((abs(model['F']-np.diag(np.diag(model['F'])))>1e-8).sum()),rotated_offdiagonal=int((abs(z['F']-np.diag(np.diag(z['F'])))>1e-8).sum()),transition_change=float(np.linalg.norm(model['F']-z['F']))))
 for rows,name in [(eq,'equivalence_audit'),(rotation,'rotation_audit'),(online,'online_audit'),(replay,'replay_audit')]:pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
 (ROOT/'AUDIT_COMPLETE.json').write_text(json.dumps(dict(time=time.time(),all_passed=True,frozen_counts=counts,freeze_times=dates,test_rows_per_batch=480*20*21,replays=len(replay),identity='refuse_unique_physical_factor_count'),indent=2));print('AUDITS PASSED',len(replay),flush=True)
if __name__=='__main__':main()

import numpy as np,json,time
from settings import *
from learn import fit_all
from ssm import *
from predict import infer,score

def main():
 rng=np.random.default_rng(819203);a=rng.normal(size=(20,160));b=rng.normal(size=(20,160));v=rng.normal(size=(10,160));tic=time.time();m=fit_all(a,b,v,20,9);graph={r:{'initial_rank':m['rank_'+r],'allowed':[0,1,2,3]} for r in ['direct','recursive']};cache={};error=0
 for g in GROUPS:
  q=infer(v,m,graph,g,cache);z=score(v,q,128);assert all(np.isfinite(w).all() for w in z.values());assert np.max(abs(q['weights'].sum(2)-1))<1e-10
  changed=v.copy();changed[:,81:]+=25.;qc=infer(changed,m,graph,g);ind=[i for i,t in enumerate(ORIGINS) if t<=80];error=max(error,float(np.max(abs(q['mean'][:,ind]-qc['mean'][:,ind]))));assert np.max(abs(q['weights'][:,:81]-qc['weights'][:,:81]))<1e-10
 c=component(v,m['direct']['models'][2]);ll=logpdf(c['mean'][:,:,:1],c['cov'][:,:1,:1],target(v)[:,:,:1]);llerr=float(np.max(abs(ll-c['post']['ll'][:,np.array(ORIGINS)+1])));assert llerr<1e-9 and error<1e-9
 (ROOT/'PIPELINE_DEVELOPMENT.json').write_text(json.dumps(dict(passed=True,unrelated_noise_only=True,groups_checked=7,max_future_prefix_error=error,max_issued_onestep_vs_next_innovation_density_error=llerr,seconds=time.time()-tic),indent=2));print('PIPELINE DEV PASSED',time.time()-tic)
if __name__=='__main__':main()

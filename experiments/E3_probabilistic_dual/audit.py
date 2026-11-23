import json,time,hashlib,copy
import numpy as np,pandas as pd
from settings import *
from ssm import unpack,component,logpdf,target
from predict import infer,score,weights
from producer import simulate,digest

def main():
 checked=0;times={}
 for name in ['PROTOCOL_REGISTERED','CODE_REGISTERED','ANALYSIS_REGISTERED','POSTCHECK_REGISTERED','MODELS_FROZEN','DECISIONS_FROZEN','TEST_A_FROZEN','TEST_B_FROZEN']:
  reg=json.loads((ROOT/(name+'.json')).read_text());times[name]=reg['time']
  for f,h in reg['files'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,(name,f);checked+=1
 assert times['PROTOCOL_REGISTERED']<times['CODE_REGISTERED']<times['MODELS_FROZEN']<times['DECISIONS_FROZEN']<times['TEST_A_FROZEN']<times['TEST_B_FROZEN'];assert times['ANALYSIS_REGISTERED']<times['DECISIONS_FROZEN'];assert times['POSTCHECK_REGISTERED']<times['TEST_A_FROZEN']
 jobs=json.loads((ROOT/'jobs.json').read_text());tables={b:pd.read_csv(OUT/f'test_{b}.csv.gz') for b in ['A','B']};maxprefix=0.;maxreplay=0.;normerr=0.;cases=0;inputcases=0;maxdensity=0.;nonmonotone=0
 for j in jobs:
  hashes=json.loads((OUT/j['case']/'INPUT_HASHES.json').read_text());x=simulate(j['index'],j['seed'],0,880)
  for label,a in [('train_R',x[:400]),('train_D',x[400:800]),('selection',x[800:840]),('confirmation',x[840:])]:assert digest(a)==hashes[label]
  for batch in ['A','B']:
   raw=simulate(j['index'],j['seed'],1 if batch=='A' else 2,40);reg=json.loads((ROOT/f'TEST_{batch}_FROZEN.json').read_text());assert digest(raw)==reg['input_hashes'][j['case']]
  inputcases+=1
  for n in SIZES:
   m=unpack(json.loads((OUT/j['case']/f'model_{n}.json').read_text()))
   for bank in ['direct','recursive','witness']:
    for z in m[bank]['models']:
     if not len(z['C']):continue
     assert np.min(np.linalg.eigvalsh(z['Q']))>0 and np.min(np.linalg.eigvalsh(z['P']))>0;assert max(abs(np.linalg.eigvals(z['F'])))<=.995001
     nonmonotone+=int(np.any(np.diff(z['EM_loglik'])<-.000001))
  if j['seed']!=3011:continue
  m=unpack(json.loads((OUT/j['case']/'model_100.json').read_text()));graph=json.loads((OUT/j['case']/'graph.json').read_text())['100']
  for batch in ['A','B']:
   raw=simulate(j['index'],j['seed'],1 if batch=='A' else 2,40);cache={};tab=tables[batch];changed=raw[:3].copy();changed[:,81:]+=23.;ind=[i for i,t in enumerate(ORIGINS) if t<=80]
   for group in GROUPS:
    r=infer(raw,m,graph,group,cache);s=score(raw,r,j['seed']+100+500000);expected=tab[(tab.case==j['case'])&(tab.n==100)&(tab.group==group)].sort_values('trajectory')
    for k,v in s.items():maxreplay=max(maxreplay,float(np.max(abs(v-expected[k].to_numpy()))))
    normerr=max(normerr,float(np.max(abs(r['weights'].sum(2)-1))));r0=infer(raw[:3],m,graph,group);r1=infer(changed,m,graph,group);maxprefix=max(maxprefix,float(np.max(abs(r0['mean'][:,ind]-r1['mean'][:,ind]))),float(np.max(abs(r0['weights'][:,:81]-r1['weights'][:,:81]))))
   cases+=1
   c=component(raw,m['direct']['models'][2]);ll=logpdf(c['mean'][:,:,:1],c['cov'][:,:1,:1],target(raw)[:,:,:1]);maxdensity=max(maxdensity,float(np.max(abs(ll-c['post']['ll'][:,np.array(ORIGINS)+1]))))
 # Exact observational-equivalence and duplicate-prediction merge check.
 j=next(x for x in jobs if x['case']=='S2_medium_3011');m=unpack(json.loads((OUT/j['case']/'model_100.json').read_text()))['direct']['models'][2];r=np.array([[1.,1.],[-1.,1.]])/np.sqrt(2);m2=copy.deepcopy(m);m2['F']=r@m['F']@r.T;m2['Q']=r@m['Q']@r.T;m2['P']=r@m['P']@r.T;m2['C']=m['C']@r.T;x=simulate(j['index'],j['seed'],1,40);a=component(x,m);b=component(x,m2);equiv=max(np.max(abs(a['mean']-b['mean'])),np.max(abs(a['cov']-b['cov'])),np.max(abs(a['post']['ll']-b['post']['ll'])));w=weights(np.stack([a['post']['ll'],b['post']['ll']],-1),np.array([.5,.5]),True);we=float(np.max(abs(w-.5)));assert equiv<1e-8 and we<1e-8
 assert maxprefix<1e-9 and maxreplay<1e-8 and normerr<1e-10 and maxdensity<1e-8
 out=dict(passed=True,hashes_checked=checked,input_cases_regenerated=inputcases,prefix_and_replay_case_batches=cases,max_prefix_error=maxprefix,max_score_replay_error=maxreplay,max_weight_normalization_error=normerr,max_issued_density_error=maxdensity,rotation_equivalence_error=float(equiv),equivalent_explanation_weight_error=we,equivalent_structure_probabilities=[.5,.5],unique_physical_structure_rejected=True,duplicate_predictive_components_can_merge_without_density_change=True,nonmonotone_EM_fits_due_to_floors_and_stability_projection=nonmonotone,EM_global_optimum_claimed=False,refitted=False)
 (ROOT/'AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()

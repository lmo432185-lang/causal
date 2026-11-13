import json,numpy as np
from scipy.integrate import quad
from scipy.fft import rfft,irfft,next_fast_len
from config import *
from generate import event
from detectors import score_batch,ramp_integral,predict
from source_C0 import templates
c0=json.loads((OUT/'frozen_C0.json').read_text());c4=json.loads((OUT/'frozen_C4.json').read_text());base=c0['0.35']['normal_AR'];fixed=c0['0.35']['shared_index'];scales=c4['0.35']['scales'];x=event(1,2,1,0)['x'][:3];sc=score_batch(x,base,fixed,scales);a=-np.log(base['q'])/DT
# Exact ramp response integral independently checked at whole/partial-rise intervals.
for R in [.07,.1,.37,2.,10.]:
 for age in [0,.02,max(0,R-.025),R,R+.2]:
  expected=quad(lambda z:np.exp(-a*(DT-z))*min((age+z)/R,1.),0,DT,epsabs=1e-13,points=[R-age] if 0<R-age<DT else None)[0]
  np.testing.assert_allclose(ramp_integral(age,R,base),expected,atol=1e-11,rtol=1e-9)
# Independently evaluate reduced-form likelihood via FFT/direct templates on prefixes.
# This code does not instantiate Y or U and does not reuse the M3 score recursion.
for end in [15,50,125,300,500]:
 residual=-(x[:,1:end+1]-base['q']*x[:,:end]-base['c']);starts=np.arange(10,end,BLOCK);generic=[];collapsed=[];direct=[]
 for kappa in GRID:
  hh=ramp_integral(np.arange(end)*DT,2*kappa,base);he=templates(end,base,[kappa])[0];gg=[];ee=[]
  for start in starts:
   w=hh[:end-start];we=he[:end-start];dot=residual[:,start:]@w;de=residual[:,start:]@we;gg.append(np.maximum(dot,0)**2/(base['var']*(w@w)));ee.append(np.maximum(de,0)**2/(base['var']*(we@we)))
  generic.append(np.stack(gg,1));collapsed.append(np.stack(ee,1))
 expected=np.stack(generic,1).max((1,2));eexp=np.stack(collapsed,1).max((1,2));u=end//BLOCK-1
 np.testing.assert_allclose(sc['GLR_Ramp'][:,u],expected,atol=1e-7,rtol=1e-7);np.testing.assert_allclose(sc['Ablation'][:,u],eexp,atol=1e-8,rtol=1e-8)
 for start in starts:direct.append(np.maximum(residual[:,start:].sum(1),0)**2/(base['var']*(end-start)))
 np.testing.assert_allclose(sc['Direct'][:,u],np.stack(direct,1).max(1),atol=1e-8)
np.testing.assert_array_equal(sc['M3'],sc['Ablation'])
# Causality for scores, estimated parameters, and prediction at both origins.
for origin in [12,15]:
 alt=x.copy();alt[:,round(origin/DT)+1:]+=100;ss=score_batch(alt,base,fixed,scales);end=round(origin/CLOCK)
 for key in sc:np.testing.assert_allclose(sc[key][:,:end],ss[key][:,:end],atol=1e-7,equal_nan=True)
 p1=predict(x,sc,base,origin,c4['0.35']['drift_window']);p2=predict(alt,ss,base,origin,c4['0.35']['drift_window'])
 for model in MODELS:np.testing.assert_allclose(p1[model],p2[model],atol=1e-7)
 np.testing.assert_array_equal(p1['M3'],p1['Ablation'])
seeds=SEEDS['calibration']+SEEDS['null_test']+sum(SEEDS['events'],[]);assert len(seeds)==len(set(seeds));assert min(seeds)>100000
checks=dict(exact_ramp_interval_integral='passed',generic_matched_GLR_vs_bruteforce='passed',direct_GLR_vs_bruteforce='passed',collapsed_likelihood_vs_independent_no_latent_model='passed',M3_ablation_exact_score_and_predictive_equivalence='passed',all_future_exclusion='passed',new_seed_disjointness='passed')
(OUT/'validation.json').write_text(json.dumps(checks,indent=2));print(checks)
# Post-evaluation pairing and frozen-reference checks.
if (OUT/'per_trajectory.csv.gz').exists():
 import hashlib
 from analyze import read
 from collections import defaultdict
 rr=read('per_trajectory.csv.gz');groups=defaultdict(list)
 for r in rr:groups[(r['sigma'],r['tau'],r['strength'],r['cohort'],r['trajectory'])].append(r)
 assert len(groups)==36*2*N
 for values in groups.values():
  assert len(values)==len(MODELS) and set(r['model'] for r in values)==set(MODELS)
  for field in ['failure','X_alarm']:np.testing.assert_allclose([r[field] for r in values],np.full(6,values[0][field]),equal_nan=True)
  mm={r['model']:r for r in values}
  for field in ['alarm','onset_estimate','shape_estimate','forecast_mse_at12','forecast_mse_at15']:np.testing.assert_equal(mm['M3'][field],mm['Ablation'][field])
 cc=json.loads((OUT/'calibration.json').read_text())
 for sig in SIGMAS:assert cc[str(sig)]['failure_threshold']==c4[str(sig)]['thresholds']['FAIL']
 st=json.loads((OUT/'TEST_START.json').read_text());assert st['protocol_sha256']==hashlib.sha256((OUT/'PROTOCOL.json').read_bytes()).hexdigest();assert st['protocol_markdown_sha256']==hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest()
 checks['paired_trajectories_and_identical_ablation']='passed';checks['C4_failure_threshold_frozen']='passed';checks['protocol_frozen_before_test']='passed'
(OUT/'validation.json').write_text(json.dumps(checks,indent=2));print('Paired controls and frozen references verified')

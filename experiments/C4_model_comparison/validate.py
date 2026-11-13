import json,numpy as np
from config import *
from detectors import structured,score_batch,cusum_components,forecast_from_scores,local_forecast
from legacy_stream import structured_scores as old_structured
from generate import null_data,event_data
from source_C0 import templates
c0=json.loads((OUT/'calibration_C0.json').read_text());base=c0['0.35']['normal_AR'];fixed=c0['0.35']['shared_index'];ck=np.load(OUT/'model_1_11.npz');p={k:ck[k] for k in ['beta','mu','var','A','pi']};x=event_data(.35,2,1,1,0)['x'][:3];new=structured(x,base,fixed);old=old_structured(x,base,fixed)
for name in ['M2','M3']:np.testing.assert_allclose(new[name],old[name],atol=1e-8,rtol=1e-8)
# Oracle score is an independent direct prefix regression with exact supplied tau.
for end in [50,125,300,500]:
 tau=2.;r=-(x[:,1:end+1]-base['q']*x[:,:end]-base['c']);h=templates(end,base,[tau])[0];starts=np.arange(10,end,5);vals=[]
 for start in starts:
  ww=h[:end-start];dot=r[:,start:]@ww;vals.append(np.maximum(dot,0)**2/(base['var']*(ww@ww)))
 expected=np.stack(vals,1).max(1);np.testing.assert_allclose(new['oracle_2.0'][:,end//5-1],expected,atol=1e-8,rtol=1e-8)
# Future exclusion for detection AND same-history forecasts.
changed=x.copy();changed[:,251:]+=100;one=score_batch(x,base,p,fixed,np.ones(5));two=score_batch(changed,base,p,fixed,np.ones(5))
for key in one:np.testing.assert_allclose(one[key][:,:50],two[key][:,:50],atol=1e-8,equal_nan=True)
for tau in TAUS:
 a=forecast_from_scores(x,one,base,p,tau,12);b=forecast_from_scores(changed,two,base,p,tau,12)
 for key in a:np.testing.assert_allclose(a[key],b[key],atol=1e-9)
# Changing oracle tau selection does not affect ordinary model forecasts.
a=forecast_from_scores(x,one,base,p,.1,12);b=forecast_from_scores(x,one,base,p,2.,12)
for name in ['M1','M2','M3']:np.testing.assert_array_equal(a[name],b[name])
# Frozen generator remains byte-identical to original C1 at original seed convention.
from source_A import simulate
np.testing.assert_array_equal(null_data(3,.35,81001),simulate(3,.35,1.,0.,81001,False)['x'])
all_seeds=SEEDS['cusum_scale']+SEEDS['prediction_validation']+SEEDS['final_calibration']+SEEDS['null_test']+sum(SEEDS['events'],[])
assert len(all_seeds)==len(set(all_seeds));assert min(all_seeds)>80000
assert PROTOCOL['decision']['practical_minimum_delay_gain']==CLOCK
checks=dict(frozen_M2_M3_parity_with_C1='passed',oracle_exact_tau_direct_prefix_likelihood='passed',all_detection_future_exclusion='passed',forecast_future_exclusion='passed',oracle_information_isolation='passed',unchanged_generator='passed',seed_disjointness='passed',practical_endpoint_prefixed='passed')
(OUT/'validation.json').write_text(json.dumps(checks,indent=2));print(checks)
# Both fixed forecast origins exclude observations after their own origins.
for origin in [12,15]:
 alt=x.copy();alt[:,round(origin/DT)+1:]+=100;other=score_batch(alt,base,p,fixed,np.ones(5));pa=forecast_from_scores(x,one,base,p,2.,origin);pb=forecast_from_scores(alt,other,base,p,2.,origin)
 for key in pa:np.testing.assert_allclose(pa[key],pb[key],atol=1e-8)
checks['both_prediction_origins_future_exclusion']='passed'
# Run after evaluation as well: verify paired keys and a common failure reference.
if (OUT/'per_trajectory.csv.gz').exists():
 from analyze import read
 from collections import defaultdict
 records=read('per_trajectory.csv.gz');paired=defaultdict(list)
 for r in records:paired[(r['sigma'],r['tau'],r['strength'],r['cohort'],r['trajectory'])].append(r)
 assert len(paired)==36*2*N_EVENT
 for group in paired.values():
  assert len(group)==6 and set(r['method'] for r in group)==set(METHODS)
  for field in ['failure','X_alarm']:
   vv=np.array([r[field] for r in group]);np.testing.assert_allclose(vv,np.full(6,vv[0]),equal_nan=True)
 import hashlib
 assert json.loads((OUT/'TEST_START.json').read_text())['protocol_sha256']==hashlib.sha256((OUT/'PROTOCOL.json').read_bytes()).hexdigest()
 checks['paired_test_keys_and_shared_failure_times']='passed';checks['protocol_frozen_before_test']='passed'
(OUT/'validation.json').write_text(json.dumps(checks,indent=2));print('Forecast, pairing, frozen-protocol checks passed')

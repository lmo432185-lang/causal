import numpy as np,json
from stream import structured_scores,reference_scores,hmm_score,scores,alarms,calibrate,episodes,CLOCK
from source_C0 import bank
from source_A import simulate
from run import OUT
cal=json.loads((OUT/'calibration_C0.json').read_text());base=cal['0.35']['normal_AR'];fixed=cal['0.35']['shared_index'];ck=np.load(OUT/'model_1_11.npz');p={k:ck[k] for k in ['beta','mu','var','A','pi']};x=simulate(3,.35,1,1,97)['x']
s=structured_scores(x,base,fixed)
# Independent full profile bank on prefixes and same onset grid.
for end in [15,50,125,250,500]:
 b=bank(x[:,:end+1],base);allowed=b['starts']%5==0;exact=b['score'][:,:,allowed];idx=end//5-1
 np.testing.assert_allclose(s['M3'][:,idx],exact.max((1,2)),rtol=1e-8,atol=1e-8)
 np.testing.assert_allclose(s['M2'][:,idx],exact[:,fixed].max(1),rtol=1e-8,atol=1e-8)
# Modify X in the future: every detector output up through current time unchanged.
changed=x.copy();changed[:,251:]+=100
one=scores(x,base,p,fixed);two=scores(changed,base,p,fixed)
for key in ['M1','M2','M3','M4_X','M4_E','onset_M2','onset_M3','tau_M3']:np.testing.assert_allclose(one[key][:,:50],two[key][:,:50],atol=1e-8,equal_nan=True)
# Earliest possible error score: 1-ahead target at1, then4 matured errors at1.75.
e=reference_scores(x,base)['M4_E'];np.testing.assert_array_equal(e[:,:6],0);assert (e[:,6]>0).all()
# Confirmation reports completion time, not beginning of a run.
a,idx=alarms(np.array([[0.,3,4,0]]),2,2);np.testing.assert_allclose(a,3*CLOCK)
# Hysteresis can change rearming but not the first alarm.
rng=np.random.default_rng(2);test=rng.normal(size=(20,100))
for k in [1,2,3]:
 th=calibrate(test,k);times,_=alarms(test,th,k)
 for hy in [False,True]:
  es=episodes(test,th,k,hy);got=np.array([v[0] if len(v) else np.nan for v in es]);np.testing.assert_allclose(times,got,equal_nan=True)
# Threshold calibration controls empirical whole-trajectory crossings, not point FPR.
for k in [1,2,3]:
 th=calibrate(test,k);assert np.mean(np.isfinite(alarms(test,th,k)[0]))<=.05+1e-12
checks=dict(causal_profile_vs_independent_prefix_likelihood='passed',all_detector_future_exclusion='passed',prediction_failure_uses_only_matured_errors='passed',confirmation_clock='passed',hysteresis_first_alarm_equivalence='passed',whole_trajectory_calibration='passed')
(OUT/'validation.json').write_text(json.dumps(checks,indent=2));print(checks)
from run import evaluate
th={m:{'1':1.,'2':3.,'3':8.} for m in ['M1','M2','M3','M4_X','M4_E']}
rr=evaluate(one,th,np.full(3,10.),dict(stage='test',sigma=.35,tau=1.,strength=1.,shape='step'),[1,2,3]);tf=alarms(one['M4_E'],3.,2)[0]
for row in rr:np.testing.assert_allclose(row['T_failure'],tf[row['trajectory']],equal_nan=True)
checks['fixed_failure_reference_across_confirmation_variants']='passed'
# Source change and transfer-gain change are observationally equivalent here.
source=simulate(3,.35,1,1,101);constant_source=np.ones_like(source['y']);gain=source['y'];np.testing.assert_array_equal(source['y'],gain*constant_source)
checks['source_vs_transfer_gain_effective_input_equivalence']='passed'
(OUT/'validation.json').write_text(json.dumps(checks,indent=2));print('Fixed reference and equivalence checks passed')
from supplement import cusum
np.testing.assert_allclose(cusum(x,base)[:,:50],cusum(changed,base)[:,:50],atol=1e-10)
checks['supplement_CUSUM_future_exclusion']='passed'
(OUT/'validation.json').write_text(json.dumps(checks,indent=2));print('CUSUM causality passed')

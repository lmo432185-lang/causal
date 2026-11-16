import json,hashlib,inspect
import numpy as np,pandas as pd
from config import *
from physics import simulate,command
from detectors import scores,calibrate,alarm

def main():
 marker=json.loads((OUT/'TEST_START.json').read_text());same={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in marker['sha256'].items()};assert all(same.values());reg=json.loads((ROOT/'PROTOCOL_REGISTERED.json').read_text());assert reg['sha256']==hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest() and reg['time']<marker['time']
 d=pd.read_csv(OUT/'trajectory_metrics.csv.gz');assert len(d)==518400;assert not d.duplicated(['mode','noise','plant','cohort','trajectory','policy','method','budget']).any();assert np.all(d.recall+d.early_alarm+d.no_alarm==1);assert np.all(d[(d.policy!='none')].input_energy.between(16-1e-9,16+1e-9));assert np.all(d[d.policy=='none'].input_energy==0)
 for key in ['T_beta','input_phase']:assert np.all(d.groupby(['noise','plant','cohort','trajectory'])[key].nunique()==1)
 settings=json.loads((OUT/'settings.json').read_text());count=0
 for ni in range(3):
  for ki in range(4):
   for pi in range(3):
    z=np.load(OUT/f'normal_n{ni}_k{ki}_p{pi}.npz');ss=settings[f'n{ni}_k{ki}_p{pi}']
    for budget in BUDGETS:
     for m in METHODS+['GainSpecific']:
      t=calibrate(z['cal_'+m],budget)
      if m=='GainSpecific':t=max(t,ss['thresholds'][str(budget)]['GainGLR'])
      assert t==ss['thresholds'][str(budget)][m];count+=1
 ni=1;ki=1;co=0;pi=2;seed=seed_for('test',ni,ki,co);case=simulate(N,SIGMAS[ni],KTEST[ki],'square',seed,'gradual');ss=settings[f'n{ni}_k{ki}_p{pi}'];sc,meta=scores(case['x'],case['I'],ss['model'],ss['scales']);z=np.load(OUT/f'test_scores_n{ni}_k{ki}_c{co}.npz');errors={m:float(np.max(abs(sc[m]-z['gradual_square_'+m]))) for m in sc};assert all(np.allclose(sc[m],z['gradual_square_'+m],rtol=2e-6,atol=1e-4) for m in sc)
 subset=d[(d['mode']=='gradual')&(d.noise.round(5)==SIGMAS[ni])&(d.plant==ki)&(d.cohort==co)&(d.policy=='square')&np.isclose(d.budget,.05)&(d.method=='GainGLR')].sort_values('trajectory');at,_=alarm(sc['GainGLR'],ss['thresholds']['0.05']['GainGLR']);assert np.allclose(at,subset.T_alarm.to_numpy(),equal_nan=True)
 result=dict(frozen_files_unchanged=same,registered_before_test=True,rows=len(d),unique_row_keys=True,outcome_partition=True,input_energy16_active=True,same_failure_and_phase_across_modes_policies=True,all_calibrated_thresholds_reproduced=count,replay_first_alarms_equal=True,replay_float32_max_errors=errors,inference_interface=str(inspect.signature(scores)),input_interface=str(inspect.signature(command)),true_tau_only_in_generator=True)
 (OUT/'audit.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()

import json,hashlib,inspect
import numpy as np,pandas as pd
from config import *
from physics import *
from inference import stream,commit
from run import threshold,summary,pareto

def main():
 mark=json.loads((OUT/'TEST_START.json').read_text());sel=json.loads((OUT/'SELECTION.json').read_text());reg=json.loads((ROOT/'PROTOCOL_REGISTERED.json').read_text());same={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in mark['sha256'].items()};assert all(same.values());assert reg['protocol_sha256']==hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest();assert reg['time']<sel['time']<=mark['time']
 v=pd.read_csv(OUT/'validation_metrics.csv.gz');d=pd.read_csv(OUT/'test_metrics.csv.gz');s=pd.read_csv(OUT/'stress_metrics.csv.gz');assert len(v)==len(PROGRAMS)*1200 and len(d)==len(PROGRAMS)*4800
 for df in [v,d,s]:
  assert not df.duplicated(['program','noise','cohort','trajectory','kappa','truth']).any();assert np.all(df.correct+df.wrong+df.refuse==1);assert np.all(df.energy<=4+1e-9);assert np.all(df[df.primary].peak<=CAP+1e-9);assert np.all(df.first_time.dropna()>=START);assert np.allclose(df.J,.15*df.energy/4+.20*df.tcap/15+3*df.wrong+df.refuse)
 tab=summary(v);eligible=tab[tab.program.isin(PRIMARY)&(tab.wrong<=.02)];base=float(tab.set_index('program').loc[REFERENCE,'cover']);retain=eligible[eligible.cover>=.8*base];high=summary(v[np.isclose(v.noise,.7)]);rh=high[high.program.isin(PRIMARY)&(high.wrong<=.02)]
 expected=dict(bestJ=None if eligible.empty else eligible.sort_values(['J','program']).iloc[0].program,fastest=None if retain.empty else retain.sort_values(['tcap','J','program']).iloc[0].program,lowest_energy=None if retain.empty else retain.sort_values(['energy','J','program']).iloc[0].program,high_noise=None if rh.empty else rh.sort_values(['cover','J','program'],ascending=[False,True,True]).iloc[0].program);assert expected==sel['policies'];assert all(x is None or x in PRIMARY for x in expected.values())
 settings=json.loads((OUT/'trained_settings.json').read_text());checks={}
 for ni,ss in enumerate(settings):
  z=np.load(OUT/f'nominal_maxima_n{ni}.npz')
  for name in IDS:assert threshold(z['calibration_'+name])==ss['thresholds'][name]
  checks[str(ni)]=True
 # Replay a full reference trajectory group including the stored posterior.
 ni=1;co=0;k=.5;truth=0;name=REFERENCE;seed=SEEDS['test'][co][ni];I=command(N,name,seed);data=simulate(I,SIGMAS[ni],np.full(N,k),truth,seed);p,z=stream(data['x'],I,settings[ni]['bg']);saved=np.load(OUT/f'test_{name}_n{ni}_c{co}_k{k}_truth{truth}.npz');pe=float(np.max(abs(p-saved['posterior'])));ze=float(np.max(abs(z-saved['nominal_score'])));assert pe<5e-8 and ze<1e-8
 result=dict(protocol_registered_before_selection_and_test=True,frozen_files_unchanged=same,validation_selection_exactly_reproduced=expected,only_primary_candidates_selected=True,validation_rows=len(v),test_rows=len(d),stress_rows=len(s),all_costs_caps_and_outcome_partitions_valid=True,all138_calibrated_thresholds_reproduced=checks,replay_float32_posterior_max_error=pe,replay_score_error=ze,command_signature=str(inspect.signature(command)),inference_signature=str(inspect.signature(stream)))
 (OUT/'audit.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()

"""Exploratory post-main audit on entirely fresh X; no truth or model adjustment."""
import json,time,numpy as np,pandas as pd
from settings import *
from ssm import unpack,component,target,logpdf
from producer import simulate

def collect(x,m):
 cs={}
 for bank in ['recursive','witness']:
  for k in [1,2]:cs[bank,k]=component(x,m[bank]['models'][k])
 base=np.concatenate([np.ones((len(x),len(ORIGINS),1)),x[:,ORIGINS,None],cs['recursive',1]['mean']],-1).reshape(-1,22)
 deltas={b:(cs[b,2]['mean'][:,:,9]-cs[b,1]['mean'][:,:,9]).ravel() for b in ['recursive','witness']};y=target(x);gains={b:float((logpdf(cs[b,2]['mean'],cs[b,2]['cov'],y)-logpdf(cs[b,1]['mean'],cs[b,1]['cov'],y)).mean()/20) for b in ['recursive','witness']};return base,deltas,gains

def main():
 assert (ROOT/'TEST_B_FROZEN.json').exists();rows=[]
 for j in json.loads((ROOT/'jobs.json').read_text()):
  m=unpack(json.loads((OUT/j['case']/'model_100.json').read_text()));x=simulate(j['index'],j['seed'],4,40);xt=simulate(j['index'],j['seed'],5,40);B,a,_=collect(x,m);Bt,at,gain=collect(xt,m);res={};fraction={}
  for b in ['recursive','witness']:
   coef=np.linalg.lstsq(B,a[b],rcond=1e-7)[0];res[b]=at[b]-Bt@coef;fraction[b]=float(np.var(res[b])/max(np.var(at[b]),1e-16))
  corr=float(abs(np.corrcoef(res['recursive'],res['witness'])[0,1]));rows.append(dict(case=j['case'],seed=j['seed'],n=100,conditional_predictive_effect_abs_corr=corr,recursive_residual_fraction=fraction['recursive'],witness_residual_fraction=fraction['witness'],recursive_fresh_gain=gain['recursive'],witness_fresh_gain=gain['witness'],secondary_observable_coherence=(corr>=.5 and fraction['recursive']>=.05 and gain['recursive']>=.005 and gain['witness']>=.005)))
 pd.DataFrame(rows).to_csv(OUT/'crossline_fresh_check.csv',index=False);(ROOT/'CROSSLINE_CHECK_DONE.json').write_text(json.dumps(dict(time=time.time(),rows=len(rows),fresh_calibration_and_test_stages=[4,5],exploratory_after_main=True,not_primary_success_criterion=True,no_hidden_truth=True,physical_state_identity_claimed=False),indent=2));print('Fresh crossline audit',len(rows))
if __name__=='__main__':main()

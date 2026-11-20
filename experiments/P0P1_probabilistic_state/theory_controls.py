"""Post-evaluation algebraic audit; not a trained method and never feeds parameters back."""
import json,time,itertools
import numpy as np,pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from settings import ROOT,DATA,OUT,T,START
from model import plant,spectrum,infer

def aliases(theta):
 a,b,c,u,v,q,r=theta;co=np.cos(np.linspace(0,np.pi,65));Da=1+a*a-2*a*co;Db=1+b*b-2*b*co;N=q*Da*Db+v*v*Da+v*v*u*u;B=np.stack([np.ones(len(co)),co,co**2],axis=1);nc=np.linalg.lstsq(B,N,rcond=None)[0];out=[]
 for aa,bb,cc in itertools.permutations([a,b,c]):
  da=1+aa*aa-2*aa*co;db=1+bb*bb-2*bb*co;pc=np.linalg.lstsq(B,da*db,rcond=None)[0];qq=nc[2]/pc[2];force=nc-qq*pc;vv2=-force[1]/(2*aa);uu2=force[0]/vv2-1-aa*aa if vv2>0 else -1
  if qq>0 and vv2>0 and uu2>1e-10:
   alt=np.array([aa,bb,cc,np.sqrt(uu2),np.sqrt(vv2),qq,r]);rel=float(np.max(abs(spectrum(theta)-spectrum(alt)))/np.max(spectrum(theta)));assert rel<1e-10;out.append(dict(theta=alt.tolist(),spectrum_relative_error=rel,identity=bool(np.max(abs(alt-theta))<1e-9)))
 return out

def main():
 records=[];examples={};private=json.loads((DATA/'sealed_manifest.json').read_text())
 for meta in private:
  if meta['seed']!=101:continue
  p=meta['params'];theta=np.array([p['a'],p['b'],p['c'],p['alpha']*np.sqrt(p['qy']/p['qz']),p['beta']*np.sqrt(p['qz']),p['qx'],p['obs']**2]);alts=aliases(theta);records.append(dict(tag=meta['tag'],true_canonical_theta=theta.tolist(),canonical_chain_aliases=alts,exact_timing_alternatives=[1,2,3]))
  if meta['tag'] not in ['baseline','information_rich']:continue
  # Fixed deterministic choice: legal nonidentity alias with largest target persistence c, no performance selection.
  candidates=[a for a in alts if not a['identity']]
  if not candidates:continue
  alt=max(candidates,key=lambda x:x['theta'][2]);x=np.load(DATA/meta['case']/'X.npy')[-200:];original=infer(x,plant(theta,1));alternative=infer(x,plant(alt['theta'],1));examples[meta['tag']+'_true_mean']=original['mean'][0,START:170,:2];examples[meta['tag']+'_alias_mean']=alternative['mean'][0,START:170,:2];examples[meta['tag']+'_X']=x[0,START:170];examples[meta['tag']+'_true_sd']=np.sqrt(np.diagonal(original['cov'][START:170,:2,:2],axis1=-2,axis2=-1));examples[meta['tag']+'_alias_sd']=np.sqrt(np.diagonal(alternative['cov'][START:170,:2,:2],axis1=-2,axis2=-1));records[-1]['fixed_alias_demonstration']=dict(theta=alt['theta'],max_one_step_X_mean_difference=float(np.max(abs(original['forecast']-alternative['forecast']))),max_one_step_X_variance_difference=float(np.max(abs(original['forecast_var']-alternative['forecast_var']))),max_sequence_LL_difference=float(np.max(abs(original['trajectory_ll']-alternative['trajectory_ll']))),Y_mean_RMS_difference=float(np.sqrt(np.mean((original['mean'][:,:,0]-alternative['mean'][:,:,0])**2))),Z_mean_RMS_difference=float(np.sqrt(np.mean((original['mean'][:,:,1]-alternative['mean'][:,:,1])**2))))
 # Scale gauge audit on an unrelated fixed chain: physical amplitudes cannot be anchored by X.
 theta=np.array([.61,.43,.27,.33,.42,.28,.07]);A,Q,R,P=plant(theta,1);S=np.diag([2.,.3,1.]);A2=S@A@np.linalg.inv(S);Q2=S@Q@S;P2=solve_discrete_lyapunov(A2,Q2);g1=np.array([(np.linalg.matrix_power(A,k)@P)[2,2] for k in range(100)]);g2=np.array([(np.linalg.matrix_power(A2,k)@P2)[2,2] for k in range(100)]);assert max(abs(g1-g2))<1e-12
 (OUT/'identifiability_controls.json').write_text(json.dumps(dict(time=time.time(),scope='post-evaluation algebraic explanation, not new model fitting or confirmation metric',records=records,scale_gauge_covariance_max_difference=float(max(abs(g1-g2)))),indent=2));np.savez_compressed(OUT/'alias_examples.npz',**examples);print('Exact alias audit completed',len(records),'conditions')
if __name__=='__main__':main()

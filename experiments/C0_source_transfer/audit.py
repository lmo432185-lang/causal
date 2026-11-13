import numpy as np,json
from scipy.linalg import expm
from run import OUT,DT,SIGMAS,bank,infer,evaluate,forecast,writecsv

def alternative(n,sigma,seed,kind):
 rng=np.random.default_rng(seed);x=np.zeros((n,501));x[:,0]=rng.normal(10,.3,n);u=np.zeros_like(x);y=np.zeros((n,500));y[:,200:]=1;hidden=np.zeros((n,4));hidden[:,0]=x[:,0];hidden[:,3]=1;q=np.exp(-.15*DT);ns=sigma*np.sqrt((1-q*q)/.3)
 if kind=='two_stage':
  mats=[]
  for level in [0,1]:
   A=np.array([[-.15,0,-1.5,1.5],[0,-2,0,2*level],[0,2,-2,0],[0,0,0,0.]])
   mats.append(expm(A*DT))
  for t in range(500):
   hidden=hidden@mats[int(t>=200)].T;hidden[:,0]+=ns*rng.normal(size=n);x[:,t+1]=hidden[:,0];u[:,t+1]=hidden[:,2]
 elif kind=='dead_time':
  uu=np.zeros(n);r=np.exp(-DT/.5);j=(r-q)/(.15-2);k=(1-q)/.15
  for t in range(500):
   source=float(t>=210);x[:,t+1]=q*x[:,t]+1.5*k-1.5*(source*k+(uu-source)*j)+ns*rng.normal(size=n);uu=source+(uu-source)*r;u[:,t+1]=uu
 else:raise ValueError(kind)
 return dict(x=x,y=y,u=u)

def main():
 cal=json.loads((OUT/'calibration.json').read_text());rows=[];examples={};summary=[]
 for gi,sigma in enumerate(SIGMAS):
  base=cal[str(sigma)]['normal_AR'];th=cal[str(sigma)]['thresholds']['M3']
  for kind in ['two_stage','dead_time']:
   data=alternative(200,sigma,25000+gi,kind);est=infer(bank(data['x'],base),th['full']);prefix=infer(bank(data['x'][:,:301],base),th['prefix']);pred=forecast(prefix,data['x'][:,300],base);rr=evaluate(data,est,'M3',dict(sigma=sigma,tau=.5,strength=1,kind=kind),pred,.5)
   # Two-stage model has no single first-order true tau: discard that comparison.
   if kind=='two_stage':
    for r in rr:
     for key in ['tau_error','tau_interval_coverage']:r[key]=np.nan
   rows.extend(rr);examples[f'{kind}_{sigma}_x']=data['x'][0];examples[f'{kind}_{sigma}_Y']=data['y'][0];examples[f'{kind}_{sigma}_U']=data['u'][0];examples[f'{kind}_{sigma}_on']=est['on'][0]
   from source_A import statistics
   summary.append(dict(kind=kind,sigma=sigma,detected=statistics([r['detected'] for r in rr]),onset_error=statistics([r['onset_error'] for r in rr]),onset_signed=statistics([r['onset_signed'] for r in rr]),tau_estimate=statistics([r['tau_estimate'] for r in rr]),U_rmse=statistics([r['U_rmse'] for r in rr])))
 # Algebraic observational equivalence when source shape is not fixed.
 t=np.arange(501)*DT;age=np.maximum(t-10,0);active=t>=10;u=(1-np.exp(-age/.5))*active;du=np.exp(-age/.5)/.5*active
 y_a=u+.5*du;y_b=u+1.*du
 eq=np.max(abs((y_a-u)/.5-(y_b-u)/1.))
 # Distinct action times can also be observationally equivalent with unknown delay.
 # Model A: action at10, deadtime.5, tau.5; B: action at10.5, deadtime0, tau.5.
 effect_a=np.maximum(t-10-.5,0);effect_b=np.maximum(t-10.5,0);ua=1-np.exp(-effect_a/.5);ub=1-np.exp(-effect_b/.5)
 np.testing.assert_allclose(ua,ub,atol=1e-14)
 ident=dict(source_tau_ambiguity={'tau_A':.5,'tau_B':1.,'max_dU_difference':float(eq),'statement':'Y_A=U+0.5*dU/dt and Y_B=U+1*dU/dt produce identical U and X but different Y and tau'},action_time_deadtime_ambiguity={'action_time_A':10,'deadtime_A':.5,'action_time_B':10.5,'deadtime_B':0,'max_U_difference':float(np.max(abs(ua-ub))),'max_X_distribution_difference':'exactly zero conditional on same X initial state and same innovations','statement':'unknown pure dead time is inseparable from source onset using only X'},gain_source_ambiguity={'b_A':1.5,'b_B':.75,'U_B':'2*U_A','statement':'same b*U gives same X; absolute hidden-state physical scale needs a gain convention or external calibration'})
 writecsv(OUT/'C0B_structural_stress.csv.gz',rows);(OUT/'C0B_summary.json').write_text(json.dumps(summary,indent=2));(OUT/'identifiability.json').write_text(json.dumps(ident,indent=2));np.savez_compressed(OUT/'C0B_examples.npz',**examples,t=t,U=u,Y_A=y_a,Y_B=y_b)
 print(json.dumps(summary,indent=2));print(json.dumps(ident,indent=2))
if __name__=='__main__':main()

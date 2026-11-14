import json
import numpy as np
from scipy.integrate import solve_ivp
from config import *
from physics import *
from inference import stream,commit

def main():
 bg=dict(q=np.exp(-.15*DT),c=1.5*(1-np.exp(-.15*DT))/.15,var=.35**2*(1-np.exp(-.3*DT))/.3);cmd=make_commands(3,990001);energies={k:np.sum(v*v,axis=1)*DT for k,v in cmd.items()}
 for k in INPUTS[1:]:assert np.allclose(energies[k],ENERGY)
 passive0=simulate(cmd['blank'],.35,np.ones(3)*.5,0,990002,True);passive1=simulate(cmd['blank'],.35,np.ones(3)*.5,1,990002,True);assert np.array_equal(passive0['x'],passive1['x']) and np.array_equal(passive0['U'],passive1['U']) and not np.array_equal(passive0['Y'],passive1['Y'])
 # Closed-form independent continuous source for H2's passive direct law yields same X as H1.
 kap=.5;Y=lambda t:1.5*(1-np.exp(-max(t-START,0)/kap)) if t>=START else 0
 I=np.zeros((1,T));I[:,200:]=1.5;h,U=response(I,[kap]);q=np.exp(-.15*DT);xx=0.;ode=0.;err=0.
 for t in range(T):
  xx=q*xx-h[0,t];ode=solve_ivp(lambda tt,v:[-.15*v[0]-Y(tt)],[t*DT,(t+1)*DT],[ode],rtol=1e-10,atol=1e-12).y[0,-1];err=max(err,abs(xx-ode))
 assert err<1e-7
 # Reduced-form recurrence is same physical model for every registered input, including random.
 red_err=0.;online=[];blank=True
 for key in INPUTS[1:]:
  I=cmd[key];h,U=response(I,np.ones(3)*kap);u=np.zeros(3);hh=[];r=np.exp(-DT/kap);a=.15;k=(1-q)/a;j=(r-q)/(a-1/kap)
  for t in range(T):
   hh.append(I[:,t]*k+(u-I[:,t])*j);u=r*u+(1-r)*I[:,t]
  red_err=max(red_err,float(np.max(abs(h-np.stack(hh,1)))))
  d=simulate(I,.35,np.ones(3)*kap,0,990002);p,z=stream(d['x'],I,bg,d['U']+.1*d['eta'],.01);prefix=255;pp,zp=stream(d['x'][:,:prefix+1],I,bg,(d['U']+.1*d['eta'])[:,:prefix],.01);delta=max(np.max(abs(p[:,:prefix//BLOCK]-pp)),np.max(abs(z[:,:prefix//BLOCK]-zp)));assert delta<1e-8;online.append(float(delta))
 assert red_err<1e-12
 d=simulate(cmd['blank'],.35,np.ones(3),1,990002);pp,_=stream(d['x'],cmd['blank'],bg);assert np.allclose(pp,.5);assert np.all(commit(pp)[1]==-1)
 # Equal-class first commits remain locked if later posterior reverses.
 pp=np.zeros((1,8,2));pp[:,:4,1]=.99;pp[:,:4,0]=.01;pp[:,4:,0]=.99;pp[:,4:,1]=.01;assert commit(pp)[1][0]==1
 # Independent exact likelihood at full prefix.
 I=cmd['I2_pulse'];d=simulate(I,.35,np.ones(3)*kap,0,990002);p,z=stream(d['x'],I,bg,d['U']+.1*d['eta'],.01);H,G=model_bank(I,bg);res=-(d['x'][:,1:]-bg['q']*d['x'][:,:-1]-bg['c']);V=d['U']+.1*d['eta'];sse=np.sum((res[:,None,:]-H)**2,axis=2)/bg['var']+np.sum((V[:,None,:]-G)**2,axis=2)/.01
 from scipy.special import logsumexp
 ll=-.5*sse;ev=np.stack([logsumexp(ll[:,:-1],1)-np.log(len(KGRID)),ll[:,-1]],1);p2=np.exp(ev-logsumexp(ev,1)[:,None]);likerr=float(np.max(abs(p[:,-1]-p2)));assert likerr<1e-8
 out=dict(equal_active_energy={k:v.tolist() for k,v in energies.items()},passive_X_U_identical=True,passive_distinct_source=True,independent_continuous_equivalence_error=err,reduced_form_all_registered_input_error=red_err,no_future_prefix_errors=online,blank_no_identity_commit=True,wrong_commit_not_overwritten=True,independent_joint_likelihood_error=likerr)
 (OUT/'validation.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()

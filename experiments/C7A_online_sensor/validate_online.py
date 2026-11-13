import json,hashlib
import numpy as np
from scipy.special import logsumexp
from config import *
from mechanisms import library,response
from generate import draw,equivalent_pair
from online import stream,commit

def main():
 bg=dict(q=np.exp(-.15*DT),c=1.5*(1-np.exp(-.15*DT))/.15,var=.35**2*(1-np.exp(-.3*DT))/.3);bank=library(bg,steps=T+1);d=draw(4,.35,.5,998001)['A_first'];errors=[];likelihood=[]
 for sensor in [None,'u','y']:
  V=None if sensor is None else d['U' if sensor=='u' else 'Y']+.1*d['eta'];s,p=stream(d['x'],bank,bg,2.,V,.01,sensor or 'u')
  prefix=225;sp,pp=stream(d['x'][:,:prefix+1],bank,bg,2.,None if V is None else V[:,:prefix],.01,sensor or 'u');err=float(max(np.max(abs(s[:,:prefix//BLOCK]-sp)),np.max(abs(p[:,:prefix//BLOCK]-pp))));assert err<1e-9;errors.append(err)
  # Independent prefix batch likelihood at a selected clock, same eligible origins.
  end=275;z=-(d['x'][:,1:end+1]-bg['q']*d['x'][:,:end]-bg['c']);ll=[]
  for m,b in bank.items():
   keep=b['on']<=end*DT-CLOCK+1e-10;h=b['h'][keep,:end];B=z@h.T/bg['var'];H=np.sum(h*h,axis=1)/bg['var']
   if V is not None:
    g=b[sensor][keep,1:end+1];B+=V[:,:end]@g.T/.01;H+=np.sum(g*g,axis=1)/.01
   from scipy.special import log_ndtr
   aa=H+.25;bf=np.log(2)-np.log(2)-.5*np.log(aa)[None,:]+B*B/(2*aa[None,:])+log_ndtr(B/np.sqrt(aa)[None,:]);ll.append(logsumexp(bf,axis=1)-np.log(len(H)))
  ll=np.stack(ll,1);pb=np.exp(ll-logsumexp(ll,axis=1)[:,None]);err=float(np.max(abs(pb-p[:,end//BLOCK-1])));assert err<1e-8;likelihood.append(err)
 # Endpoint-aligned sensor, independent closed form U(t_j).
 t=np.arange(1,T+1)*DT;expected=np.where(t>=10,1-np.exp(-np.maximum(t-10,0)/.5),0);assert np.max(abs(d['U']/d['amp'][:,None]-expected))<1e-12
 # Operational commit must preserve an earlier wrong decision, not retry using truth.
 pp=np.full((1,8,4),.01);pp[:,:,1]=.97;pp[:,4:,1]=.01;pp[:,4:,0]=.97;sc=np.ones((1,8))*100;at,it,pr=commit(sc,pp,10);assert pr[0]==1 and it[0]==.5
 a,b=equivalent_pair(4,.35,.5,998002);eq={'A':bank['A_first'],'E':dict(bank['A_first'],y=bank['A_first']['u'])};_,pp=stream(a['x'],eq,bg,2.);assert np.allclose(pp,.5)
 result=dict(no_future_max_errors=errors,batch_prefix_likelihood_errors=likelihood,sensor_endpoint_alignment=True,wrong_first_commit_not_overwritten=True,exact_equivalent_pair_X_posterior_half=True)
 (OUT/'online_validation.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()

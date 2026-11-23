import json
import numpy as np
from settings import ROOT,P,FULL
from core import *
def main():
 rng=np.random.default_rng(810271);x=rng.normal(size=(12,500));c,p,y=base_arrays(x,P);initial=learn(c,p,y);w=wseries(x,initial);wc,wp,wf=recursive_arrays(w);a=learn(wc,wp,wf);raw=apply(a,wc,wp)[:,:,0];f=reconstruct_raw_filter(initial,a,0);err=float(np.max(abs(raw-apply_raw_filter(x,f))));assert err<1e-9
 altered=x.copy();altered[:,181:]=rng.normal(size=altered[:,181:].shape)*100;wa=wseries(altered,initial);ww,pp,_=recursive_arrays(wa);raw2=apply(a,ww,pp)[:,:,0];prefix=float(np.max(abs(raw[:,:141]-raw2[:,:141])));assert prefix<1e-12
 feat=np.concatenate([c,wc],-1);head=fit_head(feat,y);dup=fit_head(np.concatenate([feat,2*wc],-1),y);de=float(np.max(abs(predict(head,feat)-predict(dup,np.concatenate([feat,2*wc],-1)))));assert de<1e-9
 result=dict(status='PASS',unrelated_development_data=True,raw_X_filter_identity_error=err,causal_prefix_error=prefix,duplicate_state_prediction_error=de);(ROOT/'DEVELOPMENT_VALIDATION.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()

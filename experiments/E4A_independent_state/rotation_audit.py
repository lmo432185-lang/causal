"""Similarity-coordinate control; never interpreted as a physical causal graph."""
import json,time,hashlib,numpy as np,pandas as pd
from settings import *
from producer import generate,SYSTEMS
from model import unpack
from predict import component
import native

def main():
 assert (ROOT/'TEST_B_FROZEN.json').exists();rows=[];angle=.61;T=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
 for i,meta in enumerate(SYSTEMS):
  b=unpack(json.loads((OUT/(meta['id']+'_4011')/'model_5000.json').read_text()));m=b['direct']['models']['factor2'];r=dict(m,F=T@m['F']@T.T,Q=T@m['Q']@T.T,C=m['C']@T.T,P=T@m['P']@T.T);x=generate(i,4011,1,NTEST)[:1];p=component(x,m);q=component(x,r);pe=float(np.max(abs(p['mean']-q['mean'])));ce=float(np.max(abs(p['cov']-q['cov'])));le=float(np.max(abs(p['post']['ll']-q['post']['ll'])));se=float(np.max(abs(p['post']['mean']@T.T-q['post']['mean'])));assert max(pe,ce,le,se)<1e-7;rows.append(dict(condition=meta['id'],forecast_mean_error=pe,forecast_cov_error=ce,loglik_error=le,transformed_state_error=se,original_offdiagonal_edges=int((abs(m['F']-np.diag(np.diag(m['F'])))>1e-8).sum()),rotated_offdiagonal_edges=int((abs(r['F']-np.diag(np.diag(r['F'])))>1e-8).sum())))
 pd.DataFrame(rows).to_csv(OUT/'rotation_audit.csv',index=False);(ROOT/'ROTATION_AUDIT_COMPLETE.json').write_text(json.dumps(dict(time=time.time(),all_passed=True,n=20,conclusion='state-coordinate lag edges can change with identical X law; not physical causal discovery'),indent=2));print('Rotation invariance verified',len(rows))
if __name__=='__main__':main()

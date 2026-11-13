"""Exploratory fixed conventional CUSUM benchmark on new event seeds.
Added after primary C1-C3 results; no tuning on either event test set.
"""
import json,numpy as np
from run import OUT,writecsv,wilson
from stream import scores,calibrate,alarms,CLOCK
from source_A import simulate,statistics,SIGMAS,TAUS

def cusum(x,base,allowance=.5):
 residual=-(x[:,1:]-base['q']*x[:,:-1]-base['c'])/np.sqrt(base['var']);acc=np.zeros(len(x));out=np.zeros((len(x),residual.shape[1]//5))
 for t in range(residual.shape[1]):
  acc=np.maximum(0,acc+residual[:,t]-allowance)
  if (t+1)%5==0:out[:,(t+1)//5-1]=acc
 return out

def main():
 c=json.loads((OUT/'calibration_C0.json').read_text());th=json.loads((OUT/'thresholds.json').read_text());rows=[];null=[];summary=[]
 for gi,sigma in enumerate(SIGMAS):
  base=c[str(sigma)]['normal_AR'];fixed=c[str(sigma)]['shared_index'];ck=np.load(OUT/f'model_{gi}_11.npz');p={k:ck[k] for k in ['beta','mu','var','A','pi']};val=simulate(2000,sigma,1,0,31000+gi,False)['x'];threshold=calibrate(cusum(val,base),2);nt=simulate(2000,sigma,1,0,35000+gi,False)['x'];na,_=alarms(cusum(nt,base),threshold,2);null.append(dict(sigma=sigma,method='CUSUM',threshold=threshold,FPR=wilson(int(np.isfinite(na).sum()),len(na))))
  null_scores=scores(nt,base,p,fixed)
  for model in ['M1','M2','M3','M4_X','M4_E']:
   ta=th[str(sigma)][model]['2'];alar=alarms(null_scores[model],ta,2)[0];null.append(dict(sigma=sigma,method=model,threshold=ta,FPR=wilson(int(np.isfinite(alar).sum()),len(alar))))
  for tau in TAUS:
   data=simulate(200,sigma,tau,1,34000+gi);sc=scores(data['x'],base,p,fixed);times={m:alarms(sc[m],th[str(sigma)][m]['2'],2)[0] for m in ['M1','M2','M3','M4_X','M4_E']};times['CUSUM']=alarms(cusum(data['x'],base),threshold,2)[0]
   for m,aa in times.items():
    for i,a in enumerate(aa):
     valid=np.isfinite(a) and a>=10;bb=times['M4_E'][i];cc=times['CUSUM'][i];vv=np.isfinite(bb) and bb>=10;vc=np.isfinite(cc) and cc>=10
     rows.append(dict(sigma=sigma,tau=tau,method=m,trajectory=i,alarm=a,valid=int(valid),delay=a-10 if valid else np.nan,lead_failure=bb-a if valid and vv else np.nan,lead_CUSUM=cc-a if valid and vc else np.nan))
 for sigma in SIGMAS:
  for tau in TAUS:
   for model in ['M1','M2','M3','M4_X','M4_E','CUSUM']:
    sub=[r for r in rows if r['sigma']==sigma and r['tau']==tau and r['method']==model];summary.append(dict(sigma=sigma,tau=tau,method=model,recall=wilson(sum(r['valid'] for r in sub),len(sub)),delay=statistics([r['delay'] for r in sub]),lead_failure=statistics([r['lead_failure'] for r in sub]),lead_CUSUM=statistics([r['lead_CUSUM'] for r in sub])))
 writecsv(OUT/'supplement_CUSUM.csv.gz',rows);(OUT/'supplement_CUSUM_summary.json').write_text(json.dumps(summary,indent=2));(OUT/'supplement_CUSUM_null.json').write_text(json.dumps(null,indent=2));print(json.dumps([r for r in summary if r['sigma']==.35 and r['method'] in ['M3','CUSUM']],indent=2))
if __name__=='__main__':main()

import json,gzip,time,hashlib,numpy as np,pandas as pd
from settings import *
from producer import generate,raw_generate
from model import unpack
from inference import mix
from predict import scores,history_scores

def main():
 for label in ['CODE_REGISTERED','PREDICTION_REGISTERED','CANDIDATES_FROZEN','FINAL_MODELS_FROZEN','TEST_A_FROZEN','TEST_B_FROZEN']:
  for f,h in json.loads((ROOT/(label+'.json')).read_text()).get('files',{}).items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
 table=pd.read_csv(OUT/'test_A.csv.gz');assert len(table)==480*20*20;extra=json.loads((OUT/'extra_plans.json').read_text());rows=[]
 for idx,case in [(0,'A0_4011'),(5,'B1_4011'),(8,'C0_4011'),(12,'D0_4011'),(17,'E1_4011'),(23,'CE_4011')]:
  train=generate(idx,4011,0,70);raw=raw_generate(idx,4011,0,70);assert train.shape==(70,1000) and np.array_equal(train,raw[:,:1000]);x=generate(idx,4011,11,20)[:1];assert x.shape==(1,5000)
  with gzip.open(OUT/(case+'.json.gz'),'rt') as f:b=unpack(json.load(f))
  b['plans'].update(extra[case]);cache={}
  for method in ['A','B','E','R0','R2','R3','StrongestSingle']:
   plan=b['plans'][method]
   if plan.get('type')=='ARhistory':ss=history_scores(x,plan['parameters'])
   elif plan['ids']:r,_=mix(x,b['models'],plan['ids'],plan.get('prior'),cache);ss=scores(x,r)
   else:continue
   stored=table.query('case==@case and method==@method and trajectory==0').iloc[0];err=abs(ss['NLL'][0]-stored.NLL);assert err<1e-8;rows.append(dict(case=case,method=method,error=err))
  plan=b['plans']['E'];r,_=mix(x,b['models'],plan['ids'],plan['prior'],{});z=x.copy();z[:,1300:]+=9.;q,_=mix(z,b['models'],plan['ids'],plan['prior'],{});ix=np.array(origins(5000))<1300;assert np.max(abs(r['mean'][:,ix]-q['mean'][:,ix]))<1e-9
 pd.DataFrame(rows).to_csv(OUT/'replay_audit.csv',index=False);(ROOT/'AUDIT_COMPLETE.json').write_text(json.dumps(dict(time=time.time(),all_passed=True,training_prefix=1000,common_test_length=5000,no_truth_opened=True,replays=len(rows)),indent=2))
if __name__=='__main__':main()

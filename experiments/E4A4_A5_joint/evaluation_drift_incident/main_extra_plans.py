import json,gzip,time,hashlib,numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
from scipy.special import logsumexp
from settings import *
from model import unpack
from producer import generate
from inference import validate
from predict import history_fit,history_scores

def one(j):
 with gzip.open(OUT/(j['case']+'.json.gz'),'rt') as f:b=unpack(json.load(f))
 p=b['plans']['E'];ix=[i for i,k in enumerate(p['ids']) if '_3_' in k];ids=[p['ids'][i] for i in ix];pr=np.array(p['prior'])[ix];pr-=logsumexp(pr);q=dict(J1_param=dict(ids=ids,prior=pr.tolist()));fixed=[]
 for bank in ['R','D']:
  for route in ['direct','residual']:
   keys=[bank+'_none',bank+'_obs',bank+'_'+route+'_latent',bank+('_direct_dense' if route=='direct' else '_joint_dense')];fixed.extend(k for k in keys if k in b['models'])
 raw=generate(j['index'],j['seed'],0,70);v=raw[40:50];_,post=validate(v,b['models'],fixed);q['J2_point']=dict(ids=fixed,prior=post.tolist());memids=['R_none','D_none'];_,mp=validate(v,b['models'],memids);q['MEMORY_POINT']=dict(ids=memids,prior=mp.tolist());choices={}
 for name in ['A','B','R0','M3']:
  choices[name]=validate(v,b['models'],b['plans'][name]['ids'])[0]
 h=history_fit(raw[:40],v);hp=dict(h,phi=h['phi'].tolist());q['Xhistory']=dict(type='ARhistory',ids=[],prior=[],parameters=hp);choices['Xhistory']=float(history_scores(v,h)['NLL'].mean());winner=min(choices,key=choices.get);q['StrongestSingle']=dict(q['Xhistory'] if winner=='Xhistory' else b['plans'][winner],selected_algorithm=winner,selection_scores=choices);return j['case'],q

def main():
 rows={}
 with ProcessPoolExecutor(max_workers=3) as pool:
  for f in as_completed([pool.submit(one,j) for j in json.loads((ROOT/'jobs.json').read_text())]):k,q=f.result();rows[k]=q
 p=OUT/'extra_plans.json';p.write_text(json.dumps(rows));(ROOT/'EXTRA_PLANS_FROZEN.json').write_text(json.dumps(dict(time=time.time(),files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()},tests_not_opened=True),indent=2));print('Parameter-only and structure-point controls frozen',flush=True)
if __name__=='__main__':main()

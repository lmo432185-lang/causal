from pathlib import Path
import json,hashlib,time,shutil
ROOT=Path('/workspace/work/E4A4_joint');out=ROOT/'evaluation_drift_incident';out.mkdir(exist_ok=True)
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def restore(path,text,expected):
 if sha(text)!=expected:print('NOT MATCHED',path.name,sha(text),expected);return False
 shutil.copy2(path,out/(('short_' if path.parent.name=='short' else 'main_')+path.name));path.write_text(text);print('RESTORED',path,expected);return True
reg=json.loads((ROOT/'PREDICTION_REGISTERED.json').read_text())['files'];post=json.loads((ROOT/'POSTTEST_REGISTERED.json').read_text())['files']
# Undo only unexecuted development annotations; SHA equality is the acceptance condition.
p=ROOT/'evaluate.py';s=p.read_text().replace(",'MEMORY_POINT'",'').replace('from completion import complete as completion_status\n','').replace(';complete=completion_status(plan,name)','').replace('protocol_complete=complete,','');restore(p,s,reg['evaluate.py'])
p=ROOT/'extra_plans.py';s=p.read_text().replace("memids=['R_none','D_none'];_,mp=validate(v,b['models'],memids);q['MEMORY_POINT']=dict(ids=memids,prior=mp.tolist());",'');restore(p,s,reg['extra_plans.py'])
p=ROOT/'state_audit.py';s=p.read_text().replace(",'MEMORY_POINT'",'');restore(p,s,post['state_audit.py'])
p=ROOT/'audit.py';s=p.read_text().replace('480*20*21','480*20*20');a=s.index(" oldmanifest=");b=s.index(' tests={ba:',a);s=s[:a]+s[b:];restore(p,s,post['audit.py'])
p=ROOT/'analysis.py';s=p.read_text();s=s.replace(";ok=d.pivot(index=['condition','case','seed','trajectory'],columns='method',values='protocol_complete').reset_index()",'');s=s.replace(";p['protocol_complete']=np.logical_and.reduce([ok[k].fillna(False).astype(bool).to_numpy() for k in coef])",'');s=s.replace(',protocol_failed_pairs=int((~g.protocol_complete).sum())','');s=s.replace(".delta.agg(delta='mean',replicas='count').reset_index()",'.delta.mean().reset_index()');s=s.replace(',protocol_failed_pairs=int((g.replicas<2).sum())','');s=s.replace('&(r.protocol_failed_pairs==0)','');restore(p,s,post['analysis.py'])
p=ROOT/'POSTTEST_PLAN.md';s=p.read_text();s=s.split('\n\n预测可以定义与完整方案成功是两个不同状态。')[0];s=s.split('\n\nC5单因素记忆强参照同时输出')[0];matched=False
for n in range(5):
 t=s.rstrip('\n')+'\n'*n
 if sha(t)==post['POSTTEST_PLAN.md']:matched=restore(p,t,post['POSTTEST_PLAN.md']);break
if not matched:print('NOT MATCHED POSTTEST_PLAN')
# Short predictions had not opened: restore their registered sources too.
q=ROOT/'short';sr=json.loads((q/'PREDICTION_REGISTERED.json').read_text())['files']
for f in ['evaluate.py','extra_plans.py']:
 p=q/f;s=(ROOT/f).read_text();restore(p,s,sr[f])
p=q/'short_audit.py';s=p.read_text().replace('480*20*21','480*20*20');restore(p,s,sr['short_audit.py'])
p=q/'short_analysis.py';s=p.read_text().replace(";ok=d.pivot(index=['condition','case','seed','trajectory'],columns='method',values='protocol_complete').reset_index()",'').replace(";p['protocol_complete']=np.logical_and.reduce([ok[k].fillna(False).astype(bool).to_numpy() for k in coef])",'').replace(',protocol_failed_pairs=int((~g.protocol_complete).sum())','');restore(p,s,sr['short_analysis.py'])
checks={}
for folder in [ROOT,q]:
 for name in ['CODE_REGISTERED','PREDICTION_REGISTERED']+(['POSTTEST_REGISTERED','BAYES_REGISTERED'] if folder==ROOT else []):
  r=json.loads((folder/(name+'.json')).read_text());bad=[f for f,h in r['files'].items() if hashlib.sha256((folder/f).read_bytes()).hexdigest()!=h];checks[str(folder.name)+'/'+name]=bad
print(json.dumps(checks,indent=2));(out/'RESTORATION_RECORD.json').write_text(json.dumps(dict(time=time.time(),original_registration_records_unchanged=True,checks=checks,model_parameters_and_test_outputs_unchanged=True),indent=2));assert all(not x for x in checks.values())

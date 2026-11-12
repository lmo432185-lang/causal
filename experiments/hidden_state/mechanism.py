import numpy as np,torch,json,csv
from run import simulate,information,OUT,SEEDS,ci
rows=list(csv.DictReader((OUT/'per_trajectory.csv').open())); result={}
for signal in ['aci','err']:
 deltas={g:[] for g in ['A2','A4']}
 for g in deltas:
  deltas[g]=[np.mean([float(r[signal+'_post'])-float(r[signal+'_pre']) for r in rows if r['group']==g and int(r['trajectory'])==j]) for j in range(100)]
 result[signal+'_post_minus_pre']={g:ci(v) for g,v in deltas.items()}
# Predeclared simple two-sided ACI shift detector. Calibration: A2 validation only.
cal=[]
for seed in SEEDS:
 ck=torch.load(OUT/f'A2_{seed}.pt',weights_only=False); va=simulate(20,[.5,1.],201); aci,_=information(ck['ssm'],(va[:,:,0]-ck['mu'])/ck['sd']); tt=np.arange(aci.shape[1])*.05
 cal.append(aci[:,(tt>=40)&(tt<60)].mean(1)-aci[:,(tt>=10)&(tt<30)].mean(1))
threshold=float(np.quantile(np.abs(np.mean(cal,0)),.95)); result['detector']={'calibration':'20 independent constant-gamma validation trajectories; average 3 model seeds; 95th percentile absolute post-pre ACI shift','threshold':threshold}
for g in ['A2','A4']:
 vals=[np.mean([float(r['aci_post'])-float(r['aci_pre']) for r in rows if r['group']==g and int(r['trajectory'])==j]) for j in range(100)]
 result['detector'][g+'_rate']=float(np.mean(np.abs(vals)>threshold))
(OUT/'mechanism.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

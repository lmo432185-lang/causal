"""Extract frozen numeric data for Chinese report; no refitting or outcome-based example selection."""
from pathlib import Path
import json,hashlib,sys
import numpy as np,pandas as pd
R=Path(__file__).parent;D=R/'figure_data';D.mkdir(exist_ok=True);W=Path('/workspace/work');sources={}
def record(p):sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
def csv(name,df):df.to_csv(D/name,index=False)
# L1, fixed seed101 first test trajectory.
p=W/'L1/results/seed101/test_predictions.npz';record(p);a=np.load(p);ids=np.arange(19,499);curves=pd.DataFrame(dict(time=ids,Y_true=a['truth'][0,:,0],Z_true=a['truth'][0,:,1],Y_L=a['L_only'][0,:,0],Z_L=a['L_only'][0,:,1],Y_current=a['X_current'][0,:,0],Z_current=a['X_current'][0,:,1]));csv('01_L1_hidden_curves.csv',curves)
p=W/'L1/results/seed101/probe_maps.json';record(p);mp=json.loads(p.read_text())['L_only'];mean=np.array(mp['mean']);sd=np.array(mp['sd']);l=(a['L'][0]-mean)/sd;csv('02_L1_coordinates.csv',pd.DataFrame(dict(time=ids,L1=l[:,0],L2=l[:,1])))
for f,name in [('summary.csv','02_03_L1_metrics.csv'),('latent_redundancy.json','L1_latent_redundancy.json')]:
 p=W/'L1/results'/f;record(p)
 if f.endswith('csv'):csv(name,pd.read_csv(p))
 else:(D/name).write_bytes(p.read_bytes())
# L2A same preselected test example.
p=W/'L2A/results/seed101/estimates.npz';record(p);a=np.load(p);c=pd.DataFrame(dict(time=ids))
for j,s in enumerate(['Y','Z']):
 c[s+'_true']=a['truth'][0,:,j]
 for key,short in [('A3_original_L','L'),('A4_Kalman_filter','KF'),('A5_RTS_smoother','RTS')]:c[s+'_'+short]=a[key][0,:,j]
csv('04_L2A_curves.csv',c);p=W/'L2A/results/model_covariances.npz';record(p);a=np.load(p);csv('06_L2A_variance.csv',pd.DataFrame(dict(time=np.arange(500),Y_KF=a['filter'][:,0,0],Z_KF=a['filter'][:,1,1],Y_RTS=a['smoother'][:,0,0],Z_RTS=a['smoother'][:,1,1])))
p=W/'L23/results/scan_summary.csv';record(p);csv('05_recoverability_scan.csv',pd.read_csv(p))
# L3B matched covariance curve and independently replayed samples, exact archived seeds.
p=W/'L23/results/equivalence_proofs.json';record(p);proof=json.loads(p.read_text())[0];sys.path.insert(0,str(W/'L23'));from core import generate,plant
from scipy.linalg import solve_discrete_lyapunov
curves=pd.DataFrame(dict(lag=np.arange(41)))
for label,key in enumerate(['H0','H1']):
 params=proof[key];A,Q,r=plant(**params);P=solve_discrete_lyapunov(A,Q);curves[f'theory{label}']=[(np.linalg.matrix_power(A,k)@P)[-1,-1] for k in range(41)];samples=[]
 for seed in [101,202,303]:
  x,_=generate(params,seed+50000+100*label,n=200,stationary=True);samples.append(np.stack([(x[:,:500-k]*x[:,k:]).mean(1) for k in range(41)],1))
 vals=np.concatenate(samples);curves[f'empirical{label}']=vals.mean(0);curves[f'low{label}']=vals.mean(0)-1.96*vals.std(0,ddof=1)/np.sqrt(len(vals));curves[f'high{label}']=vals.mean(0)+1.96*vals.std(0,ddof=1)/np.sqrt(len(vals))
csv('07_observational_equivalence.csv',curves)
p=W/'L3D/results/edge_summary.csv';record(p);q=pd.read_csv(p);q=q[(q.group=='D2_indirect')&(q.seed==101)&q.family.str.startswith('rank2_rotate')];csv('08_rotation_gains.csv',q)
p=W/'L3D/results/rotations.json';record(p);r=json.loads(p.read_text());(D/'08_rotation_examples.json').write_text(json.dumps([v for v in r if v['group']=='D2_indirect' and v['seed']==101],indent=2))
# L4 EM trace is a model fit curve, not a causal correctness curve.
rows=[]
for kind in ['G1','G2','G3','G4']:
 p=W/'L4/results/case001_101'/f'{kind}_initializations.json';record(p);z=max(json.loads(p.read_text()),key=lambda v:v['validation_penalized'])
 for i,ll in enumerate(z['fit']['trace'],1):rows.append(dict(model=kind,iteration=i,loglik_per_training_point=ll/(600*1000)))
csv('09_L4_training_curves.csv',pd.DataFrame(rows));p=W/'L4/results/system_level_confusion_A.csv';record(p);csv('09_L4_system_confusion.csv',pd.read_csv(p));p=W/'L4/results/system_level_frozen_validation_choices.csv';record(p);csv('L4_system_choices.csv',pd.read_csv(p))
# C7C1 fixed medium-noise, tau .5, first cohort, first trajectory.
p=W/'C7C1/results/test_scores_n1_k1_c0.npz';record(p);z=np.load(p);t=np.arange(641)*.05;curves=pd.DataFrame(dict(time=t))
for pol in ['none','square']:
 curves[pol+'_X']=z[f'gradual_{pol}_example_x'][0];curves[pol+'_beta']=np.r_[z[f'gradual_{pol}_example_beta'][0],z[f'gradual_{pol}_example_beta'][0,-1]]
csv('10_failure_example.csv',curves)
p=W/'C7C1/results/trajectory_metrics.csv.gz';record(p);allm=pd.read_csv(p);main=allm[np.isclose(allm.budget,.05)];q=main[(main['mode']=='gradual')&(main.noise==.35)&(main.plant==1)&(main.cohort==0)&(main.trajectory==0)&(main.method=='GainGLR')&main.policy.isin(['none','square'])];csv('10_failure_example_times.csv',q[['policy','T_beta','T_alarm','T_prediction','recall','lead']])
methods=[('none','GLR'),('none','GainGLR'),('square','GLR'),('square','GainGLR')];rows=[];leads=[]
for mode in ['abrupt','gradual']:
 for policy,method in methods:
  g=main[(main['mode']==mode)&(main.policy==policy)&(main.method==method)]
  for delay in np.arange(0,10.001,.25):rows.append(dict(mode=mode,policy=policy,method=method,delay=delay,recall_all=float(((g.recall==1)&(g.delay<=delay)).mean()),n=len(g)))
  for margin in np.arange(.25,2.001,.25):leads.append(dict(mode=mode,policy=policy,method=method,lead_required=margin,coverage_all=float(((g.recall==1)&(g.reference_valid==1)&(g.lead>=margin)).mean()),n=len(g)))
csv('11_detection_cdf.csv',pd.DataFrame(rows));csv('12_warning_coverage.csv',pd.DataFrame(leads));p=W/'C7C1/results/normal_audit.csv';record(p);csv('13_normal_audit.csv',pd.read_csv(p));q=main[(main['mode']=='source_only')&(main.policy=='square')&main.method.isin(['GainGLR','GainSpecific'])];csv('13_wrong_attribution.csv',q.groupby('method').agg(n=('no_alarm','size'),wrong_attribution=('no_alarm',lambda v:float((1-v).mean()))).reset_index());p=W/'C7C1/results/summary_main.json';record(p);(D/'C7C1_summary_main.json').write_bytes(p.read_bytes())
train_rows=[]
for seed in [101,202,303]:
 p=W/f'L1/results/seed{seed}/training_curve.json';record(p)
 for step,loss in json.loads(p.read_text()):train_rows.append(dict(seed=seed,step=step,validation_X_MSE=loss))
csv('03_L1_learning_curves.csv',pd.DataFrame(train_rows))
(R/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(example_selection='fixed original first test trajectory, seed101 for L1/L2; C7C1 medium noise tau.5 cohort0 trajectory0; no outcome selection',sources=sources,commits=dict(L1='a033238a17ccab47c417ea2f03e6da4f5bf0f9e6',L2A='d7ae4b8d1087e7d2be7ad1df7a92d54e50d46d35',L23='318d06ec382ada91d24ce659802257fbf0e0231b',L3D='10e9c2c7f60220ba8156360d8e1aeacb1f4368c2',C7C1='852dd43500a98d6808956d712d40f1fc071e3d53')),indent=2));print('Extracted frozen plot data',len(sources),'source hashes')

"""Frozen paired contrasts and trajectory/seed analysis; no model changes."""
import json,numpy as np,pandas as pd
from scipy.stats import t as td,ttest_1samp,norm
from settings import *
from confirm import holm
CONTRASTS={'update_direct':{'A':1,'B':-1},'update_recursive':{'C':1,'D':-1},'recursive_no_update':{'A':1,'C':-1},'recursive_update':{'B':1,'D':-1},'interaction':{'C':1,'D':-1,'A':-1,'B':1},'structure_direct':{'E':1,'B':-1},'structure_recursive':{'F':1,'D':-1},'vs_strong_direct':{'G':1,'D':-1}}

def interval(x,level=.95):
 x=np.asarray(x);m=float(x.mean());r=float(td.ppf((1+level)/2,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)));return m,m-r,m+r

def bootstrap(a,seed):
 rng=np.random.default_rng(seed);C,S,N=a.shape;out=[]
 for step in range(40):
  ix=rng.integers(0,S,(250,S));v=np.zeros((250,S))
  for c in range(C):
   j=rng.integers(0,N,(250,S,N));v+=a[c][ix[:,:,None],j].mean(-1)/C
  out.extend(v.mean(1))
 return np.quantile(out,[.025,.975])

def main():
 summary=[];overall=[];condition=[];seedrows=[];counts=[]
 for batch in ['A','B']:
  df=pd.read_csv(OUT/f'test_{batch}.csv.gz');df['condition']=df.case.str.rsplit('_',n=1).str[0];summary.append(df.groupby(['batch','condition','n','group']).mean(numeric_only=True).reset_index());p=df.pivot(index=['condition','seed','trajectory','n'],columns='group',values='NLL');counts.append(len(df))
  for n in SIZES:
   q=p.xs(n,level='n')
   for label,co in CONTRASTS.items():
    delta=sum(q[g]*v for g,v in co.items()).rename('gain').reset_index();groups=np.stack([z.pivot(index='seed',columns='trajectory',values='gain').loc[SEEDS].to_numpy() for c,z in delta.groupby('condition')]);sm=groups.mean((0,2));mu,lo,hi=interval(sm);l90,h90=interval(sm,.9)[1:];bl,bh=bootstrap(groups,330000+n+ord(batch)+list(CONTRASTS).index(label))
    overall.append(dict(batch=batch,n=n,contrast=label,gain=mu,t_low95=lo,t_high95=hi,bootstrap_low95=bl,bootstrap_high95=bh,low90=l90,high90=h90,equivalent_90=l90>-.005 and h90<.005,p=float(ttest_1samp(sm,.005,alternative='greater').pvalue)))
    for cond,z in delta.groupby('condition'):
     s=z.groupby('seed').gain.mean();mu,lo,hi=interval(s);condition.append(dict(batch=batch,n=n,condition=cond,contrast=label,gain=mu,low95=lo,high95=hi,p=float(ttest_1samp(s,.005,alternative='greater').pvalue)));seedrows.extend(dict(batch=batch,n=n,condition=cond,contrast=label,seed=int(seed),gain=float(v)) for seed,v in s.items())
 pd.concat(summary).to_csv(OUT/'summary.csv',index=False);o=pd.DataFrame(overall);c=pd.DataFrame(condition)
 for batch in ['A','B']:
  at=(o.batch==batch)&(o.n==100);o.loc[at,'p_holm8']=holm(o.loc[at,'p']);at=(c.batch==batch)&(c.n==100);c.loc[at,'p_holm136']=holm(c.loc[at,'p'])
 o['practical_success']=(o.bootstrap_low95>.005)&(o.p_holm8<.05);o.to_csv(OUT/'contrasts_overall.csv',index=False);c.to_csv(OUT/'contrasts_condition.csv',index=False);pd.DataFrame(seedrows).to_csv(OUT/'contrasts_seed.csv',index=False)
 cf=pd.read_csv(OUT/'confirmation.csv');cf['condition']=cf.case.str.rsplit('_',n=1).str[0];supported=cf.groupby(['condition','n','route','K']).supported.agg(['sum','count']).reset_index();supported['both_lines_supported']=supported['sum']==supported['count'];supported.to_csv(OUT/'confirmation_support.csv',index=False)
 timing=[];graphrows=[];proposal=[]
 for j in json.loads((ROOT/'jobs.json').read_text()):
  graphs=json.loads((OUT/j['case']/'graph.json').read_text())
  for n in SIZES:
   from ssm import unpack
   m=unpack(json.loads((OUT/j['case']/f'model_{n}.json').read_text()));timing.append(dict(case=j['case'],seed=j['seed'],n=n,cpu=m['cpu'],peak_RSS_kib=m['peak_RSS_kib']))
   for route in ['direct','recursive']:
    graphrows.append(dict(case=j['case'],n=n,route=route,initial_K=m['rank_'+route],allowed=','.join(map(str,graphs[str(n)][route]['allowed']))))
   for item in m['recursive']['proposals']:proposal.append(dict(case=j['case'],n=n,**item))
 pd.DataFrame(timing).to_csv(OUT/'timing.csv',index=False);pd.DataFrame(graphrows).to_csv(OUT/'graphs_summary.csv',index=False);(OUT/'proposal_feedback.json').write_text(json.dumps(proposal,indent=2));primary=o[o.n==100];required=primary[primary.contrast.isin(['interaction','recursive_update','vs_strong_direct'])];con=dict(row_counts=counts,primary_n=100,unique_synergy_demonstrated=bool(required.practical_success.all()),primary_contrasts=primary.to_dict('records'),scope='finite Gaussian candidate graph; nonlinear and drift are misspecification tests; no physical causal identification')
 (OUT/'CONCLUSION.json').write_text(json.dumps(con,indent=2));print(primary.to_string(index=False));print('SYNERGY',con['unique_synergy_demonstrated'])
if __name__=='__main__':main()

import json,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from settings import *
from ssm import unpack,component
from producer import simulate
font_manager.fontManager.addfont(str(ROOT/'fonts/NotoSansCJKsc-Regular.otf'));plt.rcParams.update({'font.family':'Noto Sans CJK SC','axes.unicode_minus':False,'font.size':10,'savefig.facecolor':'white'})
F=OUT/'figures';F.mkdir(exist_ok=True);S=pd.read_csv(OUT/'summary.csv');O=pd.read_csv(OUT/'contrasts_overall.csv');C=pd.read_csv(OUT/'contrasts_condition.csv');Q=pd.read_csv(OUT/'confirmation.csv');groups={'A':'直接·不更新','B':'直接·在线更新','C':'递进·不更新','D':'递进·在线更新','E':'直接·固定结构','F':'递进·固定结构','G':'滚动直接强参照'};colors={'A':'#aaa','B':'#7955a0','C':'#d49b55','D':'#16846c','E':'#7891b3','F':'#d76e58','G':'#263f50'};contrast_names={'update_direct':'直接路线更新收益','update_recursive':'递进路线更新收益','recursive_no_update':'无更新时递进收益','recursive_update':'有更新时递进收益','interaction':'递进×更新额外交互','structure_direct':'直接结构反馈收益','structure_recursive':'递进结构反馈收益','vs_strong_direct':'相对滚动直接收益'};baseconds=['S0','S1','S2_weak','S2_medium','S2_strong','S3_weak','S3_medium','S3_strong','S4_weak','S4_medium','S4_strong'];stress=['O_persistence','O_noise','O_delay','O_correlation','O_nonlinear','O_drift'];conds=baseconds+stress;labels=['无隐藏','单隐藏','串联·弱','串联·中','串联·强','并联·弱','并联·中','并联·强','共同驱动·弱','共同驱动·中','共同驱动·强','持续性降低','观测噪声增大','增加两步延迟','创新相关增强','非线性观测','作用强度渐变'];label=dict(zip(conds,labels));titles={}
def save(fig,n,title):
 fig.suptitle(title,y=1.01,fontsize=15);fig.tight_layout();fig.savefig(F/f'{n:02}.png',dpi=170,bbox_inches='tight');plt.close(fig);titles[n]=title
# 1 separate factor effects.
fig,ax=plt.subplots(figsize=(13,5.5));cn=list(contrast_names)
for i,b in enumerate(['A','B']):
 z=O[(O.n==100)&(O.batch==b)].set_index('contrast').loc[cn];x=np.arange(8)+(i-.5)*.15;ax.errorbar(x,z.gain,yerr=[z.gain-z.bootstrap_low95,z.bootstrap_high95-z.gain],fmt='o',capsize=3,label='独立测试批'+b)
ax.axhline(0,color='gray');ax.axhline(.005,color='red',ls=':',label='实用门槛＋0.005');ax.set(xticks=range(8),xticklabels=[contrast_names[k] for k in cn],ylabel='配对联合NLL改善 / 步（正值支持相应机制）');ax.tick_params(axis='x',rotation=25);ax.legend(fontsize=9);save(fig,1,'主要结论：更新、递进、结构反馈与协同必须分别检验')
# 2 condition detail.
fig,axes=plt.subplots(1,2,figsize=(13,9),sharey=True)
for a,contrast in zip(axes,['recursive_update','vs_strong_direct']):
 for bi,batch in enumerate(['A','B']):
  z=C[(C.n==100)&(C.batch==batch)&(C.contrast==contrast)].set_index('condition').loc[conds];yy=np.arange(17)+(bi-.5)*.15;a.errorbar(z.gain,yy,xerr=[z.gain-z.low95,z.high95-z.gain],fmt='o',capsize=2,label='测试批'+batch)
 a.axvline(0,color='gray');a.axvline(.005,color='red',ls=':');a.set(yticks=range(17),yticklabels=labels,xlabel='NLL改善 / 步',title=contrast_names[contrast]);a.invert_yaxis();a.legend(fontsize=8)
save(fig,2,'全条件保留：局部收益不能代替可复现的总体协同')
#3 training curves.
fig,axes=plt.subplots(1,2,figsize=(13,5));g=S[S.batch=='A'].groupby(['n','group']).NLL.mean().unstack()
for group in ['B','D','E','F']:
 axes[0].plot(SIZES,g.G-g[group],'o-',color=colors[group],label=groups[group])
axes[0].axhline(0,color='gray');axes[0].axhline(.005,color='red',ls=':');axes[0].set(xlabel='每条独立训练路线的轨迹数',ylabel='相对滚动直接参照的NLL改善',title='有更新模型的学习曲线');axes[0].legend(fontsize=8)
for contrast in ['update_direct','update_recursive','interaction']:
 z=O[(O.batch=='A')&(O.contrast==contrast)].sort_values('n');axes[1].plot(z.n,z.gain,'o-',label=contrast_names[contrast])
axes[1].axhline(0,color='gray');axes[1].set(xlabel='每路线训练轨迹数',ylabel='NLL改善 / 步',title='小样本与交互效应');axes[1].legend(fontsize=8);save(fig,3,'20至400条轨迹：小样本收益与架构收益分开看')
#4 horizons/calibration.
fig,ax=plt.subplots(1,2,figsize=(13,5));g=S[(S.n==100)&(S.batch=='A')].groupby('group').mean(numeric_only=True)
for group in ['A','B','C','D','E','F']:
 ax[0].plot(HORIZONS,[g.loc['G',f'NLL_h{h}']-g.loc[group,f'NLL_h{h}'] for h in HORIZONS],'o-',color=colors[group],label=groups[group])
ax[0].axhline(0,color='gray');ax[0].set(xlabel='预测时距/联合向量长度',ylabel='相对滚动直接的联合NLL改善 / 步');ax[0].legend(fontsize=7)
ax[1].bar(range(7),100*g.loc[GROUPS,'cover90'],color=[colors[k] for k in GROUPS]);ax[1].axhline(90,color='black',ls=':');ax[1].set(xticks=range(7),xticklabels=[groups[k] for k in GROUPS],ylabel='90%边际区间实际覆盖（%）');ax[1].tick_params(axis='x',rotation=30);save(fig,4,'时距与概率校准：一步改善不等于长期机制信息增加')
#5 fixed probability-state trace.
case='S3_strong_3011';e=np.load(OUT/case/'example_A.npz');ids=e['D_ids'];w=e['D_weights'];mass=np.stack([w[:,ids[:,1]==k].sum(1) for k in range(4)]);fig,ax=plt.subplots(3,1,figsize=(13,9),sharex=True);ax[0].plot(e['X'],color='black');ax[0].set_ylabel('唯一观测X');ax[1].stackplot(range(T),mass,labels=['零状态解释','一状态解释','两状态解释','三状态解释'],alpha=.8);ax[1].set_ylabel('候选维数后验概率');ax[1].legend(loc='upper right',ncol=2,fontsize=8);chosen=np.flatnonzero(ids[:,1]>=2);chosen=int(chosen[0]) if len(chosen) else next((i for i,q in enumerate(ids) if q[1]>0),-1)
if chosen>=0:
 mu=e[f'D_state_mean_{chosen}'];V=e[f'D_state_cov_{chosen}'];
 for k in range(min(2,mu.shape[1])):
  sd=np.sqrt(V[:,k,k]);ax[2].plot(mu[:,k],label=f'候选坐标{k+1}均值');ax[2].fill_between(range(T),mu[:,k]-1.644854*sd,mu[:,k]+1.644854*sd,alpha=.2)
 ax[2].legend(fontsize=8)
ax[2].set(xlabel='离散时间步',ylabel='模型内后验均值与90%区间');save(fig,5,'固定首条轨迹：保留的是完整概率状态，不是两个确定数字')
#6 posterior count proxy.
op=pd.read_csv(OUT/'operational_states.csv.gz');fig,ax=plt.subplots(1,2,figsize=(13,5));q=op[op.batch=='A'].copy();q['condition']=q.case.str.rsplit('_',n=1).str[0]
for group in ['B','D','E','F']:
 z=q[q.group==group].groupby('condition').expected_K.mean().reindex(conds);ax[0].plot(range(17),z,'o-',label=groups[group],color=colors[group])
ax[0].set(xticks=range(17),xticklabels=labels,ylabel='平均候选维数后验期望');ax[0].tick_params(axis='x',rotation=75);ax[0].legend(fontsize=8)
for gi,group in enumerate(['B','D','E','F']):
 z=q[(q.group==group)&q.condition.isin(['S0','S1'])].groupby('condition').confident_overextended_fraction.mean().reindex(['S0','S1']);ax[1].bar(np.arange(2)+(gi-1.5)*.18,100*z,.18,label=groups[group],color=colors[group])
ax[1].set(xticks=[0,1],xticklabels=['无隐藏对照','单隐藏对照'],ylabel='高置信维数过扩展的时间比例（%）');ax[1].legend(fontsize=8);save(fig,6,'确认新增与在线维数概率不是一回事：过扩展和不确定性都保留')
#7 post-freeze truth readout.
R=pd.read_csv(OUT/'state_readout.csv.gz');z=R.groupby('group').mean(numeric_only=True).loc[list('ABCDEF')];internal=pd.read_csv(OUT/'internal_calibration.csv');fig,ax=plt.subplots(1,3,figsize=(15,4.8));gg=list('ABCDEF');ax[0].bar(range(6),z.readout_gain,color=[colors[k] for k in gg]);ax[0].axhline(0,color='gray');ax[0].set(ylabel='校准读出相对无信息分布的NLL收益/维');ax[1].bar(range(6),z.marginal_cover90*100,color=[colors[k] for k in gg]);ax[1].axhline(90,color='black',ls=':');ax[1].set(ylabel='真状态读出90%边际覆盖（%）')
for a in ax[:2]:a.set(xticks=range(6),xticklabels=[groups[k] for k in gg]);a.tick_params(axis='x',rotation=35)
q=internal.groupby('bank')[['marginal_cover90','joint_cover90']].mean();q.plot.bar(ax=ax[2],legend=True);ax[2].axhline(.9,color='black',ls=':');ax[2].set(xticklabels=['直接主线','递进主线','独立见证'],ylabel='已拟合模型自模拟的覆盖率',xlabel='拟合候选库');ax[2].legend(['边际90%覆盖','联合90%覆盖'],fontsize=8);ax[2].tick_params(axis='x',rotation=20);save(fig,7,'后验质量须分层：独立真值读出与模型内校准不能混为真实因果后验')
#8 observationally identical explanations, recompute actual check.
j=next(j for j in json.loads((ROOT/'jobs.json').read_text()) if j['case']=='S2_medium_3011');m=unpack(json.loads((OUT/j['case']/'model_100.json').read_text()))['direct']['models'][2];import copy
rot=np.array([[1,1],[-1,1]])/np.sqrt(2);m2=copy.deepcopy(m);m2['F']=rot@m['F']@rot.T;m2['Q']=rot@m['Q']@rot.T;m2['P']=rot@m['P']@rot.T;m2['C']=m['C']@rot.T;x=simulate(j['index'],j['seed'],1,40);aa=component(x,m);bb=component(x,m2);from predict import weights
ww=weights(np.stack([aa['post']['ll'],bb['post']['ll']],-1),np.array([.5,.5]),True);fig,ax=plt.subplots(1,2,figsize=(12,4.7));ax[0].plot(ww[0,:,0],label='解释一：原始坐标');ax[0].plot(ww[0,:,1],'--',label='解释二：旋转后的隐藏图');ax[0].set(xlabel='新X逐点到达',ylabel='解释概率',ylim=(0,1));ax[0].legend(fontsize=8);ax[1].plot(range(1,21),aa['mean'][0,4],label='解释一预测');ax[1].plot(range(1,21),bb['mean'][0,4],'--',label='解释二预测');ax[1].set(xlabel='未来时距',ylabel='同一X下的预测均值');ax[1].legend(fontsize=8);save(fig,8,'等价性反例：相同观测似然不能支持唯一隐藏因果图')
#9 single-factor stresses.
fig,ax=plt.subplots(figsize=(13,5));keys=['S3_medium']+stress
for i,contrast in enumerate(['update_direct','update_recursive','recursive_update','structure_recursive']):
 z=C[(C.n==100)&(C.batch=='A')&(C.contrast==contrast)].set_index('condition').loc[keys];ax.plot(range(7),z.gain,'o-',label=contrast_names[contrast])
ax.axhline(0,color='gray');ax.set(xticks=range(7),xticklabels=[label[k] for k in keys],ylabel='相应机制的NLL收益 / 步');ax.tick_params(axis='x',rotation=20);ax.legend(fontsize=8);save(fig,9,'单因素压力测试：非线性、延迟和渐变保留为失配条件')
#10 independent-line confirmation scatter.
q=Q[(Q.n==100)&(Q.route=='recursive')&(Q.K==2)];p=q.pivot(index='case',columns='line',values='gain');fig,ax=plt.subplots(figsize=(7,6));ax.scatter(p.recursive,p.witness,s=18,alpha=.55);ax.axhline(.005,color='red',ls=':');ax.axvline(.005,color='red',ls=':');ax.axhline(0,color='gray',lw=.6);ax.axvline(0,color='gray',lw=.6);ax.set(xlabel='递进主线：第二方向确认NLL增量',ylabel='独立原始X见证：第二方向确认NLL增量');save(fig,10,'独立见证检查观测预测信息，不以坐标相关替代新增证据')
(OUT/'FIGURE_TITLES.json').write_text(json.dumps(titles,ensure_ascii=False,indent=2));print('10 Chinese figures rendered')

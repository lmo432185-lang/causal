import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import norm
from settings import *
from model import load
font_manager.fontManager.addfont(str(ROOT/'fonts/NotoSansCJKsc-Regular.otf'));plt.rcParams.update({'font.family':'Noto Sans CJK SC','axes.unicode_minus':False,'font.size':10,'savefig.facecolor':'white'})
F=OUT/'figures';F.mkdir(exist_ok=True);S=pd.read_csv(OUT/'summary.csv');G=pd.read_csv(OUT/'overall_comparisons.csv');C=pd.read_csv(OUT/'condition_comparisons.csv');E=pd.read_csv(OUT/'expansions.csv');D=pd.read_csv(OUT/'data_efficiency.csv');T=pd.read_csv(OUT/'timing.csv');systems=json.loads((ROOT/'SYSTEMS.json').read_text());conds=[x['id'] for x in systems];labs=['无隐藏','单隐藏','串联·弱','串联·中','串联·强','并联·弱','并联·中','并联·强','共同驱动·弱','共同驱动·中','共同驱动·强'];names={'recursive':'递进提取','direct_full':'完整直接提取','direct_fixed':'固定频率字典直接','direct_random':'固定随机字典直接','W1':'单个W1','full_W1':'完整W1历史','full_X':'完整X历史'};colors={'recursive':'#16846c','direct_full':'#7a55a2','direct_fixed':'#cc8f32','direct_random':'#528db9','W1':'#999999','full_W1':'#d66e52','full_X':'#223f50'};titles={}
def save(fig,k,title):
 fig.suptitle(title,y=1.01,fontsize=15);fig.tight_layout();fig.savefig(F/f'{k:02}.png',dpi=170,bbox_inches='tight');plt.close(fig);titles[k]=title
fig,ax=plt.subplots(figsize=(12,4.7));xx=np.arange(11);ax.bar(xx,[m['oracle_information'][1] for m in systems],color=['#aaa']*2+['#b8cfcd','#5fa49c','#16846c']*3);ax.axhline(.005,color='red',ls=':',label='预设实用门槛0.005');ax.set(xticks=xx,xticklabels=labs,ylabel='第二预测方向的理论增益（nats/步）');ax.tick_params(axis='x',rotation=25);ax.legend();save(fig,1,'事先确定难度：11个条件、每条件10个独立训练种子')
fig,ax=plt.subplots(figsize=(10,5));ms=['direct_full','direct_fixed','direct_random']
for bi,b in enumerate(['A','B']):
 z=G[(G.n==400)&(G.batch==b)].set_index('comparison').loc[ms];x=np.arange(3)+(bi-.5)*.16;ax.errorbar(x,z.gain,yerr=[z.gain-z.bootstrap_low95,z.bootstrap_high95-z.gain],fmt='o',capsize=5,label='独立测试批'+b,color=['#16846c','#cc8f32'][bi])
ax.axhline(0,color='gray');ax.axhline(.005,color='red',ls=':',label='实用胜出线＋0.005');ax.axhline(-.005,color='red',ls=':');ax.set(xticks=range(3),xticklabels=[names[m] for m in ms],ylabel='直接NLL－递进NLL / 步（正值递进更好）');ax.legend(fontsize=9);save(fig,2,'主要公平比较：总体配对收益及分层95%区间，双测试批复现')
fig,ax=plt.subplots(figsize=(12,5))
for bi,b in enumerate(['A','B']):
 z=C[(C.n==400)&(C.batch==b)&(C.comparison=='direct_full')].set_index('condition').loc[conds];ax.errorbar(xx+(bi-.5)*.18,z.gain,yerr=[z.gain-z.low95,z.high95-z.gain],fmt='o',capsize=3,label='测试批'+b)
ax.axhline(0,color='gray');ax.axhline(.005,color='red',ls=':');ax.axhline(-.005,color='red',ls=':');ax.set(xticks=xx,xticklabels=labs,ylabel='完整直接NLL－递进NLL / 步');ax.tick_params(axis='x',rotation=25);ax.legend();save(fig,3,'逐条件结果：不以总体平均掩盖弱、中、强条件差异')
fig,axes=plt.subplots(3,3,figsize=(13,10),sharex=True)
for i,a in enumerate(axes.ravel()):
 cond=conds[i+2];z=D[(D.batch=='A')&(D.condition==cond)]
 for m in ['recursive','direct_full','direct_fixed','direct_random']:
  v=z[z.method==m].sort_values('n');a.errorbar(v.n,v.gain_vs_direct400,yerr=[v.gain_vs_direct400-v.low90,v.high90-v.gain_vs_direct400],fmt='o-',label=names[m],color=colors[m],capsize=2)
 a.axhline(0,color='gray',lw=.6);a.axhspan(-.005,.005,color='#d9efdf',alpha=.5);a.set(title=labs[i+2],xticks=SIZES,xlabel='训练轨迹数',ylabel='相对直接400条的NLL收益')
axes[0,0].legend(fontsize=7);save(fig,4,'数据效率曲线：相同测试数据、嵌套训练集，绿带是预设等效范围')
fig,axes=plt.subplots(1,2,figsize=(12,5))
for ai,condition in enumerate(['S0','S1']):
 z=E[(E.n==400)&(E.condition==condition)&(E.test=='second')].set_index('method').loc[METHODS];axes[ai].bar(range(4),100*z['sum']/z['count'],color=[colors[m] for m in METHODS]);axes[ai].errorbar(range(4),100*z['sum']/z['count'],yerr=[100*(z['sum']/z['count']-z.low95),100*(z.high95-z['sum']/z['count'])],fmt='none',color='black',capsize=4);axes[ai].set(xticks=range(4),xticklabels=[names[m] for m in METHODS],ylabel='被确认的额外方向比例（%）',title=labs[ai]+'：10个种子，Wilson 95%区间',ylim=(0,45));axes[ai].tick_params(axis='x',rotation=20)
save(fig,5,'错误扩展：零次观察到错误，也不等于真实错误率为零')
fig,ax=plt.subplots(1,2,figsize=(12,5));z=T[T.n==400].groupby('method').cpu.agg(['mean','min','max']).loc[METHODS];ax[0].bar(range(4),z['mean'],color=[colors[m] for m in METHODS]);ax[0].errorbar(range(4),z['mean'],yerr=[z['mean']-z['min'],z['max']-z['mean']],fmt='none',color='black',capsize=3);ax[0].set(xticks=range(4),xticklabels=[names[m] for m in METHODS],ylabel='单进程CPU秒',title='拟合、候选选择及预测头的代码计时');ax[0].tick_params(axis='x',rotation=20);ax[1].bar(range(4),[42,82,44,44],color=[colors[m] for m in METHODS]);ax[1].set(xticks=range(4),xticklabels=[names[m] for m in METHODS],ylabel='名义滤波系数数目',title='预测头相同：各95个自由参数');ax[1].tick_params(axis='x',rotation=20);save(fig,6,'算力与容量透明记录：近似匹配字典参照＋完整直接强参照')
case='S3_strong_1011';ex=np.load(OUT/case/'example_A.npz');m=load(OUT/case/'model_n400.json');origin=120;ix=origin-START;tt=np.arange(origin+1,origin+11);fig,ax=plt.subplots(1,2,figsize=(13,4.8));ax[0].plot(np.arange(85,141),ex['X'][85:141],color='black',lw=1.3,label='真实X');ax[0].axvline(origin,color='gray',ls=':')
for name in ['W1','recursive','direct_full']:
 mu=ex[name+'_mean'][ix];sd=ex[name+'_sd'];ax[0].plot(tt,mu,'o-',color=colors[name],label=names[name]);ax[0].fill_between(tt,mu-norm.ppf(.95)*sd,mu+norm.ppf(.95)*sd,color=colors[name],alpha=.10)
ax[0].set(xlabel='离散时间步',ylabel='X及未来10步90%边际预测区间',title='预先固定第一条测试轨迹，起点120');ax[0].legend(fontsize=8)
for name in ['W1','recursive','direct_full']:
 mu=ex[name+'_mean'];sd=ex[name+'_sd'];z=(ex['future']-mu)/sd;nll=(.5*(np.log(2*np.pi*sd**2)+z*z)).mean(1);smooth=np.convolve(nll,np.ones(25)/25,mode='valid');ax[1].plot(np.arange(START+12,END-12),smooth,color=colors[name],label=names[name])
ax[1].set(xlabel='预测起点（25点移动平均，仅作展示）',ylabel='各时距边际NLL平均',title='曲线接近不等于已识别同一物理状态');ax[1].legend(fontsize=8);save(fig,7,'固定样本预测曲线：只用于展示，不参与统计或模型挑选')
fig,ax=plt.subplots(1,3,figsize=(14,4.7));primary=S[(S.n==400)&(S.batch=='A')].groupby('method').mean(numeric_only=True);methods=['W1','recursive','direct_full','direct_fixed','direct_random','full_X']
for a,metric in zip(ax,['CRPS','MSE','joint_cover90']):
 vals=[primary.loc[m,metric]*(100 if metric=='joint_cover90' else 1) for m in methods];a.bar(range(6),vals,color=[colors[m] for m in methods]);a.set(xticks=range(6),xticklabels=[names[m] for m in methods],ylabel='联合90%覆盖率（%）' if metric=='joint_cover90' else metric);a.tick_params(axis='x',rotation=45)
 if metric=='joint_cover90':a.axhline(90,color='black',ls=':');a.set_ylim(85,95)
save(fig,8,'辅助指标：均值误差、概率评分与校准分别看，不替代主要NLL结论')
(OUT/'FIGURE_TITLES.json').write_text(json.dumps(titles,ensure_ascii=False,indent=2));print('8 Chinese figures rendered')

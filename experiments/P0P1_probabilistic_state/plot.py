from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import norm
from model import spectrum
from settings import ROOT,OUT
font_manager.fontManager.addfont(str(ROOT/'fonts/NotoSansCJKsc-Regular.otf'));plt.rcParams.update({'font.family':'Noto Sans CJK SC','axes.unicode_minus':False,'font.size':10,'figure.titlesize':15,'pdf.fonttype':3,'savefig.facecolor':'white'})
F=OUT/'figures';F.mkdir(exist_ok=True);titles={};c=pd.read_csv(OUT/'case_metrics.csv');o=pd.read_csv(OUT/'observability.csv');case_meta=c[['case','tag','factor','value','seed']].drop_duplicates();colors={'B_d1':'#c25c32','B_L1':'#b088c5','B_boot':'#bfa146','C_current':'#777777','C_filter':'#23866b','C_smoother':'#3569b3','C_prior':'#999999'};names={'B_d1':'未知参数链式模型','B_L1':'冻结L1＋仅X概率头','B_boot':'参数重采样矩近似','C_current':'真实参数：只看当前X','C_filter':'真实参数：实时滤波','C_smoother':'真实参数：未来平滑','C_prior':'真实参数：无观测先验'}
def save(fig,n,title):
 fig.suptitle(title,y=1.01);fig.tight_layout(rect=(0,.045,1,1) if n==1 else None);fig.savefig(F/f'{n:02}.png',dpi=170,bbox_inches='tight');plt.close(fig);titles[str(n)]=title
#1 Five OFAT factors; confidence ranges = three-fit extrema, not independent timepoint CIs.
factors=[('persistence',[.5,.8,.95],.8,'源头持续性a'),('source_gain',[.2,.5,.8],.5,'源头→中间强度'),('output_gain',[.2,.5,.8],.8,'中间→X强度'),('measurement_noise',[.05,.25,.75],.25,'X测量噪声标准差'),('delay',[1,2,3],1,'每条边的延迟步数')]
fig,ax=plt.subplots(2,5,figsize=(16,7),sharey=True)
for j,(factor,levels,base,label) in enumerate(factors):
 for i,state in enumerate(['Y','Z']):
  for m in ['B_d1','B_L1','C_filter','C_smoother']:
   vals=[];lo=[];hi=[]
   for val in levels:
    z=c[(c.tag=='baseline')&(c.method==m)&(c.state==state)] if val==base else c[(c.factor==factor)&np.isclose(c.value,val)&(c.method==m)&(c.state==state)];vals.append(z.corr_pooled.mean());lo.append(z.corr_pooled.min());hi.append(z.corr_pooled.max())
   ax[i,j].plot(levels,vals,'o-',color=colors[m],ms=4,label=names[m]);ax[i,j].fill_between(levels,lo,hi,color=colors[m],alpha=.12)
  ax[i,j].set(title=label+'：'+state,xlabel='预设条件值',ylim=(-.04,1.03));ax[i,j].grid(alpha=.15)
  if j==0:ax[i,j].set_ylabel(state+'与真值的相关性')
h,l=ax[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=4,bbox_to_anchor=(.5,-.01),fontsize=9);save(fig,1,'P0：什么条件有利于恢复隐藏信息？阴影为三个种子的最小—最大范围')
#2 stochastic observability vs marginal/independent directions.
q=o.groupby('tag').mean(numeric_only=True).loc[['baseline','information_rich','information_poor']];fig,ax=plt.subplots(1,2,figsize=(12,4.8));x=np.arange(3)
for j,(key,label,col) in enumerate([('Y_remaining_variance','Y剩余不确定性','#c25c32'),('Z_remaining_variance','Z剩余不确定性','#23866b')]):ax[0].bar(x+(j-.5)*.32,q[key],.32,label=label,color=col)
for j,(key,label,col) in enumerate([('best_combo_remaining_fraction','最容易恢复的组合方向','#3569b3'),('worst_combo_remaining_fraction','最难恢复的独立方向','#b088c5')]):ax[1].bar(x+(j-.5)*.32,q[key],.32,label=label,color=col)
for a in ax:a.set(xticks=x,xticklabels=['基础条件','信息较强','信息较弱'],ylabel='滤波后剩余方差 / 先验方差',ylim=(0,1.1));a.axhline(1,ls=':',color='grey');a.legend(fontsize=9)
ax[0].set_title('即使知道真参数，也有恢复上限');ax[1].set_title('单个变量恢复好，不代表两个独立方向都恢复好');save(fig,2,'P0：确定性可观测秩满秩，仍可能存在很大的随机状态不确定性')
#3 fixed first trajectories, baseline/rich.
fig,ax=plt.subplots(2,2,figsize=(13,7),sharex=True)
for j,(case,title) in enumerate([('case000_101','基础条件'),('case011_101','信息较强条件')]):
 z=np.load(OUT/case/'evaluation_examples.npz');sl=slice(0,120);tt=z['time'][sl]
 for i,state in enumerate(['Y','Z']):
  a=ax[i,j];a.plot(tt,z['truth'][sl,i],color='black',lw=1,label='真实'+state+'（仅评价）')
  for m in ['B_d1','C_filter']:
   mu=z[m+'_mean'][sl,i];sd=z[m+'_sd'][sl,i];a.plot(tt,mu,color=colors[m],label=names[m],lw=1.2);a.fill_between(tt,mu-1.644854*sd,mu+1.644854*sd,color=colors[m],alpha=.15)
  a.set(title=title+'：'+state,ylabel='状态值（单位创新尺度）',xlabel='离散时间步');a.legend(fontsize=7)
save(fig,3,'P1：点估计与90%概率区间——固定首条测试轨迹，非挑选样本')
#4 calibration reliability in baseline/rich.
fig,ax=plt.subplots(2,3,figsize=(13,7))
for i,(tag,title) in enumerate([('baseline','基础条件'),('information_rich','信息较强')]):
 for j,state in enumerate(['Y','Z','joint']):
  a=ax[i,j];a.plot([50,95],[50,95],color='black',ls=':',label='理想校准')
  for m in ['B_d1','B_boot','B_L1','C_filter','C_prior']:
   z=c[(c.tag==tag)&(c.method==m)&(c.state==state)];a.plot([50,80,90,95],[100*z[f'cover{k}'].mean() for k in [50,80,90,95]],'o-',color=colors[m],label=names[m],ms=3)
  a.set(title=title+'：'+('Y/Z联合区域' if state=='joint' else state),xlabel='标称概率（%）',ylabel='真实状态覆盖率（%）',xlim=(48,97),ylim=(0,100));a.grid(alpha=.12)
ax[0,0].legend(fontsize=6.8);save(fig,4,'P1：区间写着90%，真实状态是否真的有约90%落在里面？')
#5 X probability quality versus hidden-state probability quality.
fig,ax=plt.subplots(1,2,figsize=(12,4.8));nom=[50,80,90,95]
for method,col,label in [('A_AR','#777777','未知结构AR预测'),('B_forecast',colors['B_d1'],'未知参数链式预测'),('C_forecast',colors['C_filter'],'真实参数预测参照')]:
 z=c[(c.state=='X_next')&(c.method==method)];ax[0].plot(nom,[100*z[f'cover{k}'].mean() for k in nom],'o-',color=col,label=label)
ax[0].plot(nom,nom,ls=':',color='black');ax[0].set(xlabel='下一步X的标称概率（%）',ylabel='独立测试实测覆盖率（%）',title='X预测概率几乎校准');ax[0].legend(fontsize=8)
x=np.arange(3);labs=['下一步X','隐藏Y','隐藏Z']
for j,(tag,title,col) in enumerate([('baseline','基础条件','#d58d67'),('information_rich','信息较强','#c25c32')]):
 vals=[]
 for st,m in [('X_next','B_forecast'),('Y','B_d1'),('Z','B_d1')]:vals.append(100*c[(c.tag==tag)&(c.state==st)&(c.method==m)].cover90.mean())
 ax[1].bar(x+(j-.5)*.34,vals,.34,label=title,color=col)
ax[1].axhline(90,ls=':',color='black',label='标称90%');ax[1].set(xticks=x,xticklabels=labs,ylabel='真实覆盖率（%）',title='X预测好，不保证隐藏状态概率对',ylim=(0,102));ax[1].legend(fontsize=8);save(fig,5,'P1：预测校准与隐藏状态校准是两个不同的问题')
#6 exact algebraic parameter alias.
audit=json.loads((OUT/'identifiability_controls.json').read_text());r=next(z for z in audit['records'] if z['tag']=='information_rich');a=np.load(OUT/'alias_examples.npz');fig,ax=plt.subplots(2,2,figsize=(12,7));w=np.linspace(0,1,len(spectrum(r['true_canonical_theta'])))
for theta,label,col,ls in [(r['true_canonical_theta'],'真实参数解释','#23866b','-'),(r['fixed_alias_demonstration']['theta'],'另一套严格等价解释','#c25c32','--')]:ax[0,0].semilogy(w[:len(w)//2],spectrum(theta)[:len(w)//2],label=label,color=col,ls=ls)
ax[0,0].set(xlabel='归一化频率',ylabel='X功率谱（对数刻度）',title='两套参数的X分布完全相同');ax[0,0].legend(fontsize=8)
z=c[c.tag=='information_rich'];tags=['Y','Z'];vals=[r['fixed_alias_demonstration'][s+'_mean_RMS_difference'] for s in tags];ax[0,1].bar(tags,vals,color=['#3569b3','#b088c5']);ax[0,1].set(ylabel='两套模型的后验均值均方根差',title='同样X，却给出不同隐藏状态')
for j,s in enumerate(['Y','Z']):
 for key,label,col in [('true','真实参数解释','#23866b'),('alias','严格等价参数解释','#c25c32')]:
  mu=a['information_rich_'+key+'_mean'][:,j];sd=a['information_rich_'+key+'_sd'][:,j];ax[1,j].plot(np.arange(50,170),mu,label=label,color=col);ax[1,j].fill_between(np.arange(50,170),mu-1.644854*sd,mu+1.644854*sd,color=col,alpha=.12)
 ax[1,j].set(title=s+'的后验与90%区间',xlabel='离散时间步',ylabel='状态值（相同单位创新规范）');ax[1,j].legend(fontsize=8)
save(fig,6,'可识别性审计：固定了隐藏创新尺度，仍有不同参数解释相同X的情况')
#7 delay experiment and posterior sensitivity; d unknown never selected by truth.
fig,ax=plt.subplots(1,3,figsize=(15,4.7));z=np.load(OUT/'case010_101/evaluation_examples.npz');tt=z['time'][:120]
for m,s,ls in [('C_filter','Y','-'),('C_filter','Z','--'),('C_smoother','Y','-'),('C_smoother','Z','--')]:
 vals=[]
 for d in [1,2,3]:
  g=c[(c.tag=='baseline')&(c.method==m)&(c.state==s)] if d==1 else c[(c.factor=='delay')&np.isclose(c.value,d)&(c.method==m)&(c.state==s)];vals.append(g.corr_pooled.mean())
 ax[0].plot([1,2,3],vals,'o'+ls,color=colors[m],label=names[m]+'：'+s)
ax[0].set(xlabel='真实每边延迟步数',ylabel='真值相关性',title='真实延迟增大，实时信息更少');ax[0].legend(fontsize=7)
for j,s in enumerate(['Y','Z']):
 a=ax[j+1];a.plot(tt,z['truth'][:120,j],color='black',label='真实状态',lw=1)
 for d,col in [(1,'#c25c32'),(2,'#bfa146'),(3,'#3569b3')]:a.plot(tt,z[f'B_d{d}_mean'][:120,j],color=col,label=f'兼容延迟假设{d}')
 a.set(title='相同X似然下的'+s+'估计',xlabel='离散时间步',ylabel='单位创新尺度');a.legend(fontsize=7)
save(fig,7,'P0/P1：不能用X似然选出真实传递延迟，必须保留状态估计的歧义')
#8 informative calibration vs unconditional coverage.
fig,ax=plt.subplots(1,3,figsize=(14,4.7));methods=['C_prior','B_d1','B_boot','C_filter','C_smoother'];x=np.arange(5)
for a,tag,title in zip(ax,['baseline','information_rich','information_poor'],['基础条件','信息较强','信息较弱']):
 vals=[c[(c.tag==tag)&(c.method==m)&(c.state=='joint')].NLL.mean() for m in methods];a.bar(x,vals,color=[colors[m] for m in methods]);a.set(xticks=x,xticklabels=['无观测先验','未知参数','重采样近似','真实参数滤波','未来平滑'],ylabel='联合负对数密度（越低越好）',title=title);a.tick_params(axis='x',rotation=28)
save(fig,8,'P1：覆盖率要与概率评分一起看，先验也能覆盖90%却未必提供状态信息')
#9 time calibration.
t=pd.read_csv(OUT/'time_calibration.csv');fig,ax=plt.subplots(1,2,figsize=(12,4.6));sc={'Y':'#3569b3','Z':'#c25c32'}
for a,tag,title in zip(ax,['baseline','information_rich'],['基础条件','信息较强']):
 for m,ls in [('B_d1','-'),('C_filter','--')]:
  for state in ['Y','Z']:
   q=t[(t.tag==tag)&(t.method==m)&(t.state==state)].groupby('start').coverage90.mean();a.plot(q.index,100*q.values,'o'+ls,label=names[m]+'：'+state,color=sc[state],ms=4)
 a.axhline(90,color='black',ls=':');a.set(title=title,xlabel='评价时间区间起点',ylabel='90%区间实测覆盖率（%）',ylim=(0,100));a.legend(fontsize=7)
save(fig,9,'P1：隐藏状态区间失准不是只发生在轨迹开头')
#10 Supplementary Oracle map, matched source stationary variance. No learned scores replaced.
mp=pd.read_csv(OUT/'P0_analytical_map.csv');fig,ax=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
for j,d in enumerate([1,2,3]):
 for i,state in enumerate(['Y','Z']):
  q=mp[mp.delay==d].pivot(index='a',columns='observation_std',values=state+'_corr').sort_index();xx=q.columns.to_numpy();yy=q.index.to_numpy();im=ax[i,j].imshow(q.to_numpy(),origin='lower',extent=[xx.min(),xx.max(),yy.min(),yy.max()],aspect='auto',vmin=0,vmax=1,cmap='viridis');cs=ax[i,j].contour(xx,yy,q.to_numpy(),levels=[.2,.4,.6,.8],colors='white',linewidths=.65);ax[i,j].clabel(cs,inline=True,fontsize=7,fmt='%.1f');ax[i,j].set(title=state+f'：每边延迟{d}步',xlabel='X测量噪声标准差')
  if j==0:ax[i,j].set_ylabel('源头持续性a（源头方差恒定）')
fig.subplots_adjust(right=.88,hspace=.36,wspace=.15);bar=fig.colorbar(im,ax=ax.ravel().tolist(),fraction=.025,pad=.03);bar.set_label('理论滤波估计与真值的相关性');fig.suptitle('补充解析边界图：固定源头方差的真实参数滤波上限（非学习器测试）',y=1.01);fig.savefig(F/'10.png',dpi=170,bbox_inches='tight');plt.close(fig);titles['10']='补充解析边界图：固定源头方差的真实参数滤波上限（非学习器测试）'
(OUT/'FIGURE_TITLES.json').write_text(json.dumps(titles,ensure_ascii=False,indent=2));print('Chinese figures rendered:',len(titles))

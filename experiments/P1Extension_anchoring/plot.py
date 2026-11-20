import json
import numpy as np,pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from settings import ROOT,OUT,METHODS
font_manager.fontManager.addfont(str(ROOT/'fonts/NotoSansCJKsc-Regular.otf'));plt.rcParams.update({'font.family':'Noto Sans CJK SC','axes.unicode_minus':False,'font.size':10,'figure.titlesize':15,'pdf.fonttype':3,'savefig.facecolor':'white'})
F=OUT/'figures';F.mkdir(exist_ok=True);C=pd.read_csv(OUT/'case_metrics.csv');ID=pd.read_csv(OUT/'identification.csv');G=pd.read_csv(OUT/'P2_gate_cases.csv');P=pd.read_csv(OUT/'estimated_parameters.csv');titles={}
name={'E0':'仅X','E1':'时间尺度顺序','E2':'单参数锚点','E3':'已知源头输入','E4Y':'仅稀疏Y','E4Z':'仅稀疏Z','E4YZ':'稀疏Y与Z','C_X':'真实参数：只看X','C_XI':'真实参数：X及输入'};col={'E0':'#8c8c8c','E1':'#4178b5','E2':'#ba8d2e','E3':'#a366b1','E4Y':'#e18b55','E4Z':'#b65359','E4YZ':'#18846d','C_X':'#222222','C_XI':'#424242'}
def metric(tag,policy,m,state,key):return C[(C.tag==tag)&(C.policy==policy)&(C.method==m)&(C.state==state)][key].mean()
def save(fig,n,title):
 fig.suptitle(title,y=1.01);fig.tight_layout();fig.savefig(F/f'{n:02}.png',dpi=170,bbox_inches='tight');plt.close(fig);titles[str(n)]=title
#1 pretraining mathematical count: finite known family, not unrestricted alternatives.
theory=json.loads((ROOT/'PRETRAIN_THEORY_AUDIT.json').read_text())['records'];fig,ax=plt.subplots(1,2,figsize=(13,4.7));xx=np.arange(7)
for a,policy,title in zip(ax,['passive','active'],['无外部输入','有外部输入']):
 rows=[next(v for v in theory if v['tag']=='baseline' and v['policy']==policy and v['method']==m) for m in METHODS];v=[r['retained_known_equivalents'] for r in rows];a.bar(xx,v,color=[col[m] for m in METHODS]);a.set(xticks=xx,xticklabels=[name[m] for m in METHODS],ylabel='仍成立的已知等价参数＋延迟解释数',title=title,ylim=(0,20));a.tick_params(axis='x',rotation=28)
 for j,n in enumerate(v):a.text(j,n+.25,str(n),ha='center')
save(fig,1,'训练前数学审计：每种锚定到底排除了什么？（基础条件，有限等价族）')
#2 calibration passive base/rich.
fig,ax=plt.subplots(2,3,figsize=(13,7));levels=[50,80,90,95]
for i,(tag,title) in enumerate([('baseline','基础'),('rich','信息较强')]):
 for j,st in enumerate(['Y','Z','joint']):
  a=ax[i,j];a.plot(levels,levels,'k:',lw=1)
  for m in ['E0','E1','E2','E4YZ','C_X']:
   a.plot(levels,[100*metric(tag,'passive',m,st,'cover'+str(l)) for l in levels],'o-',color=col[m],label=name[m],ms=3)
  a.set(title=title+'：'+('联合区域' if st=='joint' else st),xlabel='标称概率（%）',ylabel='真实覆盖率（%）',ylim=(0,100));a.grid(alpha=.12)
ax[0,0].legend(fontsize=7);save(fig,2,'测试时仅看X：稀疏双状态标定后，隐藏状态概率是否校准？')
#3 state curves first fixed trajectory.
fig,ax=plt.subplots(2,2,figsize=(13,7),sharex=True)
for j,(case,title) in enumerate([('case000_passive_101','基础'),('case001_passive_101','信息较强')]):
 z=np.load(OUT/case/'evaluation_examples.npz');tt=z['time'][:120]
 for i,st in enumerate(['Y','Z']):
  a=ax[i,j];a.plot(tt,z['truth'][:120,i],color='black',lw=1,label='真实'+st+'（仅评价）')
  for m,ls in [('E0','--'),('E4YZ','-'),('C_X',':')]:
   mu=z[m+'_mean'][:120,i];sd=z[m+'_sd'][:120,i];a.plot(tt,mu,color=col[m],ls=ls,label=name[m]);
   if m=='E4YZ':a.fill_between(tt,mu-1.644854*sd,mu+1.644854*sd,color=col[m],alpha=.18,label='双状态标定90%区间')
  a.set(title=title+'：'+st,xlabel='离散时间步',ylabel='状态值（声明的模拟单位）');a.legend(fontsize=7)
save(fig,3,'固定首条测试轨迹：少量训练锚定能改善状态解释，但没有增加测试传感器')
#4 input helps forecasting/time, not necessarily hidden identity.
fig,ax=plt.subplots(1,2,figsize=(12,4.8));ms=['E0','E3','C_X','C_XI'];x=np.arange(4);w=.34
for j,(tag,title) in enumerate([('baseline','基础'),('rich','信息较强')]):
 vals=[metric(tag,'active',m,'X_next','NLL') for m in ms];ax[0].bar(x+(j-.5)*w,vals,w,label=title,color=['#a9bed3','#4178b5'][j])
ax[0].set(xticks=x,xticklabels=[name[m] for m in ms],ylabel='下一步X预测负对数密度（越低越好）',title='已知输入增加预测信息');ax[0].tick_params(axis='x',rotation=22);ax[0].legend(fontsize=8)
for j,(tag,title) in enumerate([('baseline','基础'),('rich','信息较强')]):
 vals=[100*metric(tag,'active',m,'joint','cover90') for m in ms];ax[1].bar(x+(j-.5)*w,vals,w,label=title,color=['#c4b2d0','#a366b1'][j])
ax[1].axhline(90,color='black',ls=':');ax[1].set(xticks=x,xticklabels=[name[m] for m in ms],ylabel='Y/Z联合90%区域实际覆盖率（%）',title='预测收益不等于隐藏身份恢复',ylim=(0,102));ax[1].tick_params(axis='x',rotation=22);ax[1].legend(fontsize=8);save(fig,4,'已知输入组：X预测接近相同信息的Oracle，隐藏联合概率仍可能失准')
#5 delay: forced choice vs statistically unique, valid-order experimental conditions only.
fig,ax=plt.subplots(1,2,figsize=(13,4.8));x=np.arange(7)
for a,policy,title in zip(ax,['passive','active'],['无外部输入','有外部输入']):
 q=ID[(ID.policy==policy)&(ID.tag!='wrong_order_control')].groupby('method');forced=[100*q.get_group(m).forced_delay_correct.mean() for m in METHODS];unique=[100*q.get_group(m).validation_unique_delay_correct.mean() for m in METHODS];a.bar(x-.17,forced,.34,color='#a9bed3',label='强制选择正确');a.bar(x+.17,unique,.34,color='#18846d',label='验证兼容集唯一且正确');a.set(xticks=x,xticklabels=[name[m] for m in METHODS],ylabel='延迟识别比例（%）',title=title,ylim=(0,105));a.tick_params(axis='x',rotation=28);a.legend(fontsize=8)
save(fig,5,'延迟识别：碰巧选择正确，不能当作数据已经排除了其他延迟')
#6 single-node vs dual-node matched scalar measurement budget.
fig,ax=plt.subplots(2,2,figsize=(12,7));ms=['E4Y','E4Z','E4YZ','C_X'];x=np.arange(4)
for j,(tag,title) in enumerate([('baseline','基础'),('rich','信息较强')]):
 for i,(key,ylabel) in enumerate([('cover90','联合90%区域真实覆盖率（%）'),('NLL','联合负对数密度（越低越好）')]):
  vals=[metric(tag,'passive',m,'joint',key)*(100 if key=='cover90' else 1) for m in ms];ax[i,j].bar(x,vals,color=[col[m] for m in ms]);ax[i,j].set(xticks=x,xticklabels=[name[m] for m in ms],ylabel=ylabel,title=title);ax[i,j].tick_params(axis='x',rotation=15)
  if key=='cover90':ax[i,j].axhline(90,color='black',ls=':');ax[i,j].set_ylim(0,102)
save(fig,6,'辅助观测消融：相同24个标量读数预算，测一个状态与测两个状态的差别')
#7 erroneous prior negative control.
fig,ax=plt.subplots(1,3,figsize=(15,4.7));tags=['a','b','c'];x=np.arange(3)
for j,(m,label) in enumerate([('E1','错误顺序约束'),('E4YZ','双状态标定')]):
 q=P[(P.tag=='wrong_order_control')&(P.policy=='passive')&(P.method==m)];ax[0].bar(x+(j-.5)*.34,q[tags].mean(),.34,label=label,color=col[m])
ax[0].plot(x,[.55,.7,.5],'kD',label='真实动力学系数');ax[0].set(xticks=x,xticklabels=['源头a','中间b','下游c'],ylabel='自回归记忆系数',title='真实顺序是b大于a');ax[0].legend(fontsize=7)
ms=['E0','E1','E4YZ','C_X'];xx=np.arange(4)
for a,key,state,label in [(ax[1],'NLL','X_next','下一步X预测负对数密度'),(ax[2],'cover90','joint','联合90%区域真实覆盖率（%）')]:
 vals=[metric('wrong_order_control','passive',m,state,key)*(100 if key=='cover90' else 1) for m in ms];a.bar(xx,vals,color=[col[m] for m in ms]);a.set(xticks=xx,xticklabels=[name[m] for m in ms],ylabel=label);a.tick_params(axis='x',rotation=25)
 if key=='cover90':a.axhline(90,color='black',ls=':');a.set_ylim(0,102)
ax[1].set_title('X预测可以依然正常');ax[2].set_title('覆盖率也不能验证先验身份');save(fig,7,'错误先验反例：先验必须来自独立知识，不能只靠X拟合得好来自证')
#8 preregistered gate.
primary=G[(G.policy=='passive')&G.tag.isin(['baseline','rich'])];fig,ax=plt.subplots(1,2,figsize=(13,5.8));columns=[(tag,seed) for tag in ['baseline','rich'] for seed in [101,202,303]];labels=[('基础' if t=='baseline' else '较强')+'\n'+str(s) for t,s in columns]
for a,key,title in zip(ax,['conditional_probability_pass','candidate_conditional_P2_pass'],['状态概率门槛','概率＋锚定＋延迟门槛']):
 values=np.array([[int(primary[(primary.method==m)&(primary.tag==tag)&(primary.seed==seed)][key].iloc[0]) for tag,seed in columns] for m in METHODS]);a.imshow(values,cmap=matplotlib.colors.ListedColormap(['#f3d1cd','#a4d3c0']),vmin=0,vmax=1);a.set(xticks=range(6),yticks=range(7),xticklabels=labels,yticklabels=[('零输入时的E3' if m=='E3' else name[m]) for m in METHODS],title=title)
 for i in range(7):
  for j in range(6):a.text(j,i,'通过' if values[i,j] else '未过',ha='center',va='center',fontsize=9)
save(fig,8,'预设六次主检验：概率看起来正确，仍需通过机制与延迟的独立检查')
#9 robustness.
fig,ax=plt.subplots(1,2,figsize=(13,4.8));tags=['baseline','rich','delay2','delay3','weak','wrong_order_control'];labs=['基础','较强','延迟2','延迟3','较弱','顺序反例'];x=np.arange(6)
for policy,marker,label,cc in [('passive','o','无输入','#18846d'),('active','s','有输入','#4178b5')]:
 vals=[];lo=[];hi=[]
 for tag in tags:
  z=C[(C.tag==tag)&(C.policy==policy)&(C.method=='E4YZ')&(C.state=='joint')].cover90*100;vals.append(z.mean());lo.append(z.min());hi.append(z.max())
 ax[0].plot(x,vals,marker+'-',label=label,color=cc);ax[0].fill_between(x,lo,hi,color=cc,alpha=.12)
 y=[int(G[(G.tag==tag)&(G.policy==policy)&(G.method=='E4YZ')].candidate_conditional_P2_pass.sum()) for tag in tags];ax[1].bar(x+(-.17 if policy=='passive' else .17),y,.34,label=label,color=cc)
ax[0].axhline(90,color='black',ls=':');ax[0].set(xticks=x,xticklabels=labs,ylabel='联合90%区域实测覆盖（%）',title='双状态标定校准较稳定',ylim=(86,94));ax[0].legend(fontsize=8);ax[1].set(xticks=x,xticklabels=labs,ylabel='三次数据种子中通过次数',title='较弱条件仍有延迟证据不足',ylim=(0,3.4));ax[1].legend(fontsize=8);save(fig,9,'成功范围与边界：弱信息下即使点选择正确，也应保留延迟不确定性')
(OUT/'FIGURE_TITLES.json').write_text(json.dumps(titles,ensure_ascii=False,indent=2));print('Rendered',len(titles),'Chinese figures')

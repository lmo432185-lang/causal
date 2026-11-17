from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from fontTools.ttLib import TTCollection
R=Path(__file__).parent;D=R/'figure_data';O=R/'figures';O.mkdir(exist_ok=True);F=R/'fonts';F.mkdir(exist_ok=True);font=F/'NotoSansCJKsc-Regular.otf'
if not font.exists():
 c=TTCollection('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
 for f in c.fonts:
  if any('Noto Sans CJK SC' in n.toUnicode() for n in f['name'].names if n.nameID in [1,4]):f.save(font);break
font_manager.fontManager.addfont(str(font));plt.rcParams.update({'font.family':'Noto Sans CJK SC','axes.unicode_minus':False,'font.size':10,'axes.titlesize':12,'figure.titlesize':14,'pdf.fonttype':3,'savefig.facecolor':'white'})
C=['#222222','#e87d23','#2474af','#ba3e53','#39936b'];titles={}
def read(n):return pd.read_csv(D/n)
def save(fig,n,title):
 fig.suptitle(title,y=1.01);fig.tight_layout(rect=(0,.055,1,1) if n==5 else None);fig.savefig(O/f'{n:02}.png',dpi=170,bbox_inches='tight');plt.close(fig);titles[str(n)]=title
# 1
q=read('01_L1_hidden_curves.csv');q=q[q.time<180];fig,ax=plt.subplots(2,1,figsize=(11,6),sharex=True)
for a,s,ch in zip(ax,['Y','Z'],['隐藏源头','中间状态']):
 a.plot(q.time,q[s+'_true'],color=C[0],lw=1.15,label='真实'+s+'（仅用于评价）');a.plot(q.time,q[s+'_L'],color=C[1],lw=1.4,label='由潜在表示L读出（验证集对齐）');a.plot(q.time,q[s+'_current'],color=C[2],lw=1,ls='--',alpha=.7,label='仅用当前X重构');a.set_ylabel(ch+s+'的状态值');a.legend(fontsize=8,ncol=3);a.grid(alpha=.15)
ax[-1].set_xlabel('离散时间步（固定首条测试轨迹）');save(fig,1,'阶段一：只看X学到的L，能重构多少真实隐藏信息？')
# 2
q=read('02_L1_coordinates.csv');s=read('02_03_L1_metrics.csv');fig,ax=plt.subplots(1,2,figsize=(12,4));q=q[q.time<180];ax[0].plot(q.time,q.L1,label='潜在坐标1（L1）',color=C[2]);ax[0].plot(q.time,q.L2,label='潜在坐标2（L2）',color=C[1],ls='--');ax[0].set(xlabel='离散时间步',ylabel='按验证集标准化的坐标值',title='两个坐标几乎重合');ax[0].legend(fontsize=9)
methods=['random_L','X_current','L_only','X_history'];names=['随机坐标','当前X','学习表示L','20点历史'];x=np.arange(4)
for j,state in enumerate(['Y','Z']):ax[1].bar(x+(j-.5)*.34,[s[(s.task==state)&(s.method==m)].NMSE.iloc[0] for m in methods],.34,label=state+'重构误差')
ax[1].set(xticks=x,xticklabels=names,ylabel='归一化均方误差（越低越好）',title='超过随机对照，但与当前X相比改善较小');ax[1].legend(fontsize=9);save(fig,2,'阶段一：表示冗余与隐藏状态重构对照')
# 3
s=read('02_03_L1_metrics.csv');q=read('03_L1_learning_curves.csv');fig,ax=plt.subplots(1,2,figsize=(12,4.6));ms=['A1_current','A2_history','A3_history_L','A_random_history','A4_history_trueYZ'];ns=['当前X','历史X','历史＋L','历史＋随机L','历史＋真实Y/Z']
v=[s[(s.task=='X_next')&(s.method==m)].MSE.iloc[0] for m in ms];ax[0].bar(np.arange(5),v,color=[C[2],C[4],C[1],'#999999',C[3]]);ax[0].set(xticks=np.arange(5),xticklabels=ns,ylabel='测试集下一步X预测MSE',title='L不增加完整历史之外的新观测');ax[0].tick_params(axis='x',rotation=22)
for i,a in enumerate(v):ax[0].text(i,a+.013,f'{a:.5f}',ha='center',fontsize=8)
ax[0].axhline(.25,color='#777777',ls=':',label='过程创新方差0.25');ax[0].legend(fontsize=8)
for seed,g in q.groupby('seed'):ax[1].plot(g.step,g.validation_X_MSE,label=f'随机种子{seed}')
ax[1].set(xlabel='训练迭代步',ylabel='验证集标准化X预测MSE',title='预测损失下降不代表真实状态已分离');ax[1].legend(fontsize=8);save(fig,3,'阶段一：预测对照和真实训练曲线')
# 4
q=read('04_L2A_curves.csv');q=q[q.time<180];fig,ax=plt.subplots(2,1,figsize=(11,6),sharex=True)
for a,s in zip(ax,['Y','Z']):
 for key,label,color,style in [('true','真实状态',C[0],'-'),('L','原始L读出',C[2],'--'),('KF','卡尔曼滤波：当前及过去',C[1],'-'),('RTS','卡尔曼平滑：使用未来',C[4],'-')]:a.plot(q.time,q[s+'_'+key],label=label,lw=1.1,color=color,ls=style,alpha=.85)
 a.set_ylabel(s+'的状态值');a.legend(fontsize=8,ncol=2);a.grid(alpha=.15)
ax[-1].set_xlabel('离散时间步（相同首条测试轨迹）');save(fig,4,'阶段二：同一观测下，实时估计与事后平滑的差别')
#5
s=read('05_recoverability_scan.csv').set_index(['cell','state','method']);factors=[('source_gain',[.15,.5,1.],.5,'上游作用强度α'),('middle_gain',[.25,.8,1.4],.8,'中间→X强度β'),('measurement_noise',[0.,.25,.75],0.,'X测量噪声标准差'),('source_speed',[.3,.8,.95],.8,'源头持续性a'),('middle_memory',[.3,.7,.9],.7,'中间记忆系数c')];fig,axs=plt.subplots(2,5,figsize=(16,6.8),sharey='row');ml=[('L1_frozen_OOD','原模型冻结迁移',C[2]),('L1_matched','同条件重训',C[1]),('KF_known','已知参数滤波',C[4]),('RTS_future','使用未来的平滑',C[3])]
for j,(factor,levels,base,label) in enumerate(factors):
 for i,state in enumerate(['Y','Z']):
  a=axs[i,j]
  for m,lab,col in ml:
   vals=[s.loc[('base' if v==base else f'{factor}_{v}',state,m),'NMSE'] for v in levels];a.plot(levels,vals,marker='o',label=lab,color=col,ms=4)
  a.set_title(label+'：'+state);a.set_xlabel('预设参数值');a.grid(alpha=.15)
  if j==0:a.set_ylabel(state+'重构NMSE（越低越好）')
handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=10,bbox_to_anchor=(.5,-.01));fig.subplots_adjust(bottom=.12);save(fig,5,'阶段二：五类动力学条件对隐藏信息可恢复性的影响')
#6
q=read('06_L2A_variance.csv');fig,ax=plt.subplots(2,1,figsize=(10,5.8),sharex=True)
for a,s,floor in zip(ax,['Y','Z'],[.41,.3125]):
 a.plot(q.time,q[s+'_KF'],label='滤波后验方差',color=C[1]);a.plot(q.time,q[s+'_RTS'],label='平滑后验方差（用了未来）',color=C[4]);a.axhline(floor,ls=':',color=C[0],label='最新未传递创新的实时误差下界');a.set_ylabel(s+'剩余不确定性');a.legend(fontsize=8,ncol=3);a.grid(alpha=.15)
ax[-1].set_xlabel('保留轨迹中的时间步');save(fig,6,'阶段二：信息上限不是更换网络就能突破的')
#7
q=read('07_observational_equivalence.csv');fig,ax=plt.subplots(1,2,figsize=(11,4.2))
for j,label,col in [(0,'直接机制',C[2]),(1,'间接机制',C[1])]:
 ax[0].plot(q.lag,q[f'theory{j}'],label=label+'：解析协方差',color=col,ls='-' if j==0 else '--',lw=2);ax[1].plot(q.lag,q[f'empirical{j}'],label=label+'：600条轨迹均值',color=col);ax[1].fill_between(q.lag,q[f'low{j}'],q[f'high{j}'],color=col,alpha=.15)
for a in ax:a.set(xlabel='滞后步数',ylabel='X的自协方差');a.legend(fontsize=8);a.grid(alpha=.15)
ax[0].set_title('理论曲线完全重合');ax[1].set_title('独立模拟数据及逐轨迹不确定性');save(fig,7,'阶段三：不同因果机制可以产生完全相同的X分布')
#8
q=read('08_rotation_gains.csv');r=json.loads((D/'08_rotation_examples.json').read_text());fig,ax=plt.subplots(1,2,figsize=(12,4.5))
for source,target,label in [('L1','X','潜在1→X'),('L2','X','潜在2→X'),('L1','L2','潜在1→潜在2'),('L2','L1','潜在2→潜在1')]:
 z=q[(q.source==source)&(q.target==target)].copy();z['deg']=z.family.str.replace('rank2_rotate_','',regex=False).astype(float)*180/np.pi;z=z.sort_values('deg');ax[0].plot(z.deg,100*z.test_gain,marker='o',label=label)
ax[0].axhline(1,color='grey',ls=':');ax[0].set(xlabel='潜在坐标旋转角度（度）',ylabel='条件预测MSE降低（%）',title='预测关系依赖坐标基');ax[0].legend(fontsize=8,ncol=2);ax[1].plot([v['angle']*180/np.pi for v in r],[v['forecast_max_difference']*1e15 for v in r],marker='o',color=C[4]);ax[1].set(xlabel='潜在坐标旋转角度（度）',ylabel='整体预测最大数值差（×10的负15次方）',title='整体预测几乎完全不变');save(fig,8,'阶段三：固定二维容量诊断——同样预测，不同候选关系图')
#9
q=read('09_L4_training_curves.csv');conf=read('09_L4_system_confusion.csv').set_index('truth').reindex(index=['G1','G2','G3','G4'],columns=['G1','G2','G3','G4'],fill_value=0);fig,ax=plt.subplots(1,2,figsize=(12,4.7));names={'G1':'并联直达模型','G2':'正向串联模型','G3':'混合路径模型','G4':'反向串联模型'}
for j,(k,g) in enumerate(q.groupby('model')):ax[0].plot(g.iteration,g.loglik_per_training_point,label=names[k],ls='--' if k=='G4' else '-',lw=1.5)
ax[0].set(xlabel='广义EM迭代次数',ylabel='每训练观测的对数似然',title='真实结构为串联的固定示例');ax[0].legend(fontsize=8);im=ax[1].imshow(conf,vmin=0,vmax=1,cmap='Blues');labs=['并联直达\nG1','正向串联\nG2','混合路径\nG3','反向串联\nG4'];ax[1].set(xticks=range(4),yticks=range(4),xticklabels=labs,yticklabels=labs,xlabel='冻结验证集的强制选择',ylabel='真实结构',title='基础四结构：12次系统级重复')
for i in range(4):
 for j in range(4):ax[1].text(j,i,f'{int(round(conf.iloc[i,j]*3))}/3',ha='center',va='center',color='white' if conf.iloc[i,j]>.5 else 'black')
fig.colorbar(im,ax=ax[1]);save(fig,9,'阶段三：未知参数竞争结构拟合与独立结构结果')
#10
q=read('10_failure_example.csv');times=read('10_failure_example_times.csv');fig,ax=plt.subplots(2,2,figsize=(12,6.5),sharex=True)
for j,(policy,title,col) in enumerate([('none','无主动输入',C[2]),('square','已知方波主动输入',C[1])]):
 row=times[times.policy==policy].iloc[0];t0=row.T_beta;t1=t0+4
 for i in [0,1]:ax[i,j].axvspan(t0,t1,color='#eeeeee');ax[i,j].axvline(t0,color='black',ls=':',label='真实机制开始变化');ax[i,j].axvline(t1,color='grey',ls='-.',label='路径完全失效')
 ax[0,j].plot(q.time,q[policy+'_beta'],color=C[0]);ax[0,j].set(title=title,ylabel='真实作用系数β（仅评价可见）',ylim=(-.05,1.15));ax[1,j].plot(q.time,q[policy+'_X'],color=col,label='实际观测X')
 for a in [ax[0,j],ax[1,j]]:
  if np.isfinite(row.T_alarm):a.axvline(row.T_alarm,color=C[3],ls='--',label=f'变化检测报警：{row.T_alarm:g}')
  if np.isfinite(row.T_prediction):a.axvline(row.T_prediction,color=C[4],ls='-',lw=1,label=f'预测误差报警：{row.T_prediction:g}')
 ax[1,j].set(xlabel='模型时间单位（不是现实秒数）',ylabel='观测X');ax[1,j].legend(fontsize=7,loc='best')
save(fig,10,'阶段四：固定渐进失效样本的真实变化与两种报警时间')
#11
q=read('11_detection_cdf.csv');fig,ax=plt.subplots(1,2,figsize=(12,4.8));methodnames={('none','GLR'):'无输入＋通用变化检测',('none','GainGLR'):'无输入＋作用强度检测',('square','GLR'):'方波＋通用变化检测',('square','GainGLR'):'方波＋作用强度检测'}
for a,mode,title in zip(ax,['abrupt','gradual'],['突发失效','渐进衰减']):
 for (policy,method),g in q[q['mode']==mode].groupby(['policy','method']):a.step(g.delay,100*g.recall_all,where='post',label=methodnames[policy,method])
 if mode=='gradual':a.axvline(4,color='grey',ls=':',label='真实路径完全消失')
 a.set(xlabel='真实机制开始变化后的时间',ylabel='全部事件的合法累计检出率（%）',title=title,ylim=(0,102));a.legend(fontsize=8);a.grid(alpha=.15)
save(fig,11,'阶段四：主动输入能让检测变快，但比较必须保留早报和漏报')
#12
q=read('12_warning_coverage.csv');fig,ax=plt.subplots(1,2,figsize=(12,4.6))
for a,mode,title in zip(ax,['abrupt','gradual'],['突发失效','渐进衰减']):
 for policy,method in [('none','GainGLR'),('square','GLR'),('square','GainGLR')]:
  z=q[(q['mode']==mode)&(q.policy==policy)&(q.method==method)];a.step(z.lead_required,100*z.coverage_all,where='post',marker='o',ms=3,label=methodnames[policy,method])
 a.set(xlabel='要求领先预测误差报警的时间',ylabel='全部事件的有效提前覆盖率（%）',title=title);a.legend(fontsize=8);a.grid(alpha=.15)
save(fig,12,'阶段四：检测更快，不等于相对预测失效的预警窗口更大')
#13
q=read('13_normal_audit.csv');a=read('13_wrong_attribution.csv');fig,ax=plt.subplots(1,2,figsize=(12,4.8));ordinary=q[q.method!='GainSpecific'];rg=ordinary.groupby('budget').fpr.agg(['min','max']);ax[0].fill_between(rg.index*100,rg['min']*100,rg['max']*100,color='#dddddd',label='全部普通工况的实测范围')
for policy,method in [('none','GainGLR'),('square','GLR'),('square','GainGLR')]:
 z=q[(q.policy==policy)&(q.method==method)].groupby('budget').fpr.mean();ax[0].plot(z.index*100,z*100,marker='o',label=methodnames[policy,method])
ax[0].plot([1,5],[1,5],ls=':',color='black',label='名义与实测相等参考');ax[0].set(xlabel='名义整轨迹误报目标（%）',ylabel='独立正常审计实测误报率（%）',title='同目标不等于每个工况实测相等');ax[0].legend(fontsize=7)
v=[a[a.method==k].wrong_attribution.iloc[0]*100 for k in ['GainGLR','GainSpecific']];ax[1].bar([0,1],v,color=[C[1],C[4]]);ax[1].set(xticks=[0,1],xticklabels=['普通作用强度报警','增加特异性证据后'],ylabel='源变化被声称为β失效的比例（%）',title='β没有失效，只改变隐藏源头')
for i,value in enumerate(v):ax[1].text(i,value+2,f'{value:.2f}%',ha='center')
ax[1].set_ylim(0,102);save(fig,13,'阶段四：正常误报与错误因果归因，是两个不同问题')
(R/'FIGURE_TITLES.json').write_text(json.dumps(titles,ensure_ascii=False,indent=2));print('Rendered',len(titles),'Chinese figures')

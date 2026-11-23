import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from settings import ROOT,OUT,DATA
from core import *
from io_model import load_model
from fit import features_for,initial_feature
font_manager.fontManager.addfont(str(ROOT/'fonts/NotoSansCJKsc-Regular.otf'));plt.rcParams.update({'font.family':'Noto Sans CJK SC','axes.unicode_minus':False,'font.size':10,'figure.titlesize':15,'pdf.fonttype':3,'savefig.facecolor':'white'})
F=OUT/'figures';F.mkdir(exist_ok=True);D=pd.read_csv(OUT/'all_decisions.csv');T=pd.read_csv(OUT/'all_trajectory_metrics.csv.gz');C=pd.read_csv(OUT/'all_test_checks.csv');I=pd.read_csv(OUT/'paired_intervals.csv');O=pd.read_csv(OUT/'oracle_predictive_directions.csv');titles={}
labels=['无隐藏对照','单隐藏普通','串联普通','并联普通','共同驱动普通','强单状态对照','强双状态对照'];colors={'initial':'#888888','recursive':'#16846c','witness':'#4477ad','direct2':'#a469b1','full_X':'#b98d33','full_W1':'#d4764c','random':'#bdbdbd','duplicate':'#dfb4af'};names={'initial':'当前X＋单个W1','recursive':'加入递进候选','witness':'独立X路径候选','direct2':'直接两状态','full_X':'完整X历史','full_W1':'完整W1历史','random':'随机坐标','duplicate':'重复W1'}
def save(fig,n,title):
 fig.suptitle(title,y=1.01);fig.tight_layout();fig.savefig(F/f'{n:02}.png',dpi=170,bbox_inches='tight');plt.close(fig);titles[str(n)]=title
#1 all question criteria, primary/followup separate rows.
cols=['initial_state_supported','Q1','Q2','Q3','accept_new_coordinate'];v=D.groupby('system_index')[cols].sum().reindex(range(7));fig,ax=plt.subplots(figsize=(10,6));im=ax.imshow(v,cmap='YlGnBu',vmin=0,vmax=3);ax.set(xticks=range(5),xticklabels=['初始W1通过','W1自身动态证据','独立X路径确认','新X预测增量','最终保留'],yticks=range(7),yticklabels=labels);ax.axhline(4.5,color='black',lw=2)
for i in range(7):
 for j in range(5):ax.text(j,i,f'{int(v.iloc[i,j])}/3',ha='center',va='center',color='white' if v.iloc[i,j]>1.5 else 'black')
fig.colorbar(im,ax=ax,label='通过的数据种子次数');save(fig,1,'三问分开检验：原15次与独立强信息对照不混为一次成功率')
#2 theoretical max incremental info.
fig,ax=plt.subplots(figsize=(12,4.6));x=np.arange(7)
for j,rank in enumerate([1,2]):
 vals=O[O.predictive_direction==rank].set_index('system_index').reindex(range(7)).ideal_information_gain_nats_per_step.to_numpy();ax.bar(x+(j-.5)*.34,np.maximum(vals,1e-7),.34,label='最优第'+str(rank)+'个方向')
ax.set_yscale('log');ax.axhline(.005,color='red',ls=':',label='预设实用门槛0.005');ax.set(xticks=x,xticklabels=labels,ylabel='未来10步联合信息增益 / 步（对数刻度）',ylim=(5e-8,.1));ax.tick_params(axis='x',rotation=20);ax.legend(fontsize=9);save(fig,2,'解析诊断：普通组第二方向收益很弱，强对照在新数据生成前保证可检测')
#3 fixed strong two-state test trajectory, independently aligned residuals.
m=load_model(OUT/'case06_101/model.json');xx=np.load(DATA/'case06_101/test_X.npy');q=feature_bundle(xx,m['short'],m['long'],m['recursive_encoder'],m['B'],m['match']);b=affine(initial_feature(q));ar=q['A'][:,:,m['candidate_index']:m['candidate_index']+1]-b@m['A_base_regression'];br=(q['B'][:,:,:m['rank_B']]-b@m['B_base_regression'])@m['B_axis'];fig,ax=plt.subplots(2,1,figsize=(12,6),sharex=True);tt=np.arange(40,180);ax[0].plot(tt,q['W1'][0,:140,0],color='#777777',label='原始W1');ax[0].plot(tt,ar[0,:140,0],color='#16846c',label='递进候选的条件残差');ax[0].set_ylabel('候选坐标（非物理变量单位）');ax[0].legend(fontsize=9);ax[1].plot(tt,ar[0,:140,0],color='#16846c',label='从W1历史递进提取');ax[1].plot(tt,br[0,:140,0],'--',color='#4477ad',label='另一批原始X独立提取并匹配');ax[1].set(xlabel='预测起点的离散时间步',ylabel='控制当前X和W1后的残差');ax[1].legend(fontsize=9);save(fig,3,'强双状态对照：固定首条测试轨迹上的新方向与双路径一致性')
#4 forecast gains, capacity controls.
fig,ax=plt.subplots(1,3,figsize=(15,4.8));ms=['initial','direct2','full_X','full_W1','random','duplicate'];x=np.arange(6)
for a,metric,label in zip(ax,['NLL','CRPS','MSE'],['联合负对数密度改善 / 步','平均CRPS改善','多步MSE改善']):
 z=I[(I.system_index==6)&(I.metric==metric)].set_index('comparison').loc[ms];a.bar(x,z.gain,color=['#16846c','#a469b1','#b98d33','#d4764c','#bdbdbd','#dfb4af']);a.errorbar(x,z.gain,yerr=np.array([z.gain-z.low95,z.high95-z.gain]),fmt='none',color='black',capsize=3);a.axhline(0,color='grey');a.set(xticks=x,xticklabels=['单W1','直接2状态','完整X历史','完整W1历史','随机坐标','重复W1'],ylabel=label,title='递进模型相对各对照');a.tick_params(axis='x',rotation=28)
save(fig,4,'未见X上的配对收益：相对单状态明显改善，与直接两状态模型相当')
#5 horizon-specific probability performance.
fig,ax=plt.subplots(1,2,figsize=(12,4.6));g=T[T.system_index==6].groupby('method').mean(numeric_only=True);h=[1,2,5,10];gains=[g.loc['initial',f'NLL_h{k}']-g.loc['recursive',f'NLL_h{k}'] for k in h];ax[0].bar(range(4),gains,color='#16846c');ax[0].set(xticks=range(4),xticklabels=[str(k)+'步' for k in h],ylabel='该时距边际NLL改善',title='主要收益并非一步预测');ax[0].axhline(0,color='grey')
for method in ['initial','recursive','direct2']:ax[1].plot(range(4),[100*g.loc[method,f'cover90_h{k}'] for k in h],'o-',label=names[method],color=colors[method])
ax[1].axhline(90,color='black',ls=':');ax[1].set(xticks=range(4),xticklabels=[str(k)+'步' for k in h],ylabel='90%边际区间实际覆盖（%）',title='概率校准保持接近90%',ylim=(86,94));ax[1].legend(fontsize=8);save(fig,5,'分预测时距看结果：多步改善与一步几乎不变必须区分')
#6 reproducibility.
S=pd.concat([pd.read_csv(OUT/'cross_seed_stability.csv'),pd.read_csv(OUT/'cross_seed_stability_positive.csv')]);fig,ax=plt.subplots(1,2,figsize=(13,4.7));tc=C.copy();tc['system_index']=tc.case.str[4:6].astype(int)
for a,key,source,label in [(ax[0],'partial_corr',tc,'独立路径的条件相关'),(ax[1],'subspace_min_CCA',S,'跨训练种子的最小子空间相关')]:
 q=source.groupby('system_index')[key].agg(['mean','min','max']).reindex(range(7));a.bar(range(7),q['mean'],color=['#b5c5cb']*5+['#d49664','#16846c']);a.errorbar(range(7),q['mean'],yerr=[q['mean']-q['min'],q['max']-q['mean']],fmt='none',color='black',capsize=3);a.set(xticks=range(7),xticklabels=labels,ylabel=label,ylim=(0,1.06));a.tick_params(axis='x',rotation=25)
ax[0].axhline(.5,color='red',ls=':',label='确认门槛0.5');ax[0].legend(fontsize=8);save(fig,6,'复现性：稳定的是强双状态预测子空间，不是任意递进生成的坐标')
#7 predictive rank vs physical hidden-state count.
fig,ax=plt.subplots(figsize=(12,5.5));mat=[]
for i in range(7):
 row=[]
 for seed in [101,202,303]:
  r=load_model(OUT/f'case{i:02}_{seed}/model.json');row.extend([r['rank_initial'],r['rank_B'],r['rank_match']])
 mat.append(row)
a=np.array(mat);im=ax.imshow(a,cmap='Blues',vmin=0,vmax=3);ax.set(xticks=range(9),xticklabels=['初始\n101','独立X\n101','直接全\n101','初始\n202','独立X\n202','直接全\n202','初始\n303','独立X\n303','直接全\n303'],yticks=range(7),yticklabels=labels)
for i in range(7):
 for j in range(9):ax.text(j,i,str(a[i,j]),ha='center',va='center')
fig.colorbar(im,ax=ax,label='自动保留的额外预测方向数');save(fig,7,'维数不是预先告知：选出的预测秩也不等于真实物理隐藏变量个数')
#8 false self-memory in null control.
Q=pd.read_csv(OUT/'confirmation_tests.csv');fig,ax=plt.subplots(1,2,figsize=(12,4.6));nulls=['case00_101','case00_202','case00_303'];x=np.arange(3)
for j,(label,title,col) in enumerate([('Q1_W1_dynamics','改善W1自身未来','#d4764c'),('Q3_vs_initial','改善原始X未来','#16846c')]):
 vals=[Q[(Q.case==c)&(Q.comparison==label)].gain_nats_per_step.iloc[0] for c in nulls];ax[0].bar(x+(j-.5)*.34,vals,.34,label=title,color=col)
ax[0].axhline(.005,color='red',ls=':');ax[0].set(xticks=x,xticklabels=['种子101','种子202','种子303'],ylabel='独立确认集NLL改善 / 步',title='系统无隐藏状态，W1仍可带有滤波记忆');ax[0].legend(fontsize=8)
metrics=T[T.system_index==6].groupby('method').NLL.mean();ms=['initial','recursive','duplicate','random'];ax[1].bar(range(4),[metrics['initial']-metrics[m] for m in ms],color=[colors[m] for m in ms]);ax[1].set(xticks=range(4),xticklabels=['单W1','递进候选','重复W1','随机坐标'],ylabel='未见X联合NLL改善 / 步',title='增加数字维度本身没有收益');ax[1].axhline(0,color='grey');save(fig,8,'必要反例：能预测表示自身的记忆，不能直接认定为新的系统原因')
(OUT/'FIGURE_TITLES.json').write_text(json.dumps(titles,ensure_ascii=False,indent=2));print('Rendered',len(titles),'Chinese figures')

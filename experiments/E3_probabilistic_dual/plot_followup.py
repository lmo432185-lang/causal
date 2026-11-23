import json,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from settings import *
font_manager.fontManager.addfont(str(ROOT/'fonts/NotoSansCJKsc-Regular.otf'));plt.rcParams.update({'font.family':'Noto Sans CJK SC','axes.unicode_minus':False,'font.size':10});F=OUT/'figures';R=pd.read_csv(OUT/'replication_contrasts.csv');C=pd.read_csv(OUT/'crossline_fresh_check.csv');fig,ax=plt.subplots(1,3,figsize=(14,5));comparisons=['recursive_update','interaction','vs_strong_direct'];titles=['递进在线相对直接在线','递进×更新交互收益','相对滚动直接参照']
for a,contrast,title in zip(ax,comparisons,titles):
 rows=[]
 for budget in ['em5','em25']:
  for batch in ['A','B']:
   z=R[(R.analysis==f'new10_{budget}_{batch}')&(R.contrast==contrast)].iloc[0];rows.append(z)
 for i,z in enumerate(rows):a.errorbar(i,z.gain,yerr=[[z.gain-z.bootstrap_low95],[z.bootstrap_high95-z.gain]],fmt='o',color='#16846c' if i<2 else '#7955a0',capsize=4)
 a.axhline(.005,color='red',ls=':');a.axhline(0,color='gray');a.set(xticks=range(4),xticklabels=['5次·A','5次·B','25次·A','25次·B'],ylabel='联合NLL收益 / 步',title=title)
fig.suptitle('新增10个独立训练种子：有限迭代优势复现，延长优化后协同减弱',fontsize=15,y=1.02);fig.tight_layout();fig.savefig(F/'11.png',dpi=170,bbox_inches='tight');plt.close(fig)
C['condition']=C.case.str.rsplit('_',n=1).str[0];z=C.groupby('condition').agg(rate=('secondary_observable_coherence','mean'),corr=('conditional_predictive_effect_abs_corr','mean'),count=('secondary_observable_coherence','count')).reset_index();labs={'S0':'无隐藏','S1':'单隐藏','S2_weak':'串联弱','S2_medium':'串联中','S2_strong':'串联强','S3_weak':'并联弱','S3_medium':'并联中','S3_strong':'并联强','S4_weak':'共同驱动弱','S4_medium':'共同驱动中','S4_strong':'共同驱动强','O_persistence':'低持续性','O_noise':'高噪声','O_delay':'两步延迟','O_correlation':'相关创新','O_nonlinear':'非线性','O_drift':'渐变'};fig,ax=plt.subplots(1,2,figsize=(14,6));xx=np.arange(len(z));ax[0].bar(xx,z.rate*100,color='#16846c');ax[0].set(ylabel='新增独立X中可重复的候选效果比例（%）',ylim=(0,100));ax[1].bar(xx,z['corr'],color='#7955a0');ax[1].axhline(.5,color='red',ls=':');ax[1].set(ylabel='控制旧预测后，新候选效果绝对相关',ylim=(0,1))
for a in ax:a.set(xticks=xx,xticklabels=[labs[c] for c in z.condition]);a.tick_params(axis='x',rotation=75)
fig.suptitle('探索性新X审计：两路线都增加预测收益，不保证找到同一条线索',fontsize=15,y=1.02);fig.tight_layout();fig.savefig(F/'12.png',dpi=170,bbox_inches='tight');plt.close(fig)
print('Follow-up figures rendered; coherent',int(C.secondary_observable_coherence.sum()),'of',len(C))

meta=json.loads((OUT/'FIGURE_TITLES.json').read_text());meta['11']='新增独立种子与优化预算';meta['12']='新增独立X的候选一致性';(OUT/'FIGURE_TITLES.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))

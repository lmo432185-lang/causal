"""Post-test Chinese scientific plots. Does not fit or choose any predictive model."""
import json,gzip,numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager,FontProperties
from scipy.special import ndtr
from settings import *
from producer import generate,SYSTEMS,matrices,arma_alias
from model import unpack
from predict import component
import native
FONT=ROOT/'fonts'/'NotoSansCJKsc-Regular.otf'
if not FONT.exists():FONT=OLD/'fonts'/'NotoSansCJKsc-Regular.otf'
fontManager.addfont(str(FONT));plt.rcParams.update({'font.family':FontProperties(fname=FONT).get_name(),'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120,'savefig.dpi':155})
FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
COL=['#2563eb','#d97706','#059669','#9333ea','#dc2626','#64748b']
LABEL={'R0':'强直接点模型','R1':'后验路径提案','R2':'创新提案','R3':'联合状态点模型','J0':'原规范模型混合','J1':'共享状态点估计','J2':'双线结构／参数积分','A':'仅直接路线','B':'仅递进路线','C':'双线无共享状态','D':'共享状态点估计','E':'双线结构／参数积分','BestSingle':'选择集最强A／B','StrongestSingle':'选择集完整强参照','Xhistory':'原始X历史','J1_param':'固定共享结构／参数积分','J2_point':'多结构／点参数'}
def load(case):
 with gzip.open(OUT/(case+'.json.gz'),'rt') as f:return unpack(json.load(f))
def save(name):
 plt.tight_layout();plt.savefig(FIG/(name+'.png'),bbox_inches='tight');plt.close()
def acf(x,n=30):
 x=x-x.mean();return np.array([np.mean(x[:len(x)-k]*x[k:]) if k else np.mean(x*x) for k in range(n+1)])/np.mean(x*x)
def whisker(ax,z,ys,color,label):
 ax.errorbar(z.gain,ys,xerr=[np.maximum(z.gain-z.lo,0),np.maximum(z.hi-z.gain,0)],fmt='o',color=color,capsize=3,label=label)
def main():
 assert (ROOT/'ANALYSIS_COMPLETE.json').exists() and (ROOT/'AUDIT_COMPLETE.json').exists() and (ROOT/'short/ANALYSIS_A_COMPLETE.json').exists()
 c=pd.read_csv(OUT/'contrasts.csv');cal=pd.read_csv(OUT/'calibration_intervals.csv');conds=[s['id'] for s in SYSTEMS]
 fig,ax=plt.subplots(1,3,figsize=(13,3.8))
 for a,name in zip(ax,['B0','D0','CE']):
  index=next(i for i,m in enumerate(SYSTEMS) if m['id']==name);x=generate(index,4011,11,20)[:1];b=load(name+'_4011');m=b['models']['R_none'];p=native.run(x/m['scale'],m);C=m['C'];mu=np.einsum('i,nti->nt',C,p['pred_mean'])*m['scale'] if 'pred_mean' in p else None
  # Innovation is reconstructed causally from the previous filtered state.
  prior=np.einsum('i,ij,ntj->nt',C,m['F'],p['mean'][:,:-1])*m['scale']+m.get('a',0)*x[:,:-1];e=x[:,1:]-prior
  a.plot(acf(x[0,200:]),label='原始观测X',color=COL[0]);a.plot(acf(e[0,200:]),label='基础模型一步创新',color=COL[1]);a.axhline(0,color='gray',lw=.7);a.set_title(name+'／固定种子4011');a.set_xlabel('滞后步数');a.set_ylabel('自相关');a.legend(fontsize=8)
 fig.suptitle('图1：原始动态与基础模型尚未解释的动态（独立测试A）');save('01_innovation')
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for ax,names,title in zip(axes,[['residual_gain','joint_gain','original_vs_residual','path_vs_residual'],['complement_gain','full_direct_reference','feedback_gain','posterior_vs_point']],[['创新相对强直接','联合相对强直接','创新相对原路径混合','创新相对新增路径'],['双线相对最强A／B','双线相对完整强参照','共享相对无共享','结构参数积分相对点模型']]):
  for j,ba in enumerate(['A','B']):
   z=c.query('batch==@ba and condition=="ALL"').set_index('contrast').loc[names];whisker(ax,z,np.arange(len(names))+(j-.5)*.16,COL[j],'测试'+ba)
  ax.set_yticks(range(len(names)),title);ax.axvline(0,color='gray');ax.set_xlabel('NLL改善／步（正值有利），95%区间');ax.legend();ax.set_title('实用门槛＋0.005（本图为局部放大）',fontsize=9)
 fig.suptitle('图2：预注册机制比较及双线独立增量');save('02_main_gain')
 fig,axes=plt.subplots(1,2,figsize=(13,4))
 for k,(ax,contrast) in enumerate(zip(axes,['complement_gain','full_direct_reference'])):
  for j,ba in enumerate(['A','B']):
   z=c.query('batch==@ba and condition!="ALL" and contrast==@contrast').set_index('condition').reindex(conds);ax.errorbar(np.arange(24)+(j-.5)*.16,z.gain,yerr=[z.gain-z.lo,z.hi-z.gain],fmt='.',capsize=2,label='测试'+ba,color=COL[j])
  ax.axhline(0,color='gray');ax.axhline(.005,color='red',ls='--',label='实用门槛');ax.set_xticks(range(24),conds,rotation=65);ax.set_ylabel('NLL改善／步');ax.set_title('相对选择集最强A／B' if k==0 else '相对选择集完整强参照');ax.legend(fontsize=8)
 fig.suptitle('图3：双线收益是否跨条件稳定（相同测试轨迹配对）');save('03_condition_gain')
 fig,ax=plt.subplots(figsize=(13,4.5));labs={'obs':'仅增加观测记忆','direct_latent':'仅增加动态坐标','direct_both':'记忆＋动态坐标','direct_dense':'同容量完整联合状态'}
 for j,(key,label) in enumerate(labs.items()):
  z=c[(c.batch=='A')&(c.condition!='ALL')&(c.contrast=='capacity_'+key)].set_index('condition').reindex(conds)
  ax.plot(range(24),z.gain,'-o',ms=3,label=label,color=COL[j])
 ax.axhline(0,color='gray');ax.axhline(.005,color='red',ls='--',label='实用门槛');ax.set_xticks(range(24),conds,rotation=65);ax.set_ylabel('相对基础模型的NLL改善／步');ax.set_title('图4：四类容量对照，记忆维数增加不等于发现新原因');ax.legend(ncol=2,fontsize=8);save('04_capacity')
 d=pd.read_csv(OUT/'test_A.csv.gz');jx=pd.read_csv(OUT/'joint_X_probability.csv.gz');fig,axes=plt.subplots(1,3,figsize=(14,4))
 for j,m in enumerate(['J0','J1','J2','A','B']):
  z=d[d.method==m];axes[0].plot(HORIZONS,[z['NLL_h'+str(h)].mean() for h in HORIZONS],'-o',label=LABEL[m],color=COL[j]);axes[1].scatter(z.cover90.mean(),z.width90.mean(),s=55,label=LABEL[m],color=COL[j]);q=jx[jx.method==m];axes[2].plot([.5,.8,.9,.95],[q['joint_cover'+str(k)].mean() for k in [50,80,90,95]],'-o',label=LABEL[m],color=COL[j])
 axes[0].set_xlabel('联合预测步数');axes[0].set_ylabel('联合NLL／步');axes[0].legend(fontsize=7);axes[1].axvline(.9,color='gray',ls='--');axes[1].set_xlabel('边际90%覆盖率');axes[1].set_ylabel('90%区间宽度');axes[2].plot([.5,.95],[.5,.95],'--',color='gray');axes[2].set_xlabel('名义概率');axes[2].set_ylabel('未来20维联合区域覆盖率');fig.suptitle('图5：X的概率预测：评分、宽度与联合覆盖（测试A）');save('05_X_probability')
 sig=pd.read_csv(OUT/'signal_summary.csv');fig,axes=plt.subplots(1,2,figsize=(12,4))
 for j,m in enumerate(['J0','J1','J2','J1_param']):
  z=sig.query('batch=="A" and method==@m').set_index('condition').reindex(conds);axes[0].plot(range(24),z.cover90,'-o',ms=3,label=LABEL[m],color=COL[j]);axes[1].plot(range(24),z.CRPS,'-o',ms=3,label=LABEL[m],color=COL[j])
 axes[0].axhline(.9,color='gray',ls='--');axes[0].set_ylabel('共同作用S的90%区间覆盖率');axes[1].set_ylabel('S的CRPS（越小越好）')
 for a in axes:a.set_xticks(range(24),conds,rotation=65);a.set_xlabel('测试A条件')
 axes[0].legend(fontsize=8);fig.suptitle('图6：共同作用S的评价，不强行给任意潜在坐标贴U／V标签');save('06_signal')
 methods=['INDEP_POINT','PARAM_ONLY','FULLQ_PARAM','BAYES_FULL','ORACLE'];names=['独立创新点估计','参数自助积分','全Q＋参数自助积分','连续参数Bayes＋全Q（采样未达标）','已知真参数审计']
 fig,axes=plt.subplots(1,4,figsize=(15,3.8),sharey=True)
 for ax,co,rho in zip(axes,['CL','CM','CH','CE'],[0,.5,.9,.99]):
  for j,(m,label) in enumerate(zip(methods,names)):
   z=cal.query('batch=="A" and condition==@co and method==@m and coordinate!="joint"');v=z.groupby('metric')[['mean','lo','hi']].mean().reindex(['cover50','cover80','cover90','cover95']);ax.plot([.5,.8,.9,.95],v['mean'],linestyle='--' if j==1 else '-',marker=['o','s','^','D','x'][j],ms=3,label=label,color=COL[j])
  ax.plot([.5,.95],[.5,.95],'--',color='gray');ax.set_title(f'{co}：真实创新相关{rho}');ax.set_xlabel('名义概率');ax.set_ylim(0,1.01)
 axes[0].set_ylabel('U／V平均边际覆盖率');axes[-1].legend(fontsize=7,loc='lower right');fig.suptitle('图7：相关状态概率校准，允许全Q也未必识别真实相关性');save('07_state_calibration')
 fig,axes=plt.subplots(1,3,figsize=(14,4));x=np.arange(4)
 for j,(m,label) in enumerate(zip(methods,names)):
  for ax,metric,coordinate in zip(axes,['width90','CRPS','joint_NLL'],['marg','marg','joint']):
   z=cal[(cal.batch=='A')&(cal.method==m)&((cal.coordinate=='joint') if coordinate=='joint' else (cal.coordinate!='joint'))&(cal.metric==metric)].groupby('condition')['mean'].mean().reindex(['CL','CM','CH','CE']);ax.plot(x,z,'-o',label=label,color=COL[j]);ax.set_xticks(x,['CL','CM','CH','CE'])
 axes[0].set_ylabel('平均90%区间宽度');axes[1].set_ylabel('边际CRPS（越低越好）');axes[2].set_ylabel('二维联合状态NLL（越低越好）');axes[0].legend(fontsize=7);fig.suptitle('图8：覆盖率不能单独看：宽度、边际评分与联合评分');save('08_state_scores')
 z=np.load(OUT/'physical_example_CE_4011.npz');fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
 for k,ax in enumerate(axes):
  ax.plot(z['times'],z['truth'][:,k],color='black',lw=1,label='真实状态（事后答案）')
  for j,(m,label) in enumerate([('FULLQ_PARAM','全Q对称先验'),('POSITIVE_PRIOR','仅正相关先验'),('ORACLE','已知真实模型')]):
   mu=z[m+'_mu'][:,:,k];v=z[m+'_cov'][:,:,k,k];w=z[m+'_w'];mean=(mu*w).sum(1);lo=np.min(mu-8*np.sqrt(v),1);hi=np.max(mu+8*np.sqrt(v),1)
   qs=[]
   for target in [.05,.95]:
    l=lo.copy();h=hi.copy()
    for _ in range(45):mid=(l+h)/2;cdf=(w*ndtr((mid[:,None]-mu)/np.sqrt(np.maximum(v,1e-30)))).sum(1);l=np.where(cdf<target,mid,l);h=np.where(cdf>=target,mid,h)
    qs.append((l+h)/2)
   ax.plot(z['times'],mean,label=label,color=COL[j]);ax.fill_between(z['times'],qs[0],qs[1],alpha=.12,color=COL[j])
  ax.set_ylabel('贡献坐标U' if k==0 else '贡献坐标V');ax.legend(ncol=4,fontsize=8)
 axes[0].set_title('图9：同一X下的先验敏感性（CE／种子4011／首条测试A，非择优示例）');axes[-1].set_xlabel('时间步');save('09_prior_sensitivity')
 fig,axes=plt.subplots(2,3,figsize=(12,6.5))
 for row,name in enumerate(['B1','CE']):
  b=load(name+'_4011');key=b['candidates']['R3']['R']['best'];m=b['models'][key];z=np.load(OUT/(name+'_4011_example_A.npz'));F=m['F'];Q=m['Q'];q=Q/np.sqrt(np.outer(np.diag(Q),np.diag(Q)));P=z['J1_state_0_cov'][350];p=P/np.sqrt(np.maximum(np.outer(np.diag(P),np.diag(P)),1e-30))
  for ax,a,title in zip(axes[row],[F,q,p],['转移矩阵F','创新相关矩阵','条件后验相关矩阵']):
   im=ax.imshow(a,cmap='coolwarm',vmin=-1,vmax=1);ax.set_title(name+'：'+title);ax.set_xticks(range(len(a)),[f'坐标{i+1}' for i in range(len(a))],rotation=30);ax.set_yticks(range(len(a)),[f'坐标{i+1}' for i in range(len(a))]);fig.colorbar(im,ax=ax,shrink=.7)
 fig.suptitle('图10：联合模型允许相关性；这些坐标仍不是已识别的物理原因');save('10_joint_matrices')
 fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True);b=load('B1_4011');z=np.load(OUT/'B1_4011_example_A.npz')
 for ax,m in zip(axes,['A','B','J2']):
  plan=b['plans'][m];w=z[m+'_weights'];groups={}
  for k,key in enumerate(plan['ids']):
   st=int(key.split('_')[-2]);label=['基础记忆','基础＋观测记忆','新增动态结构','完整联合结构'][st];groups[label]=groups.get(label,0)+w[:,k]
  for j,(label,v) in enumerate(groups.items()):ax.plot(z['origins'],v,label=label,color=COL[j])
  ax.set_title(LABEL[m]);ax.set_xlabel('预测发出时点');ax.set_ylim(-.02,1.02);ax.legend(fontsize=7)
 axes[0].set_ylabel('离散候选的在线概率');fig.suptitle('图11：在线模型权重，不等于真实因素数量的概率');save('11_weights')
 support=pd.read_csv(OUT/'support_summary.csv');fig,axes=plt.subplots(1,2,figsize=(13,4))
 for j,m in enumerate(['R1','R2','R3']):
  z=support[support.route==m].set_index('condition').reindex(conds);axes[0].bar(np.arange(24)+(j-1)*.25,z.support,.25,label=LABEL[m],color=COL[j])
 axes[0].set_xticks(range(24),conds,rotation=65);axes[0].set_ylim(0,20);axes[0].set_ylabel('严格独立确认通过种子数／20');axes[0].legend(fontsize=7)
 read=pd.read_csv(OUT/'state_readout.csv.gz');read['family']=read.case.str[0]
 for j,m in enumerate(['Xonly','R0','R2','R3']):
  a=read[read.route==m].groupby('family').R2.mean().reindex(list('ABCDE'));axes[1].bar(np.arange(5)+(j-1.5)*.19,a,.19,label='仅当前X' if m=='Xonly' else LABEL[m],color=COL[j])
 axes[1].set_xticks(range(5),list('ABCDE'));axes[1].set_ylabel('独立真值审计读出R²');axes[1].legend(fontsize=7);fig.suptitle('图12：新增模式确认与可读出信息是两项不同检验');save('12_confirmation_readout')
 name='E1';index=next(i for i,m in enumerate(SYSTEMS) if m['id']==name);x=generate(index,4011,11,20)[:1];b=load('E1_4011');m=b['bases']['R']['canonical_point'];a=arma_alias(m);p=component(x,m);q=component(x,a);fig,axes=plt.subplots(1,3,figsize=(14,4))
 axes[0].plot(np.arange(200,400),x[0,200:400],color=COL[0]);axes[0].set_xlabel('时间步');axes[0].set_ylabel('两种解释共同的X')
 axes[1].plot(range(1,21),p['mean'][0,10],'-o',label='双AR1解释');axes[1].plot(range(1,21),q['mean'][0,10],'--x',label='单ARMA(2,1)解释');axes[1].set_xlabel('从时刻1200起的预测步数');axes[1].set_ylabel('预测均值');axes[1].legend()
 eq=pd.read_csv(OUT/'equivalence_audit.csv');e=eq[eq.kind=='two_AR1_one_ARMA'];axes[2].plot(range(24),e.pair_probability,'o');axes[2].axhline(.5,color='gray',ls='--');axes[2].set_xticks(range(24),e.condition,rotation=65);axes[2].set_ylim(.45,.55);axes[2].set_ylabel('等先验下双AR1的配对概率');fig.suptitle('图13：精确观测等价：更多相同类型X也无法唯一判断物理因素数');save('13_equivalence')
 opt=pd.read_csv(OUT/'optimization_summary.csv');mc=pd.read_csv(OUT/'mcmc_summary.csv');sens=pd.read_csv(OUT/'integration_sensitivity.csv');fig,axes=plt.subplots(1,3,figsize=(14,4))
 g=opt.groupby('route').converged.mean();axes[0].bar(range(len(g)),g,color=COL[0]);axes[0].set_xticks(range(len(g)),g.index,rotation=65);axes[0].set_ylim(0,1);axes[0].set_ylabel('固定EM预算内收敛比例')
 axes[1].scatter(mc.Rhat_max,mc.ESS_min,s=16,alpha=.6);axes[1].axvline(1.1,color='red',ls='--');axes[1].axhline(100,color='red',ls='--');axes[1].set_xlabel('连续参数最大Rhat');axes[1].set_ylabel('最小有效样本数ESS')
 for j,(key,limit,label) in enumerate([('cover90_difference',.02,'覆盖差／0.02'),('width90_relative_difference',.1,'宽度相对差／0.10'),('joint_NLL_difference',.05,'联合NLL差／0.05')]):axes[2].plot(range(len(sens)),sens[key]/limit,'o',label=label,alpha=.75)
 axes[2].axhline(1,color='red',ls='--');axes[2].set_xlabel('固定积分敏感性检查编号');axes[2].set_ylabel('差异／预定数值容差');axes[2].legend(fontsize=7);fig.suptitle('图14：有限优化、连续参数采样与数值积分诊断');save('14_numerics')
 short=pd.read_csv(ROOT/'short/results/contrasts.csv');fig,axes=plt.subplots(1,2,figsize=(13,4))
 for ax,contrast,title in zip(axes,['complement_gain','joint_gain'],['双线相对最强A／B','联合状态相对强直接']):
  for j,(cc,L) in enumerate([(short,1000),(c,5000)]):
   z=cc.query('batch=="A" and condition!="ALL" and contrast==@contrast').set_index('condition').reindex(conds);ax.errorbar(np.arange(24)+(j-.5)*.16,z.gain,yerr=[z.gain-z.lo,z.hi-z.gain],fmt='.',capsize=2,label=f'训练／选择长度{L}',color=COL[j])
  ax.axhline(0,color='gray');ax.axhline(.005,color='red',ls='--');ax.set_xticks(range(24),conds,rotation=65);ax.set_ylabel('NLL改善／步');ax.set_title(title);ax.legend(fontsize=8)
 fig.suptitle('图15：相同5000步测试轨迹上的训练前缀敏感性（非独立复现）');save('15_length')
 print('15 Chinese figures completed')
if __name__=='__main__':main()

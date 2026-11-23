"""Chinese scientific figures from completed E4-A results only."""
import json,numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
from matplotlib.patches import Ellipse
from scipy.linalg import solve_discrete_lyapunov
from settings import *
from producer import generate,matrices,SYSTEMS,arma_alias
from model import unpack
from predict import component,infer
import native
FONT=ROOT/'fonts'/'NotoSansCJKsc-Regular.otf';fontManager.addfont(str(FONT));plt.rcParams.update({'font.family':FontProperties(fname=FONT).get_name(),'axes.unicode_minus':False,'figure.dpi':140,'savefig.dpi':170,'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
NAMES={'M1':'直接／不更新','M2':'递进／不更新','M3':'完整直接／在线','M4':'递进双线／在线','Xhistory':'原始历史AR参照'}
COLORS=['#9ca3af','#d97706','#2563eb','#059669','#9333ea']
def save(name):plt.tight_layout();plt.savefig(FIG/(name+'.png'),bbox_inches='tight');plt.close()
def main():
 assert (ROOT/'ANALYSIS_COMPLETE.json').exists() and (ROOT/'TRUTH_AUDIT_COMPLETE.json').exists()
 c=pd.read_csv(OUT/'contrasts.csv');summ=pd.read_csv(OUT/'group_summary.csv');checks=pd.read_csv(OUT/'confirmation_checks.csv');conf=pd.read_csv(OUT/'confirmation_summary.csv');state=pd.read_csv(OUT/'state_quality.csv');read=pd.read_csv(OUT/'state_readout.csv')
 conditions=sorted(summ.condition.unique());x=np.arange(20)
 fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
 for row,metric in enumerate(['working_factor_count','Markov_dimension']):
  for i,g in enumerate(['M3','M4']):
   z=summ.query('batch=="A" and length==5000 and group==@g').set_index('condition').reindex(conditions);axes[row].plot(x,z[metric],'-o',label=NAMES[g],color=COLORS[2+i])
  axes[row].set_ylabel('工作族因素数' if row==0 else '马尔可夫存储维数');axes[row].legend();axes[row].grid(alpha=.2)
 axes[0].set_title('图1：工作模型的因素数与记忆维数必须分开解释');axes[1].set_xticks(x,conditions);axes[1].set_xlabel('真实系统条件（仅评价时开放）');save('01_counts')
 fig,axes=plt.subplots(1,2,figsize=(12,4));names=['recursive_gain','history_gain'];lab=['递进相对完整直接','递进相对原始历史AR']
 for i,n in enumerate(names):
  ax=axes[i]
  for j,b in enumerate(['A','B']):
   z=c.query('length==5000 and condition=="ALL" and contrast==@n and batch==@b').iloc[0];ax.errorbar(z.gain,j,xerr=[[z.gain-z.lo],[z.hi-z.gain]],fmt='o',capsize=5,color=COLORS[2+j])
  ax.axvline(0,color='gray');ax.set_yticks([0,1],['测试A','测试B']);ax.set_title(lab[i]);ax.set_xlabel('NLL改善（nats／步），正值更好');ax.text(.98,.5,'实用门槛＋0.005\n超出此局部放大范围',transform=ax.transAxes,ha='right',va='center',color='firebrick');ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
 fig.suptitle('图2：两批独立测试的预测增量及95%置信区间');save('02_main_gain')
 fig,ax=plt.subplots(figsize=(12,4))
 for i,L in enumerate(LENGTHS):
  z=conf[conf.length==L].set_index('condition').reindex(conditions);ax.bar(x+(i-.5)*.35,z.rate,.35,label=f'长度{L}')
 ax.axhline(.8,color='red',ls='--',label='双状态条件参考门槛80%');ax.set_xticks(x,conditions);ax.set_ylim(0,1.08);ax.set_ylabel('双路线、双确认块均通过的种子比例');ax.set_title('图3：新增预测证据的严格独立确认（非物理因素识别率）');ax.legend(ncol=3);save('03_confirmation')
 d=pd.read_csv(OUT/'test_A.csv.gz');fig,axes=plt.subplots(1,2,figsize=(12,4))
 for i,g in enumerate(['M1','M2','M3','M4','Xhistory']):
  z=d.query('length==5000 and group==@g');axes[0].plot(HORIZONS,[z[f'NLL_h{h}'].mean() for h in HORIZONS],'-o',label=NAMES[g],color=COLORS[i]);axes[1].scatter(z.cover90.mean(),z.width90.mean(),label=NAMES[g],color=COLORS[i],s=65)
 axes[0].set_xlabel('联合预测步数');axes[0].set_ylabel('平均联合NLL／步');axes[0].legend(fontsize=8);axes[1].axvline(.9,color='gray',ls='--');axes[1].set_xlabel('90%预测区间实际覆盖率');axes[1].set_ylabel('平均区间宽度（原始X单位）');axes[1].legend(fontsize=8);fig.suptitle('图4：概率预测质量与区间可靠性（测试A，所有条件）');save('04_forecast_quality')
 index=next(i for i,m in enumerate(SYSTEMS) if m['id']=='B1');meta=SYSTEMS[index];b=unpack(json.loads((OUT/'B1_4011'/'model_5000.json').read_text()));xx,yy=generate(index,4011,1,NTEST,truth=True);m=b['recursive']['models']['factor2'];m=b['direct']['models']['factor2'] if m is None else m;p=native.run(xx[:1]/m['scale'],m);mu=p['mean'][0]*m['scale'];cov=p['cov']*m['scale']**2;t=np.arange(200,450);fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
 for k,ax in enumerate(axes):
  sd=np.sqrt(cov[t,k,k]);ax.plot(t,yy[0,t,k],color='black',lw=1,label='真实状态（评价答案）');ax.plot(t,mu[t,k],color=COLORS[3],label='族条件后验均值');ax.fill_between(t,mu[t,k]-1.645*sd,mu[t,k]+1.645*sd,color=COLORS[3],alpha=.2,label='条件90%区间');ax.set_ylabel('贡献坐标U' if k==0 else '贡献坐标V');ax.legend(ncol=3,fontsize=8)
 axes[0].set_title('图5：固定示例B1／种子4011，模型保留完整状态不确定性');axes[-1].set_xlabel('时间步');save('05_state_curves')
 fig,ax=plt.subplots(figsize=(6,5));t0=350;vals,vecs=np.linalg.eigh(cov[t0]);angle=np.degrees(np.arctan2(vecs[1,-1],vecs[0,-1]));e=Ellipse(mu[t0],2*np.sqrt(4.605170186*vals[-1]),2*np.sqrt(4.605170186*vals[0]),angle=angle,alpha=.25,color=COLORS[3]);ax.add_patch(e);ax.scatter(*mu[t0],label='后验均值');ax.scatter(*yy[0,t0],marker='x',s=70,color='red',label='真实状态');ax.autoscale();ax.set_xlabel('U贡献坐标');ax.set_ylabel('V贡献坐标');ax.set_aspect('equal',adjustable='datalim');ax.legend();ax.set_title('图6：同一时刻的二维联合90%后验椭圆');save('06_joint_posterior')
 i=next(i for i,m in enumerate(SYSTEMS) if m['id']=='D0');xx,yy=generate(i,4011,1,NTEST,truth=True);b=unpack(json.loads((OUT/'D0_4011'/'model_5000.json').read_text()));m=b['direct']['models']['memory2'];p=native.run(xx[:1]/m['scale'],m);fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True);axes[0].plot(t,yy[0,t,0],label='单个真实因素U',color='black');axes[0].plot(t,p['mean'][0,t,0]*m['scale'],label='单因素AR2后验均值');axes[0].legend();axes[0].set_ylabel('状态值');trace=np.load(OUT/'D0_4011'/'example_A.npz');w=trace['M4_weights'];ids=trace['M4_ids']%5;axes[1].plot(t,w[t][:,ids==2].sum(1),label='单因素AR2权重');axes[1].plot(t,w[t][:,ids==4].sum(1),label='双AR1模式权重');axes[1].set_ylim(0,1.05);axes[1].legend();axes[1].set_ylabel('工作模型权重');axes[1].set_xlabel('时间步');axes[0].set_title('图7：单因素长记忆需要两个存储坐标，不能当作两个原因');save('07_long_memory')
 fig,ax=plt.subplots(figsize=(8,5));z=checks[checks.length==5000].copy();z['family']=z.case.str[0]
 for fam,g in z.groupby('family'):ax.scatter(g.pole_error,g.conditional_effect_corr,s=15,alpha=.6,label=f'{fam}类')
 ax.axvline(.05,color='red',ls='--',label='跨路线极点误差门槛');ax.set_xlabel('跨独立训练池最大极点差');ax.set_ylabel('控制当前X和原解释后的预测增量相关');ax.set_title('图8：不同训练数据中的功能复现，不等于物理独立性');ax.legend(ncol=3);save('08_matching')
 state['family']=state.condition.str[0];read['family']=read.condition.str[0];fig,axes=plt.subplots(1,2,figsize=(12,4))
 for col,metric in enumerate(['corr','cover90']):
  z=state.query('batch=="A" and available==True').pivot_table(index='family',columns='line',values=metric).reindex(index=list('ABCDE'),columns=['direct','recursive','witness','oracle']);im=axes[col].imshow(z,vmin=0,vmax=1,cmap='YlGnBu');axes[col].set_xticks(range(4),['直接R','递进R','独立D','已知参数\n上界']);axes[col].set_yticks(range(5),list('ABCDE'));axes[col].set_title('后验均值与真值相关' if col==0 else '边际90%状态区间覆盖率')
  for (r,k),v in np.ndenumerate(z.to_numpy()):axes[col].text(k,r,f'{v:.3f}',ha='center',va='center',color='white' if v>.65 else 'black')
  fig.colorbar(im,ax=axes[col],shrink=.8)
 fig.suptitle('图9：真实族条件下的状态诊断；C的独立创新假设可能失配');save('09_state_quality')
 fig,ax=plt.subplots(figsize=(10,4));cats=list('ABCDE');pos=np.arange(5)
 for i,line in enumerate(['Xonly','direct','recursive','witness']):
  vals=read[read.line==line].groupby('family').R2.mean().reindex(cats);ax.bar(pos+(i-1.5)*.19,vals,.19,label={'Xonly':'只有当前X','direct':'直接状态＋X','recursive':'递进状态＋X','witness':'独立见证状态＋X'}[line])
 ax.set_xticks(pos,cats);ax.set_ylabel('独立真值审计轨迹的线性读出R²');ax.set_title('图10：可恢复信息子空间评价，不要求潜在坐标等于物理变量');ax.legend(ncol=2);save('10_readout')
 i=next(i for i,m in enumerate(SYSTEMS) if m['id']=='E1');xx,yy=generate(i,4011,1,NTEST,truth=True);F,Q,C,R=matrices(SYSTEMS[i]);m=dict(F=F,Q=Q,C=C,R=R,P=solve_discrete_lyapunov(F,Q),a=0.,scale=1.,kind='factor2');a=arma_alias(m);c0=component(xx[:1],m);c1=component(xx[:1],a);fig,axes=plt.subplots(2,1,figsize=(12,6));axes[0].plot(t,xx[0,t],label='两种结构共享的同一X');axes[0].set_xlabel('时间步');axes[0].set_ylabel('观测X');axes[0].legend();origin=10;h=np.arange(1,21);axes[1].plot(h,c0['mean'][0,origin],'-o',label='两个独立AR1因素');axes[1].plot(h,c1['mean'][0,origin],'--x',label='一个ARMA(2,1)因素');axes[1].set_xlabel('从时间步1200发出预测的未来步数');axes[1].set_ylabel('预测均值');axes[1].legend();axes[0].set_title('图11：观测等价对的X与预测完全重合，不能唯一判断物理因素数');save('11_equivalence')
 i=next(i for i,m in enumerate(SYSTEMS) if m['id']=='C0');xx,yy=generate(i,4011,1,NTEST,truth=True);F,Q,C,R=matrices(SYSTEMS[i]);P=solve_discrete_lyapunov(F,Q);p0=native.run(xx[:1],dict(F=F,Q=Q,C=C,R=R,P=P,a=0.));P1=np.diag(P@C);Q1=P1-F@P1@F.T;p1=native.run(xx[:1],dict(F=F,Q=Q1,C=C,R=R,P=P1,a=0.));fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
 for k,ax in enumerate(axes):
  ax.plot(t,yy[0,t,k],color='black',label='实际相关状态');
  for p,label,color in [(p0,'真实相关创新结构',COLORS[2]),(p1,'观测等价的独立创新结构',COLORS[3])]:
   mu=p['mean'][0,t,k];sd=np.sqrt(p['cov'][t,k,k]);ax.plot(t,mu,label=label,color=color);ax.fill_between(t,mu-1.645*sd,mu+1.645*sd,color=color,alpha=.15)
  ax.set_ylabel('U' if k==0 else 'V');ax.legend(ncol=3,fontsize=8)
 axes[0].set_title('图12：相同X概率分布可以对应不同隐藏状态后验');axes[-1].set_xlabel('时间步');save('12_correlated_alias')
 fig,ax=plt.subplots(figsize=(12,4))
 for L,color in [(1000,COLORS[1]),(5000,COLORS[3])]:
  z=c.query('batch=="A" and length==@L and condition!="ALL" and contrast=="recursive_gain"').set_index('condition').reindex(conditions);ax.errorbar(x+(-.1 if L==1000 else .1),z.gain,yerr=[z.gain-z.lo,z.hi-z.gain],fmt='o',capsize=2,label=f'长度{L}',color=color)
 ax.axhline(0,color='gray');ax.text(.98,.95,'实用门槛＋0.005超出局部纵轴范围',transform=ax.transAxes,ha='right',va='top',color='firebrick');ax.set_xticks(x,conditions);ax.set_ylabel('递进相对完整直接的NLL改善／步');ax.set_title('图13：递进架构的跨条件收益与数据长度（两长度为嵌套前缀）');ax.legend();save('13_length')
 print('13 Chinese figures complete')
if __name__=='__main__':main()

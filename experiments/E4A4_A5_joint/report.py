"""Chinese final report, generated only from frozen, audited outputs."""
import json,hashlib,html,numpy as np,pandas as pd
from pathlib import Path
from settings import *
from figures import FONT,LABEL
NAMES={'residual_gain':'创新提案相对强直接','joint_gain':'联合状态相对强直接','shared_vs_old_mix':'共享状态相对原混合','posterior_vs_point':'结构参数积分相对共享点模型','feedback_gain':'共享状态相对双线无共享','complement_gain':'双线相对选择集最强A/B','full_direct_reference':'双线相对完整强参照','dense_parameter_gain':'固定联合结构内的参数积分','structure_gain':'参数积分之外的结构积分','parameter_with_structures':'多结构内的参数积分','online_update':'原直接模型在线更新','original_vs_residual':'创新相对原后验路径混合','original_vs_joint':'联合相对原后验路径混合','path_vs_residual':'创新相对新增后验路径'}
def table(d):
 d=d.copy()
 for col in ['contrast','method','route']:
  if col in d:d[col]=d[col].replace(NAMES if col=='contrast' else LABEL)
 for col in d.select_dtypes(include='number'):
  d[col]=d[col].map(lambda v:'—' if pd.isna(v) else str(int(v)) if float(v).is_integer() and abs(v)<1e6 else f'{v:.3g}' if abs(v)<.0001 and v!=0 else f'{v:.6f}')
 return d.rename(columns={'batch':'测试批','condition':'条件','contrast':'比较','gain':'改善量','lo':'95%下界','hi':'95%上界','p_holm':'校正p值','practical_success':'实用通过','method':'方法','support':'通过数','n':'总种子数','route':'路线','coordinate':'目标','mean':'均值','NLL':'NLL／步','CRPS':'CRPS','cover90':'90%覆盖','joint_cover90':'90%联合覆盖','width90':'90%宽度','joint_NLL':'联合状态NLL','converged':'收敛','R2':'读出R²','pole_MAE':'极点绝对误差','row_sum_MAE':'谱行和绝对误差','R_error':'观测方差误差','Rhat_max':'最大Rhat','ESS_min':'最小ESS','cpu':'CPU秒','count':'数量'}).to_markdown(index=False)
CAPTIONS=[
('01_innovation','原始X与一步创新的动态','左、中、右分别固定为B0双状态、D0单因素长记忆、CE极高相关条件，均用种子4011第一条测试A，不按结果选图。横轴是滞后，纵轴为样本自相关。一步创新使用前一时刻的滤波状态发出预测，再减去新到达的X，没有未来信息。原始X有持续性不意味着基础模型仍遗漏持续性；创新接近白噪声时，从创新中另找动态状态可能只拟合噪声。创新中有剩余动态也只是提出新候选的依据，必须经独立确认，不能直接把残差命名为真实上游状态。'),
('02_main_gain','主要比较与机制来源','每个点代表相同测试轨迹上的NLL配对改善，横线为10000次种子与轨迹分层自助法95%区间；正值有利于新增方法。左图检查创新和联合初始化相对强直接、原路径与新增路径的收益，右图检查双线、共享状态和结构参数积分。实用门槛始终为0.005 nats/步；图上能看见微小正值或区间不跨零，并不足以通过对H0≤0.005的Holm校正检验。详细成败以表格为准。'),
('03_condition_gain','双线增量是否跨条件存在','左图参照仅在选择集选定的最强A/B；右图进一步允许选择集在A、B、R0、原M3和原始历史AR中选完整强参照。参照不按测试成绩重选。A、B测试批是两个独立数据批；每个条件仍共享训练模型，因此不是40个独立训练种子。红虚线表示实用门槛。个别条件的小幅优势不能代替整体、跨批次、跨条件的稳定收益，也不能作为发现物理因素的证据。'),
('04_capacity','记忆增加与新动态坐标的容量消融','四条曲线均相对同一基础none模型：仅增加观测AR记忆、仅增加一个潜在动态坐标、两者一起增加、放开同维数完整F/Q。基础本身可能是单因素AR2/AR3记忆，而非已知真实W1。这里采用同测试轨迹、同训练池平均的配对增量；图仅展示测试A，原始表保留A/B及置信区间。若完整记忆已经解释预测信息，增加坐标不应被解释成发现新原因。'),
('05_X_probability','预测X的评分与概率可靠性','左图是1、2、5、10、20步联合NLL除以步数，并非把边际NLL相加；越低越好。中图把边际90%覆盖与区间宽度一起看，防止只靠放宽区间提高覆盖。右图是完整未来20维混合预测分布的Rosenblatt联合区域覆盖，不是把20个边际区间拼起来，也不是矩匹配椭圆。用于判断区域成员关系的未来前缀不反馈预测模型。X预测校准仍不能证明潜在U/V后验校准。'),
('06_signal','不同潜在坐标之间可公平比较的作用S','统一目标为真实噪声前作用S=CW。带观测AR的候选按a·X过去＋CW计算其作用解释。不同结构的潜在坐标可能旋转、混合或有不同维数，不能把其第一、第二坐标强行当U/V。左图检查S的90%覆盖，右图检查proper score CRPS，越低越好。即便S可以预测和校准，也不意味着产生S的物理分解唯一。'),
('07_state_calibration','相关创新条件下的状态覆盖曲线','四列真实创新相关分别为0、0.5、0.9、0.99；真值只在测试完成后的审计中开放。曲线展示U、V平均边际覆盖，45度线代表名义与实际一致。独立创新点估计、有限bootstrap参数积分、全Q不可识别方向积分、连续参数MCMC积分和真参数oracle分开报告。全Q允许正确机制存在于模型族，并不意味着X能够选出真实相关性。均值曲线不能代替逐坐标覆盖及置信区间，后者见数据表。'),
('08_state_scores','校准、宽度与proper score要一起解释','左图是U/V边际90%区间宽度，中图是边际CRPS，右图是二维联合状态NLL；后两者越低越好。扩大不确定性可能改善覆盖，也可能付出宽度与密度评分代价。full-Q或Bayes若表现不好，可能来自观测不可识别、有限样本、参数近似、MCMC混合及积分误差，不能只归因为区间太窄。oracle使用真实参数，仅是事后参考，不是可部署方法。'),
('09_prior_sensitivity','相同X可对应不同物理状态后验','固定CE、种子4011、第一条测试A、时间200至399。黑线是真实U/V；有色曲线为混合后验均值，阴影由混合CDF反演得到90%区间，没有用均值±标准差冒充混合分位数。全Q对称先验与仅正相关先验使用相同可识别参数粒子，仅改变X无法辨别的相关方向先验。若曲线改变，说明物理解释依赖先验，而非新观测发现了真实相关性；正相关先验恰巧更接近本实验真值也不能反向证明它被数据识别。'),
('10_joint_matrices','联合状态的完整协方差已实现，但坐标身份未识别','固定B1和CE种子4011，显示R训练池联合模型F、Q的相关形式以及时刻350的完整滤波后验相关。非零非对角项说明实现没有把各状态强行独立处理。坐标1、2等是计算坐标，不能据此命名为源头Y、中介Z或推断物理因果边；可逆坐标旋转可改变这些矩阵，同时保持X的预测与似然不变。'),
('11_weights','在线概率更新的是候选解释，而非物理身份','固定B1种子4011，将每条路线的候选粒子按工作结构聚合显示。新X到达后，依其条件观测似然更新权重；每个模型内同一X只计一次，双线不把同一X似然相乘。概率集中可能只是有限粒子竞争的结果，不等于某个真实隐藏变量存在的后验概率，更不等于已经证明因果方向。状态条件协方差、候选权重和连续参数不确定性是不同层次。'),
('12_confirmation_readout','新增证据确认与真值读出必须分开','左图的通过要求R/D两独立训练池、两个新确认块全部达到0.005门槛，并在5760项确认检验Holm校正后通过及满足跨池功能匹配。0通过说明本方案未确认新增实用预测模式，不说明生成系统没有隐藏因素。右图另用stage13前10条拟合线性读出、后10条评价R²，真实状态仅用于审计读出，不回流推断。R²提高表示潜在表示包含信息，不能将该监督读出当成X-only物理后验。'),
('13_equivalence','精确等价证书与物理因素数拒判','固定E1示例：左为共享X，中为同一时点发出的双AR1与单ARMA(2,1)预测，二者重合。右图在所有条件的规范双AR1解释与其精确映射ARMA解释之间，采用映射相同先验后配对概率保持约0.5。此证书说明某些物理分解不能仅靠同类X区分，不是模型已经能自动识别一切未知等价结构。单AR2一般不与双AR1精确等价，审计表单独给出其协方差差异。'),
('14_numerics','数值近似的限制如实保留','左图显示新候选在固定3初值、24次EM上限内的收敛比例；未收敛不是模型被证明错误，也不能宣称已找到全局最优。中图是80个连续参数MCMC任务的最大Rhat与最小ESS，红线是预定1.1与100，未达标准的样本仍归档但不能冒称可信的收敛后验。右图固定四个新C条件种子4011两批，比较8/16个参数样本及17/33个相关方向节点，按预定容差标准化；超过1代表积分近似尚不稳定，禁止据此事后更换主设置。'),
('15_length','1000与5000训练前缀的配对敏感性','两种设置从同一训练、选择、确认轨迹分别取1000点前缀或5000点；测试均为相同5000点轨迹，相同预测时点。该图隔离学习历史长度对收益的影响，不是两次独立训练数据复现，也不能挑选更有利的长度替代原5000主要结论。1000条件仅补跑预测与确认，没有重复全部物理状态和连续参数Bayes审计。')]
def main():
 for p in ['PIPELINE_COMPLETE.json','short/ANALYSIS_A_COMPLETE.json','short/AUDIT_A_COMPLETE.json','small_validation/COMPLETE.json']:assert (ROOT/p).exists(),p
 c=pd.read_csv(OUT/'contrasts.csv');s=pd.read_csv(OUT/'support_summary.csv');cal=pd.read_csv(OUT/'calibration_intervals.csv');mc=pd.read_csv(OUT/'mcmc_summary.csv');op=pd.read_csv(OUT/'optimization_summary.csv');sens=pd.read_csv(OUT/'integration_sensitivity.csv');eq=pd.read_csv(OUT/'equivalence_audit.csv');rot=pd.read_csv(OUT/'rotation_audit.csv');sup=pd.read_csv(OUT/'confirmation_support.csv')
 win=c.query('condition=="ALL" and contrast=="complement_gain"').practical_success.all();strong=c.query('condition=="ALL" and contrast=="full_direct_reference"').practical_success.all();B=s[(s.condition.str.startswith('B'))&(s.route=='R2')];AD=s[s.condition.str[0].isin(['A','D'])];supportgo=(B.support.ge(16).sum()>=3 and AD.support.max()<=1)
 # Per-coordinate calibrated coverage rule; no truth-based changes to posteriors.
 cv=cal.query('condition in ["CL","CM","CH","CE"] and metric=="cover90" and coordinate in ["U","V"]').copy();cv['calibrated']=(cv['mean'].between(.87,.93))&(cv.lo<=.9)&(cv.hi>=.9);cv.to_csv(OUT/'calibration_decisions.csv',index=False)
 decision=dict(practical_threshold=.005,confirmation_passed=int(sup.supported.sum()),confirmation_total=len(sup),novel_mode_reference_go=bool(supportgo),dual_vs_best_single_both_batches=bool(win),dual_vs_full_reference_both_batches=bool(strong),physical_identity_go=False,E4_B_C_started=False,mcmc_converged=int(mc.converged.sum()),mcmc_total=len(mc),integration_stable=int(sens.stable.sum()),integration_total=len(sens),limitations=['finite EM and bootstrap budget','continuous posterior MCMC and quadrature diagnostics retained','exact observational and coordinate nonidentifiability','two operational incidents recorded'])
 (ROOT/'DECISION_SUMMARY.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2))
 mainc=c.query('condition=="ALL" and family=="main"')[['batch','contrast','gain','lo','hi','p_holm','practical_success']];sec=c.query('condition=="ALL" and family in ["secondary","mechanism"]')[['batch','contrast','gain','lo','hi','p_holm','practical_success']]
 pred=pd.read_csv(OUT/'prediction_summary.csv');pred=pred.groupby(['batch','method'])[['NLL','CRPS','cover90','width90','MSE']].mean().reset_index();pred=pred[pred.method.isin(['M3','J0','R0','R1','R2','R3','A','B','C','E','J1_param','J2_point','StrongestSingle','Xhistory'])]
 cvtab=cv[cv.method.isin(['INDEP_POINT','PARAM_ONLY','FULLQ_PARAM','BAYES_FULL','RECURSIVE_POINT','ORACLE'])].groupby(['batch','condition','method'])[['mean','lo','hi','calibrated']].mean().reset_index()
 extra=cal[(cal.batch=='A')&cal.method.isin(['INDEP_POINT','FULLQ_PARAM','BAYES_PARAM','BAYES_FULL','RECURSIVE_POINT','ORACLE'])&(cal.condition.isin(['CL','CM','CH','CE']))];pt=extra.pivot_table(index=['condition','method'],columns='metric',values='mean').reset_index();pt=pt[['condition','method','CRPS','width90','joint_NLL','joint_cover90']]
 pa=pd.read_csv(OUT/'canonical_parameter_errors.csv');pa['condition']=pa.case.str.split('_').str[0];pas=pa.groupby('condition')[['pole_MAE','row_sum_MAE','R_error']].mean().reset_index();runtime=pd.read_csv(OUT/'runtime_summary.csv');short=pd.read_csv(ROOT/'short/results/contrasts.csv');shortmain=short.query('condition=="ALL" and family=="main"')[['batch','contrast','gain','lo','hi','p_holm']]
 mem=pd.read_csv(OUT/'memory_signal_probability.csv.gz');mem['condition']=mem.case.str.split('_').str[0];sig=pd.read_csv(OUT/'small_common_signal_summary.csv');ms=pd.concat([mem.groupby(['batch','condition','method']).mean(numeric_only=True).reset_index(),sig],ignore_index=True);ms=ms[(ms.batch=='A')&ms.condition.isin(['CL','CM','CH','CE'])&ms.method.isin(['C5_memory','J0','J1','J2'])][['condition','method','NLL','CRPS','cover90','width90']];pw=pd.read_csv(OUT/'registered_power.csv');pw=pw[(pw.sd==.0075)&(pw.true_gain==.01)]
 sm=pd.read_csv(ROOT/'small_validation/test_B_small.csv.gz').pivot(index=['case','seed','trajectory'],columns='method',values='NLL');sm['condition']=sm.index.get_level_values('case').str.split('_').str[0];sm['双线相对最强A/B']=sm.BestSingle-sm.E;sm['双线相对完整强参照']=sm.StrongestSingle-sm.E;smt=sm.groupby('condition')[['双线相对最强A/B','双线相对完整强参照']].mean().reset_index()
 lines=[f'''# E4-A4／A5：双线递进状态发现与概率可靠性实验报告

日期：2026年11月24日。实际完成A4.1机制修正、A4.2概率诊断、A4.3联合推断、A4.4等价性审计及A5消融。未启动E4-B/C、ARD、Split-Merge、PCMCI+、SCM和失效预警。

## 结论先读

- 新候选的严格独立确认通过 **{int(sup.supported.sum())}/{len(sup)}**；尚未证明新增实用预测模式。隐藏生成因素存在与否不能由“0通过”反向判断。
- 双线结构／参数积分相对选择集最强A/B，在两批独立测试都通过0.005 nats/步门槛与校正检验：**{'是' if win else '否'}**；相对完整强参照：**{'是' if strong else '否'}**。
- 已实现完整状态协方差、全Q族、bootstrap参数积分与连续参数MCMC，但“实现了概率推断”不等于概率已校准。连续参数任务 **{int(mc.converged.sum())}/{len(mc)}** 达到预设混合标准；固定数值积分检查 **{int(sens.stable.sum())}/{len(sens)}** 达到全部容差。未通过项全部保留，不挑选或重跑有利版本。
- **物理身份No-Go**：双AR1／单ARMA(2,1)、全Q不可识别方向及可逆坐标变换保留相同X分布、预测或似然。不能声称恢复了真实Y→Z→X，更没有证明上游失效提前预警。
- 本轮最明确的结果是界定方法边界：受限表示可含隐藏动态信息，但递进提案、额外维数和更复杂联合推断，没有因此自动获得强直接方法无法得到的稳定实用增量。

## 1. 范围、规模与生成方程

共24条件×20训练种子，5000主长度与1000训练前缀各480组合，共960组合。两个长度使用嵌套前缀，不是960套独立底层数据。5000主实验每批测试24×20×20＝9600条轨迹，20种方法＝192000条逐轨迹评分；测试A/B独立，总384000条主评分。用户随后要求转为小规模思路验证，故保留已完成的短前缀完整测试A（192000条评分），取消全量短测试B，只补6条件×3种子×5轨迹＝90条轨迹、1800条评分，不再新训练。容量对照、概率审计、确认与连续链另存。

一般系统：W(t+1)=F W(t)+η(t)，X(t)=C W(t)+ε(t)，η~N(0,Q)，ε~N(0,R)，初态取稳态。A为单AR1因素；B为两个独立AR1贡献之和；C为相关创新的双AR1；D为一个AR2长记忆因素（状态为当前值与滞后）；E保留双AR1与单ARMA(2,1)的精确等价。新增CL/CM/CH/CE取F=diag(0.9,0.85)、两坐标稳态边际方差1、R=0.5，真实创新相关ρ分别0/0.5/0.9/0.99。完整数值在[SYSTEMS.json](SYSTEMS.json)。本轮不是任意未知非线性系统。

模型学习函数只接收X和优化种子，不提供真实W、ρ、因素数量或系统标签。预先知道的候选族、最大状态容量4、线性高斯假设是方法限制，应与“完全无结构的自动因果发现”区分。

## 2. 代码审查与方法的具体改变

旧实验固定维数缓冲全部扩展至4，矩阵16、向量4、增广求逆32，并用1至4维参考滤波与预测密度验证。原代码已有完整Kalman／RTS／FFBS协方差，但旧M4只是候选模型的规范混合；不能称为共同状态向量的统一后验。审查见[CODE_REVIEW.md](CODE_REVIEW.md)。

| 对象 | 实际实现 | 能支持的解释 |
|---|---|---|
| R0 | 原始X谱矩／延迟初始化，保留原完整工作族及同维数dense参照 | 强直接点模型；避免用弱基准制造递进收益 |
| R1 | 完整FFBS路径初始化，保留原路径候选 | 路径是X的概率变换，不是外部新信息 |
| R2 | 严格一步先验创新产生提案，再用原X似然拟合 | 残差只提出候选，不作为独立观测重复计分 |
| R3／J1／D | 基础路径与创新交叉矩初始化，共同F/Q/R完整重估 | 共享状态空间内积分条件不确定性，不保证物理坐标身份 |
| J0 | 原M4规范模型混合 | 多解释混合，非共享状态向量 |
| J2／E | 4类结构×2粒子／池，双线各取4，总8粒子／池 | 经验Bayes有限离散结构／参数积分，非任意连续后验 |
| A／B | 仅直接／仅递进，各8粒子／池 | 与E匹配最终粒子预算、数据与原始历史 |
| C | 双线无共享状态的点模型混合 | 分离组合与共享状态效果 |
| J1_param／J2_point | 固定dense结构参数积分／多结构点参数 | 分解参数与结构不确定性的预测增量 |

所有方法用R/D独立训练池各20条，共40条；选择10条。A/B的单线是算法单线，不是只获得一个训练池。新拟合3初值×最多24次EM，参数自助粒子12次EM；稳定性投影、失败及收敛历史全部保留。训练可平滑，测试只能因果滤波。完整直接参照另保留旧80次EM工作族，因此全部训练CPU不完全一致；不能将全部差异归因为递进架构。数据处理不能凭同一完整X历史生成额外外部信息，只能改善受限表示、模型选择或优化。

## 3. 数据隔离、冻结与统计规则

stage0生成70条：R前20、D次20、选择次10，旧确认后20本轮不用。stage10新20条分两块确认；stage11/12各20条新测试A/B；stage13另20条真值审计读出，前10拟合、后10评价。不同stage独立SeedSequence；模型／选择冻结后才打开测试，测试预测完成后才开放隐藏真值。1000设置只获得训练／选择／确认同一轨迹前1000点，测试仍5000；前缀复制与无未来泄漏已审计。

主要指标是未来20步联合NLL／步，发出时点200、300、…、4900。H0：改善≤0.005，门槛从未降低。主6比较每批整体Holm6、分24条件Holm144；次要4比较Holm4／96；机制3比较Holm3／72；容量4比较Holm4／96。独立确认5760项Holm；候选四次确认均通过且功能匹配才支持新增。C相关条件诊断Holm16，连续Bayes额外Holm12。报告未校正95%效应区间和校正p值，不把区间当同步多重置信区间。主bootstrap10000次，以20种子为独立训练层，再在选中种子内重采配对轨迹；状态覆盖按种子聚类，不能把大量时间点当独立样本。

功效是预先指定假设下的近似，不是事后用实际效果解释失败。20训练种子、跨种子SD假设0.0075、真实改善假设0.010、H0门槛0.005时，单检验及保守Bonferroni界下的功效如下（Holm实际功效依其它检验结果而变）：

{table(pw)}

## 4. 主要预测结果

改善量为参照NLL减新方法NLL，正值有利。以下是全部预注册主比较，不按显著性筛选。

{table(mainc)}

### 4.1 强参照与不确定性分解

{table(sec)}

### 4.2 绝对预测评分

各条件等权均值，所有方法均在同一批轨迹上评价。M1弱无更新消融不作为主胜出参照；R1_original与J0同一原混合；E与J2、D与J1与R3是有意别名，不算额外独立方法复现。

{table(pred)}

## 5. 新增证据的独立确认

{table(s[['condition','route','support','n']])}

通过率的分母为20训练种子，不是时间点。B类至少3/4条件各16/20通过、A/D每条件最多1/20错误新增的参考Go：**{'通过' if supportgo else '未通过'}**。A/D即使0/20，Wilson双侧95%上界约16.1%，不是已证明真实错误率≤5%。不把确认未通过的候选删除成“没有失败”；保留其原样测试评分及缺失状态。全部预测方法可用率见逐轨迹available字段。代数可观测矩阵满秩也不代表物理身份可识别。

## 6. 概率状态估计：三种不确定性要分开

第一层是在给定参数和结构下的完整状态条件协方差；第二层是有限bootstrap参数与候选结构不确定性；第三层是X的观测分布根本无法消除的物理解构歧义。本轮分别评价，不能将其中一种冒充另外两种。

贡献规范模型F=diag(a,b)、C=[1,1]，X只约束P的行和k。令P(s)=[[k1−s,s],[s,k2−s]]，Q=P−FPFᵀ，在合法PSD区间内改变s可以保持整个X分布。全Q主诊断对ρ∈[−0.9999,0.9999]均匀先验取17点Gauss–Legendre积分；正相关先验为明确敏感性对照，非数据学出的规律。INDEP_POINT与FULLQ_POINT在ρ=0处同一规范代表，覆盖对照相等是设计结果，并非已检验所有可能全Q点估计。

PARAM_ONLY是4个训练bootstrap参数；FULLQ_PARAM为4×17；新C条件另用连续参数MCMC的8个固定样本×17相关节点。前两者是离散经验Bayes；后者目标为显式先验下精确观测似然的连续参数后验，但样本与相关方向积分仍是近似。不可识别ρ的后验应保留先验，更多同类X不能将其收缩到真实ρ。

### 6.1 新C条件90%覆盖与置信区间

下表为U/V两个坐标的覆盖、下界及上界的平均，便于阅读；严格逐坐标判定见[calibration_decisions.csv](results/calibration_decisions.csv)，平均区间不用于宣称校准通过。注册规则为覆盖落入[0.87,0.93]且各自95%CI包含0.9，测试A/B分别检查。

{table(cvtab)}

### 6.2 宽度、CRPS与联合状态密度

测试A，所有数字来自精确有限高斯混合分布的评分。任意坐标J1/J2不伪造U/V联合NLL，而用共同S评价；物理U/V只针对贡献坐标候选。下表联合覆盖为Rosenblatt区域，不是HPD或矩匹配椭圆。

{table(pt)}

### 6.3 C5单因素记忆基线与共同S

C5使用已经冻结的R_none／D_none单因素AR1/AR2/AR3训练BIC父模型，等先验、按当前及过去X在线更新。没有重新拟合或用测试选记忆阶数。其任意记忆坐标不对应物理U/V，因此U/V联合状态评分明确不适用。原冻结预测中none容量对照已保留；下表额外补齐共同S的描述性评分。该补表在主测试之后实现，按用户小规模要求仅用4个新C条件×3种子×5条轨迹；各方法都取完全相同轨迹。补表不新增检验，不替换主要或预注册C16比较。

{table(ms)}

oracle真实F/Q只在审计时提供，不属于可实现模型。固定真实ρ下的频率覆盖与先验平均的Bayes可信度不同，不能因有Bayes标签就保证每个固定真机制均达到90%。覆盖不足可能包含均值偏差、协方差方向错误、模型失配和数值误差，不只意味着区间过窄；宽度也有可能超过oracle却仍漏掉真值。

### 6.4 连续参数后验与数值诊断

新C共80任务；5维规范谱参数取显式截断正态先验，精确Gaussian likelihood，MAP初始化后两条独立自适应Metropolis链，预热1000、保留3000，保留期停止自适应。每条完整链、Rhat、ESS、接受率和优化器状态归档。标准为最大Rhat≤1.1、最小ESS≥100、接受率0.05–0.8；**{int(mc.converged.sum())}/80**通过。诊断不保证全局模式探索充分；未通过任务不能宣称连续后验已经可靠收敛，也不能只报通过子集的好成绩。

{table(mc.groupby('converged').agg(count=('case','size'),Rhat_max=('Rhat_max','median'),ESS_min=('ESS_min','median'),cpu=('cpu','mean')).reset_index())}

固定CL/CM/CH/CE种子4011两批，8/16参数样本和17/33相关节点比较，覆盖差≤0.02、宽度相对差≤10%、联合NLL差≤0.05为预定数值容差：**{int(sens.stable.sum())}/16**全部达到。完整结果如下，不据此选择新主设置。

{table(sens)}

## 7. 参数恢复、坐标自由度与不可识别性

以下只评价观测规律可约束的极点、P行和与R；不把真正不可识别的Q相关性当成应该精准恢复的参数。任意共享状态的F/Q可以旋转而不改变X，故原始矩阵条目不能直接解释为因果强度。

{table(pas)}

连续参数逐参数90%区间覆盖、误差与MCMC状态在[continuous_parameter_audit.csv](results/continuous_parameter_audit.csv)，没有把测试真值用于后验校准。

精确双AR1→单ARMA(2,1)映射下，观测自协方差、滤波观测似然、未来均值和协方差最大审计差 **{eq.query('kind=="two_AR1_one_ARMA"').maximum_error.max():.3g}**；配对后验概率最大偏离0.5为 **{abs(eq.query('kind=="two_AR1_one_ARMA"').pair_probability-.5).max():.3g}**。全Q不可识别方向最大观测误差 **{eq.query('kind=="correlated_Q_null"').maximum_error.max():.3g}**；坐标旋转最大预测／似然误差 **{rot.maximum_error.max():.3g}**。这些是程序证书，不是从标签猜拒判。

单AR2一般不等价于单ARMA(2,1)，不能把两者混称；AR2移除MA项后的协方差差异已单独存表。一个物理因素可能需要多个Markov存储坐标；两个贡献模式也可能改写成一个记忆因素。新增模式证据、代数可观测秩、物理因素数与因果链身份四者必须区分。[SCHEMA_ERRATUM.md](SCHEMA_ERRATUM.md)解释旧兼容factor_count字段不参与数值、不得作为发现因素数。

## 8. 1000训练前缀敏感性

1000设置480组合完成扩展／确认及完整测试A，旧基础族来自同一封存提交的1000模型。测试是与5000完全相同的5000点轨迹。用户在运行中要求小规模验证，原短前缀全量B已取消，不声称完成两批完整长度复现。以下仅A为原规则的次要校正结果，不能替代5000主要结论；没有全部短前缀物理校准或连续MCMC重复。

{table(shortmain)}

### 8.1 按最新要求补跑的小规模验证

固定A0、B1、C0、D0、E1、CE六类代表系统，种子4011–4013，各取原B中前5条轨迹，共90条。复用冻结1000训练模型，保持既有测试轨迹5000点以便与主设置配对，未再拟合长序列。结果与输入哈希在[small_validation](small_validation/PLAN.json)，只用于检查思路和计算链路，不做3种子显著性或误报率证明；也不冒充全新尚未接触的测试分区。

小样本平均NLL改善如下，正值有利于双线。均远小于0.005实用参考门槛；它验证了计算流程可以运行，尚未给出递进／双线具有独特价值的正面证据。个别正负数值不作显著性或总体稳定性判断。

{table(smt)}

后续新实验默认先用2–3类系统、3种子、约20训练／5验证／10测试轨迹、长度300–500验证一个核心假设，不自动扩到大网格。详见[USER_SMALL_SCALE_AMENDMENT.md](USER_SMALL_SCALE_AMENDMENT.md)。

## 9. 优化预算与实际运行限制

主设置新候选拟合累计记录CPU **{runtime.cpu.sum()/3600:.3f}小时**，每组合中位 **{runtime.cpu.median():.2f}秒**；continuous Bayes另 **{mc.cpu.sum()/3600:.3f} CPU小时**。这些不包含测试、状态审计、报告生成及新C旧基础族预拟合的全部成本，不能称端到端总运行时间。2核云实例限制下的并行只改变墙钟时间。新扩展EM收敛比例 **{op.converged.mean():.2%}**，未收敛与失败详见原参数history及诊断表。固定预算结果反映本实现，不是证明其它优化器都无效。

## 10. 执行事故与审计透明性

1. 第一轮480拟合写到临时/tmp；云环境重建后丢失。已保留旧冻结清单与日志，[storage_incident/INCIDENT.json](storage_incident/INCIDENT.json)。该次运行不计有效结果或独立复现。随后用相同冻结代码、数据与种子重建到持久/workspace，当前480模型哈希全部复核。旧20条件400份5000基础模型逐字节恢复自Git提交38cda63b1ee470cdc53efa5abd86b2208028d921。
2. 主测试长进程启动并加载原冻结代码后，磁盘开发副本曾加入未登记诊断标签等改动，造成后续状态审计KeyError以及短测试哈希检查阻断。主测试A/B使用已加载的原20方法代码并正确完成，未重写结果；短测试当时未打开。根据原预注册SHA逐文件恢复原始版本，不更新旧登记记录“追认”改动。恢复记录及晚版本保存在[evaluation_drift_incident/RESTORATION_RECORD.json](evaluation_drift_incident/RESTORATION_RECORD.json)，恢复脚本按SHA查验。之后再运行状态审计及首次短测试，冷重放与无未来泄漏检查通过。故不能声称开发目录始终没有变化，但已保存执行边界、原哈希与恢复证据。

统计结果未据事故改阈值、调模型或重挑测试。所有registered标记原样保存，额外报告脚本属于测试后描述性展示。用户缩小规模属于明确的后续范围调整，不追认成原预注册已全部完成。冻结和重放审计见[AUDIT_COMPLETE.json](AUDIT_COMPLETE.json)、[short/AUDIT_A_COMPLETE.json](short/AUDIT_A_COMPLETE.json)。

## 11. 十五张中文曲线图与逐图解释
''']
 for n,(file,title,caption) in enumerate(CAPTIONS,1):lines.append(f'### 图{n}：{title}\n\n![图{n}：{title}](figures/{file}.png)\n\n{caption}\n')
 lines.append('''## 12. 对原课题意味着什么

仅观测X时，可以估计对预测有用的状态分布与动态作用，但从同一X递进提取并不会产生未传到X的上游信息。即使潜在表示能读出一些真实状态，也不能因此唯一确定真实直接／间接因果链。在本轮限定候选、预算与数据下，尚未得到递进／双线结构本身的稳定实用优势，不能直接进入因果发现或预警并把前提当作已解决。

本轮的Go／No-Go保持冻结规则：新增模式参考未达到时不宣称发现；双线未超过强单线时不宣称互补；状态概率未可靠校准时不把其置信度当物理证据；存在精确等价证书时拒绝唯一物理解释。后续如继续，应先明确增加了哪项可识别假设、外部信息或有效限制，再安排新的独立协议，而不是降低本轮门槛或加深递归次数。

### 失败来源分别判断

| 问题 | 本轮能得出的判断 | 不能据此声称 |
|---|---|---|
| 后验路径递进提案 | 没有确认新增实用预测模式 | 不是证明系统没有隐藏因素，或一切递进方法永远无效 |
| 条件创新提案 | 计算严格因果、只作初始化；收益未达到冻结标准 | 不是仅凭残差动态就发现真实新状态 |
| 共享联合概率 | 已实现完整条件协方差且数值重放通过；预测收益和物理校准另未确证 | 不能把统计未胜出统称概率公式算错，也不能把实现正确当科学目标已成功 |
| 参数估计／积分 | 观测参数误差可审计；80个连续链未达到完整混合标准，积分敏感性可能超容差 | 不能称连续后验已可靠积分或精确恢复物理Q |
| 结构失配 | 独立Q物理解构可能与相关真值不符；允许全Q后仍有等价方向 | 不能只换成全Q就保证真实隐藏状态被识别 |
| 隐藏状态身份不可识别 | 精确观测等价及坐标旋转给出证书 | 不能靠更多同分布X、递归深度或更低预测误差消除这些歧义 |
| 计算预算限制 | 固定EM／采样上限有未收敛项；限制本实现可得结论 | 不能事后说“只因数据不够”或降低0.005门槛 |

具体下一步应先在与本轮测试隔离的开发任务上，检查规范谱参数的后验几何、采用更有效的参数化／采样方案并通过多链ESS及积分稳定性；随后重新登记独立验证，不能用修复后的结果覆盖本轮。若目标仍是物理因果链，则须另外说明哪种额外假设、干预或观测环境能打破已给出的等价证书。在这些前提明确之前，不进入因果图恢复和失效预警。

## 13. 复现与数据索引

- [PROTOCOL.md](PROTOCOL.md)、[POSTTEST_PLAN.md](POSTTEST_PLAN.md)、[PRETEST_METRIC_AMENDMENT.md](PRETEST_METRIC_AMENDMENT.md)、[BAYES_PROTOCOL.md](BAYES_PROTOCOL.md)、[SHORT_PROTOCOL.md](SHORT_PROTOCOL.md)：原冻结定义及预先明确的区域／指标修订。
- [RUN.md](RUN.md)：安装、旧基线来源、分区与按阶段复现；parameters在results各case.json.gz，连续链在results/bayes。
- results/test_A.csv.gz、test_B.csv.gz、controls_A/B.csv.gz：逐轨迹评分；contrasts.csv：全部条件与校正比较；confirmation.csv与confirmation_support.csv：确认；prediction_summary.csv：汇总。
- state_marginal／state_joint／signal_probability／state_readout：完整概率与审计读出；calibration_intervals／calibration_contrasts／calibration_decisions：区间、校正与规则判定；joint_X_probability：未来联合区域；interpretation：坐标与身份解释。
- equivalence_audit／rotation_audit／online_audit／replay_audit：数值证书；optimization_summary／fit_failures／proposal_summary／runtime_summary／mcmc_summary／integration_sensitivity：未收敛、失败及近似误差。
- short目录保留独立短前缀代码、全部480模型、完整测试A逐轨迹评分；small_validation保留90条小样本B与配对结果。原短全量B未完成不纳入统计。主与短结果均不覆盖旧A1–A3。
- 每份模型记录训练输入与旧基线SHA，生成器种子可重建X；冻结JSON及最终MANIFEST.sha256覆盖源代码、参数、评分、图表与报告。清单自身不自哈希。
- 大CSV.gz以compact/INDEX.json指定的CSV.xz载荷无损保存，以节省云端空间。先运行restore_scores.py --verify-only验证可逐字节恢复原gzip哈希，再运行restore_scores.py恢复原路径；不重算实验、不改变任何数值或冻结记录。图表与报告可直接阅读；归档哈希清单覆盖实际载荷与恢复说明。
''')
 text='\n\n'.join(lines);(ROOT/'REPORT.md').write_text(text)
 import markdown
 body=markdown.markdown(text,extensions=['tables','fenced_code']);css='''@font-face{font-family:CJK;src:url("fonts/NotoSansCJKsc-Regular.otf")} @page{size:A4;margin:16mm 13mm;@bottom-center{content:counter(page);font-size:9pt}} body{font-family:CJK,sans-serif;font-size:9pt;line-height:1.65;color:#18212e} h1{font-size:20pt} h2{font-size:14pt;break-after:avoid} h3{font-size:11pt;break-after:avoid} img{max-width:100%;height:auto} table{border-collapse:collapse;width:100%;font-size:6.5pt;table-layout:auto} th,td{border:1px solid #cbd5e1;padding:3px;overflow-wrap:anywhere} th{background:#eef2ff} tr{break-inside:avoid} a{color:#1d4ed8;text-decoration:none} code{font-size:8pt;overflow-wrap:anywhere}'''
 page=f'<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8"><title>E4-A4 A5实验报告</title><style>{css}</style><body>{body}</body></html>';(ROOT/'REPORT.html').write_text(page)
 from weasyprint import HTML
 HTML(string=page,base_url=str(ROOT)).write_pdf(ROOT/'REPORT.pdf');print('Chinese report, HTML and PDF generated')
if __name__=='__main__':main()

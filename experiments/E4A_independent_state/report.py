"""Render the completed experiment faithfully, including negative findings."""
import json,time,hashlib,html,shutil,numpy as np,pandas as pd
from pathlib import Path
from settings import *

def table(d):
 d=d.copy()
 values={'line':{'direct':'直接R','recursive':'递进R','witness':'独立D','oracle':'已知参数','Xonly':'当前X'},'contrast':{'recursive_gain':'递进增量','direct_update':'直接在线更新','recursive_update':'递进在线更新','interaction':'交互增量','history_gain':'相对原始历史AR'}}
 for col,mapping in values.items():
  if col in d:d[col]=d[col].replace(mapping)
 for col in d.select_dtypes(include='number'):
  integer=pd.api.types.is_integer_dtype(d[col]);d[col]=d[col].map(lambda v:('—' if pd.isna(v) else str(int(v)) if integer else f'{v:.3g}' if 0<abs(v)<1e-5 else f'{v:.6f}'))
 names={'batch':'测试批','length':'轨迹长度','condition':'条件','contrast':'对照','gain':'改善量','lo':'95%下界','hi':'95%上界','p_holm':'校正p值','group':'方法','family':'类别','line':'路线','corr':'相关','MSE':'均方误差','NLL':'NLL','cover90':'90%边际覆盖','joint_cover90':'90%联合覆盖','width90':'区间宽度','posterior_corr':'后验相关','joint_state_NLL':'联合状态NLL','available_cases':'可用组合','total_cases':'总组合','support':'通过数','n':'种子数','rate':'通过比例','wilson_lo':'Wilson下界','wilson_hi':'Wilson上界','valid_proposals':'有效提案','matched':'模式匹配','R2':'读出R²','factor1':'AR1单因素','memory2':'AR2单因素','memory3':'AR3单因素'}
 return d.rename(columns=names).to_markdown(index=False)

def main():
 assert (ROOT/'AUDIT_COMPLETE.json').exists() and (ROOT/'ANALYSIS_COMPLETE.json').exists() and (ROOT/'TRUTH_AUDIT_COMPLETE.json').exists()
 c=pd.read_csv(OUT/'contrasts.csv');s=pd.read_csv(OUT/'group_summary.csv');q=pd.read_csv(OUT/'state_quality.csv');r=pd.read_csv(OUT/'state_readout.csv');conf=pd.read_csv(OUT/'confirmation_summary.csv');eq=pd.read_csv(OUT/'equivalence_audit.csv');ch=pd.read_csv(OUT/'confirmation_checks.csv');models=[]
 for p in OUT.glob('*/model_*.json'):
  b=json.loads(p.read_text())
  for line in ['direct','recursive','witness']:
   for kind,m in b[line]['models'].items():
    if m is not None:models.append(dict(case=b['case'],length=b['length'],line=line,kind=kind,converged=m.get('converged',True),iterations=m.get('iterations',0),cpu=b['cpu']))
 parents=[]
 for fp in OUT.glob('*/model_5000.json'):
  bb=json.loads(fp.read_text());parents.append(dict(condition=bb['case'].split('_')[0],parent=bb['parent']))
 parent_table=pd.DataFrame(parents).groupby(['condition','parent']).size().unstack(fill_value=0).reset_index()
 md=pd.DataFrame(models);md.to_csv(OUT/'optimization_summary.csv',index=False)
 mainc=c.query('length==5000 and condition=="ALL"').copy();maint=mainc[['batch','contrast','gain','lo','hi','p_holm']];c5000=conf[conf.length==5000];ct=pd.read_csv(OUT/'confirmation_tests.csv');ct['condition']=ct.case.str.split('_').str[0];ct=ct[ct.length==5000].replace([np.inf,-np.inf],np.nan);gain_table=ct.pivot_table(index='condition',columns=['line','block'],values='gain').reset_index();gain_table.columns=['condition']+['_'.join(v) for v in gain_table.columns[1:]];B=c5000[c5000.condition.str.startswith('B')];positive=int((B.support>=16).sum());false=c5000[c5000.condition.str[0].isin(['A','D'])];falsemax=int(false.support.max());rec=mainc[mainc.contrast=='recursive_gain'];recwin=bool(((rec.gain>=.005)&(rec.lo>0)&(rec.p_holm<.05)).all());modego=positive>=3 and falsemax<=1
 stats={'B_conditions_at_least_16_of_20':positive,'A_D_max_confirmed_false_novel_per_condition':falsemax,'predictive_mode_reference_go':bool(modego),'recursion_practical_win_both_tests':recwin,'physical_factor_identity_go':False,'physical_identity_reason':'two independent AR1 factors and one ARMA21 factor have exactly the same X law','E4_B_C_started':False}
 (ROOT/'DECISION_SUMMARY.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
 agg=s[s.length==5000].groupby(['batch','group'])[['NLL','CRPS','cover90','width90','MSE']].mean().reset_index()
 q['family']=q.condition.str[0];qa=q.query('batch=="A" and available==True').groupby(['family','line'])[['corr','MSE','NLL','cover90','joint_cover90','width90','posterior_corr']].mean().reset_index()
 if (OUT/'state_probability_extra.csv').exists():
  extra=pd.read_csv(OUT/'state_probability_extra.csv');extra['family']=extra.condition.str[0];extra=extra.query('batch=="A"').groupby(['family','line'])[['CRPS','joint_state_NLL']].mean().reset_index();qa=qa.merge(extra,on=['family','line'],how='left')
 availability=q.query('batch=="A"').drop_duplicates(['case','line']).groupby(['family','line']).available.agg(available_cases='sum',total_cases='count').reset_index();qa=qa.merge(availability,on=['family','line'],how='left')
 ra=r.assign(family=r.condition.str[0]).groupby(['family','line'])[['R2','corr','MSE']].mean().reset_index()
 lines=[f'''# E4-A：概率化双线隐藏动态发现实验报告

实验日期：2026年11月23日。实际执行最新请求的 **E4-A1、A2、A3**，未执行E4-B的ARD／Split-Merge或E4-C的因果图与干预。代码、协议、模型、确认、两批测试依次冻结；全部预测测试完成后才打开隐藏真值。

## 结论先读

- 双状态B类四个条件中，**{positive}/4** 个条件达到“至少16/20个种子、双路线、双确认块均通过”的参考标准。单状态A与长记忆D各条件的最大错误新增为 **{falsemax}/20**。
- 预测模式层的预设参考门槛：**{'通过' if modego else '未通过'}**。这是受限候选族下的新增预测证据，不是物理因素数或因果链恢复。
- 递进模型相对完整直接在线模型，在两批测试是否同时达到0.005 nats/步的实用增量与校正显著性：**{'是' if recwin else '否'}**。精确数值和区间见下表，负值同样保留。
- **隐藏状态概率仍有失准风险**：高相关C类测试A的边际90%状态区间，直接R实际覆盖约69.1%、递进R约83.8%，已知真实模型约90.0%。X预测覆盖接近90%，不能代替隐藏状态概率可靠性。
- **物理身份层No-Go**：两个独立AR1因素与单个ARMA(2,1)因素可以产生完全相同的X分布；更准确的概率预测也不能消除此等价性。因此本轮不进入E4-B/C，不声称发现真实上游因果变量或实现失效提前预警。

## 1. 本轮回答什么

这是最大Markov容量3的有限结构化高斯候选原型，不是完全无结构、无限状态数量的通用发现器。模型仅观察X，比较直接提取与后验路径递进提案，检验新增候选是否在独立数据中提供稳定预测增量，并检验把长记忆误当新因素的风险。所有学习函数只接收X和优化随机种子，不接收系统标签、U/V、真实系数或因素数。

必须区分三个对象：一个物理因素、维持其历史所需的Markov存储坐标数、可辨别的预测模式数。单个AR2过程需要两个存储坐标；两个AR1之和则可以改写为单个ARMA21过程。这些对象不能互换命名。

由于候选状态完全由同一X历史和算法随机数得到，若条件已经包含全部X历史，它不能再提供历史之外的新外部信息。本轮采用相对受限单因素摘要的样本外NLL增量，不把它冒称严格的条件互信息或因果独立性证明。

## 2. 系统与规模

| 类别 | 真实生成系统 | 要检验的边界 |
|---|---|---|
| A | 单个AR1因素＋观测噪声 | 错误新增 |
| B | 两个独立AR1因素之和＋观测噪声 | 额外预测模式 |
| C | 两个高相关AR1因素之和＋观测噪声 | 相关不等于重复；独立规范表示可能失配 |
| D | 单个振荡AR2因素＋观测噪声 | 长记忆不等于第二个物理因素 |
| E | 双AR1与单ARMA21的精确观测等价对 | 必须拒绝唯一物理身份判断 |

每类4个单因素变化条件，20种子4011–4030，共400个系统／种子组合。每个组合分别拟合长度1000和5000，得到800个模型组合。1000为同一轨迹的前缀，不能当独立复现。两批独立测试使用同一冻结模型，并不是两批独立训练。

每组合：R训练20条、D独立训练20条、选择10条、独立确认A/B各10条、独立测试A/B各20条，全部轨迹长5000。最终真值审计另20条，10条拟合读出、10条评价。两批预测指标各80000行（包括五方法、两长度、20条测试轨迹），不意味着80000个独立训练种子。完整参数见SYSTEMS.json与PROTOCOL.md。

## 3. 概率模型与公平性

工作族包括无潜在状态的观测AR1、单因素AR1／AR2／AR3、双独立AR1模式。潜在族均采用X=Cz+噪声，不再增加可与潜在记忆相互补偿的自由观测AR系数。采用自身训练X的RMS归一化并将预测密度与状态区间还原至原单位。C固定贡献坐标规范并不代表已识别未知物理量纲。

候选族的每次EM起始预算相同，但递进另有后验抽样提案开销，本轮未严格匹配总计算时间，也未证明数据效率优势。每个直接候选最多3初值×80次EM。递进线从选择的单因素完整FFBS后验路径（每提案2条、3组提案）形成双模式初值，再用X联合重估；无有效提案时保留缺失，不借用直接双模式冒充发现。不同路线均保留完整条件协方差和高斯混合预测；参数采用点估计，未积分全部参数不确定性。FFBS平滑只用于训练阶段的候选提案，正式测试使用因果滤波，未使用未来测试X修正当前预测。

| 方法 | 主要提取线 | 独立见证线 | 200点前缀后的更新 |
|---|---|---|---|
| M1 | 完整直接R | 直接D | 停止测量更新（消融） |
| M2 | 递进R | 直接D | 停止测量更新（消融） |
| M3 | 完整直接R | 直接D | 每到达一个X更新 |
| M4 | 递进R | 直接D | 每到达一个X更新 |
| Xhistory | 原始X的AR40／200 | 无 | 使用当前完整有限窗口 |

M3是主要公平直接对照。R与D训练数据互相独立；递进R与直接R共享单因素拟合，是配对架构对照。固定参数下的选择集惩罚预测似然用于解释权重，不是积分参数后的完整贝叶斯模型证据。两条路线对同一个X的似然不能当独立物理证据相乘。本实现采用规范混合，而非重复相乘。不同维度与不同路线的状态后验保留在各自候选空间中，以模型解释权重构成混合；没有声称两条路线的潜在坐标天然一一对应，也没有建立跨路线的物理联合后验。

在线候选状态先通过转移模型预测，再由新到达X进行Kalman更新；模型解释权重也按逐点预测似然更新。预测在看到未来X之前发出。无更新冻结隐状态测量滤波与解释权重，但不会遮蔽已经观测到的当前X，观测AR候选仍以当前X发出预测。无更新组仅为消融，不作为“递进优于强基线”的主要依据。

## 4. 主要概率预测对比

正增量=右侧改进方法NLL更低；主要单位nats／步。recursive_gain=M3−M4；direct_update=M1−M3；recursive_update=M2−M4；interaction=(M2−M4)−(M1−M3)。history_gain=Xhistory−M4为次要参照。

{table(maint)}

95%区间按训练种子块及轨迹进行10000次分层bootstrap。主四对照分别Holm4，20条件主对照Holm80；history_gain为未纳入主校正的次要参照。显著性检验针对实用门槛0.005，而不只是针对零。整体改善不允许替代跨条件稳定性。

{table(agg)}

NLL按未来20步的完整联合协方差计算；CRPS与覆盖在1／2／5／10／20步边际上汇总。预测覆盖评价针对X，不等于隐藏物理状态覆盖。

## 5. 独立确认与错误扩展

候选相对选择集最佳单因素AR1／AR2／AR3记忆解释，需要R递进与D直接见证在确认A/B四次检验中均改善至少0.005，经3200项Holm校正通过；另要求极点间距≥0.05、跨路线最大极点差≤0.05及响应相关≥0.95。控制当前X和单因素预测后的功能相关仅作诊断，不冒充独立变量识别。

{table(c5000)}

确认集上的平均NLL增量如下（仅有效候选计算均值；所有种子仍计入上方支持率分母，缺失提案不算通过）：

{table(gain_table)}

单因素记忆参照并不是一个没有动态容量的标量：它可以含2／3个Markov存储坐标。下面是选择集选中的原始解释数量；递进候选需要超越这一已有记忆解释，而不是只超越当前X。

{table(parent_table)}

因此确认未通过，不能解释成B类真实系统没有两个因素；它表示本轮未证明候选比强单因素记忆解释多提供了达到门槛的预测信息。

A、D上的支持属于错误新增预测解释风险；B上的支持属于两模式预测证据。即使B支持率高，也无法区分双AR1与单ARMA21的物理身份。20次试验的Wilson区间较宽，观察到≤1/20不能证明总体错误率≤5%。等价E的拒判依据是普遍适用的代数证书，而非一个训练出来的可识别性分类器；本轮未证明能自动识别所有未知等价结构。

## 6. 隐藏状态概率诊断

状态汇总只对有效的族条件后验计算，并列出available_cases／total_cases；缺失候选的原始行完整保留，不作为成功或覆盖达标。下面是**已知真实工作族后的条件审计**：A选factor1，B/C/E选factor2，D选memory2第一坐标。真实族仅用于模型冻结后的评价，不是模型自动选择出物理族的准确率。贡献坐标归一化约定也不能保证现实物理身份唯一。oracle为已知真实参数的Kalman滤波，仅用于信息参考上界；它仍受单变量可观测信息限制。

{table(qa[[k for k in ['family','line','corr','MSE','NLL','CRPS','joint_state_NLL'] if k in qa]])}

{table(qa[['family','line','cover90','joint_cover90','width90','posterior_corr','available_cases','total_cases']])}

C的真实创新高度相关，而工作双模式族使用独立创新；相同X预测分布可以对应不同内部状态后验，因而C出现预测好、物理状态区间不准完全可能。E表中的U/V是双AR1解释下的诊断，不能据此排除单ARMA解释。

独立真值审计读出：对选择集选中的工作族，用“当前X＋后验均值”拟合线性映射；只在独立审计前10条拟合、后10条评价。它衡量可恢复信息子空间，不是重新拟合真实物理后验，也不参与预测或确认。

{table(ra)}

## 7. 观测等价性与不确定性边界

对两个AR1的a,b,qu,qv，定义phi1=a+b，phi2=−ab；g0=qu(1+b²)+qv(1+a²)，g1=−(bqu+aqv)；v=(g0+sqrt(g0²−4g1²))/2，theta=g1/v。单个W的ARMA(2,1)具有相同的X自协方差、Kalman预测分布和每点观测似然。此证书对任一合法双AR1工作解释均适用，不只对E类。

本轮等价审计的最大自协方差误差为 {eq.autocov_error.max():.3g}，最大逐点对数似然误差为 {eq.loglik_error.max():.3g}，属于浮点精度范围。相同映射先验下，精确等价解释的后验赔率保持先验赔率；等先验示例为1/2、1/2。本轮验证的是参数对应关系与似然同一，没有声称完成了参数积分的贝叶斯证据计算。

C还存在更直接的反例：固定两实极点F，将真实P换成diag(P·1)，相应Q=P−FPFᵀ，能保留全部X自协方差，但隐藏状态协方差不同。因此，仅凭预测功能重复出现不能认定物理创新独立。

## 8. 计算与完整性

保留全部失败提案、未收敛拟合和负收益。递进有效提案：{int(ch.proposal_valid.sum())}/{len(ch)}（两长度分别训练）。重复记录中的EM收敛率为 {md[md.kind!='obs_ar'].converged.mean():.2%}；直接与递进共享单因素模型，因此该比例按保存的路线记录计数，并非独立优化次数。到80次上限不等于已证明收敛；多初值也不保证全局最优。详见optimization_summary.csv。

AUDIT_COMPLETE.json核验了冻结文件、时间顺序、800个模型的稳定性／协方差、代表条件的概率归一、未来扰动不改变过去预测、先发一步预测密度与下一步创新似然一致，以及80组测试分数重放。原始X由固定种子和生成器可重建；输入哈希保存在每组合及测试冻结记录中。额外rotation_audit.csv核验可逆坐标变化保持X预测与似然，却能改变潜在转移矩阵的非对角连接；它是坐标不唯一性的控制，不是因果边恢复。state_probability_extra.csv补充全部条件的高斯状态CRPS及完整联合状态NLL。

## 9. 图表与逐图解释
''']
 explanations=[
 ('01_counts','工作因素数和存储维数','横轴是评价端的20个真实条件，纵轴分别是工作族因素数和Markov存储维数的预测权重平均。两条线为完整直接在线M3和递进双线在线M4。上图只能描述有限候选族倾向，下图描述记忆容量；D类若需要两个存储坐标，并不能宣布发现第二个物理原因。权重接近整数也不等于真实因素数已确定。'),
 ('02_main_gain','递进相对强对照的真实增量','点为5000长度下所有条件平均NLL增量，横线为分层bootstrap95%区间；A/B是独立测试轨迹。右侧为正代表递进更好。坐标局部放大显示微小增量及区间，预先冻结的＋0.005实用门槛远超图示范围，红字单独注明。左图才是架构的主要公平比较；右图与有限阶原始历史AR比较，不能替代与完整直接状态模型的比较。'),
 ('03_confirmation','新增预测证据是否重复出现','每柱分母20个独立训练种子。只有双路线、双确认块、校正检验和极点匹配都通过才计一次支持。1000与5000是嵌套数据长度，不能当两次独立复现。A和D柱升高意味着假新增风险，B柱升高意味着额外预测模式证据；C、E不能仅凭柱高命名真实独立因素。'),
 ('04_forecast_quality','预测时距和概率区间','左图用不同预测步数组成的联合NLL按步归一化，越低越好；右图同时显示90%区间覆盖与宽度。覆盖接近90%且宽度较小更有意义，不能只扩大区间换覆盖。无更新组长时间停止滤波，本来就可能显著变差，它不能证明递进结构优于直接结构。'),
 ('05_state_curves','只看X时的状态均值和不确定性','固定B1／4011／测试A第一条轨迹，未按效果挑选。黑线是真值，彩线与阴影是拟合双AR1族下的条件均值及90%边际区间；若递进提案缺失，绘图退回直接R并在图的生成代码中保留此逻辑。曲线相似只证明该族条件下恢复了一部分动态信息，单次落在区间内也不能代替整体覆盖评价。'),
 ('06_joint_posterior','两个状态的联合不确定性','图5同一示例第350步的二维90%高斯椭圆。椭圆倾斜说明观测到二者之和后，U/V条件后验会相互依赖，即使工作过程创新独立。红叉为评价真值，圆点为后验均值；此图强调保留联合协方差，不能把两个边际点估计当独立证据。'),
 ('07_long_memory','一个因素可以需要两个记忆坐标','D0／4011的真实因素是一个AR2过程。上图展示单因素AR2后验对U的跟踪；下图展示单因素AR2和双AR1的模型解释权重。模型需要保存U当前与过去值，不表示现实中有两个不同原因。振荡根也可能使双实AR1拟合不足，因此必须保留长记忆强对照。'),
 ('08_matching','不同数据训练后的功能复现','横轴为独立R／D训练路线的最大极点差，越小表示响应时标越接近；纵轴为控制当前X及单因素预测后的两条增量预测残差相关。高相关和低极点差支持预测功能复现，但不证明物理状态身份，也不把同一X的不同窗口当独立传感器。'),
 ('09_state_quality','真实族条件后的恢复与覆盖','热图汇总测试A的状态相关和边际覆盖，平均跨4条件、20种子及目标坐标。oracle使用真实生成参数，其余是拟合模型。真实族由评价端选择，因此这不是自动因素数识别准确率。C的独立创新工作假设可能给出与真实物理后验不同的区间，预测好并不能免除这一问题。'),
 ('10_readout','信息子空间能否被独立读出','只用独立审计轨迹前半拟合线性读出、后半评价R²；柱高表示所选工作表示中有可用于恢复真状态的信息。它允许混合、缩放与符号差异，不要求某个坐标就是U或V。与当前X比较可看历史状态表示的帮助，但读出监督并未提供给原模型，不算完成了无监督物理身份识别。'),
 ('11_equivalence','不同物理解释可以预测完全相同','E1／4011同一X可同时被解释为两个独立AR1因素或一个ARMA21因素。下图两条未来预测均值重合，协方差和逐点似然也经数值审核相同。这里没有通过加大噪声制造困难，而是精确观测等价；更长数据不能消除这种身份不确定性。'),
 ('12_correlated_alias','相同X不保证相同隐藏状态后验','固定C0／4011，两种内部创新结构产生相同完整X分布，但U/V的条件均值与区间不同。黑线是真实相关状态，两种彩线各自在其结构假设下成立。图中差异说明：模型可以拥有可靠X预测，却无法仅凭X决定哪一套内部状态解释是真实物理结构。'),
 ('13_length','递进增量是否跨条件稳定','每条件给出1000／5000长度下M3−M4及95%区间。正值表示递进NLL更好；是否超过0.005和多重校正应看contrasts.csv，不能只挑正向条件。两长度共享原始轨迹前缀，图仅用于数据长度敏感性；测试B的独立复现另外保留在数据表中。')]
 for num,(name,title,desc) in enumerate(explanations,1):lines.append(f'### 图{num}：{title}\n\n![图{num}](figures/{name}.png)\n\n{desc}\n')
 lines.append('''## 10. 下一步边界

20个条件是单因素参数变体，并非完整参数空间扫描，未覆盖非线性、非平稳与联合极端条件。本轮结果针对当前后验提案式递进原型、有限模型族和预算，未证明所有递进方法都不可能获益。本轮将预测模式支持与物理身份识别分开报告。若统计参考门槛未通过，应先处理候选证据强度与模型失配；即使通过，精确等价仍阻止唯一物理身份结论。在只有X且没有额外可识别约束／可控干预的设定下，不能直接把潜在模式连成真实上游因果链，更不能假定尚未影响X的上游突变已经被看见。

后续研究应明确可检验附加假设与等价类，并在独立实验中验证。此次没有启动ARD、Split-Merge、PCMCI+、SCM、因果强度估计或失效预警；没有将本轮的模型族修改归因于Yang方法。出版商及Crossref访问均为403，仅保留用户提供的思想来源，不声称核验／复现原论文。

## 文件索引与复现

- PROTOCOL.md／SYSTEMS.json：冻结协议与参数。
- CODE_REGISTERED.json、MODELS_FROZEN.json、DECISIONS_FROZEN.json、TEST_A/B_FROZEN.json：顺序与哈希。
- results/test_A/B.csv.gz：所有预测方法逐轨迹指标；family_A/B.csv.gz：单候选对照。
- results/confirmation_tests.csv、confirmation_checks.csv：全部确认及失败记录。
- results/state_quality.csv、state_readout.csv：真值审计。
- results/equivalence_audit.csv、online_audit.csv、replay_audit.csv、stability_audit.csv：数值审核。
- results/各组合/model_1000/5000.json：完整参数、协方差、最佳初值的EM轨迹、各初值终值似然与权重依据。
- RUN.md：完整从头运行与只复核现有模型的方式。
''')
 text='\n\n'.join(lines);(ROOT/'REPORT.md').write_text(text)
 import markdown
 from weasyprint import HTML
 body=markdown.markdown(text,extensions=['tables','fenced_code']);css='''@font-face{font-family:Noto;src:url(fonts/NotoSansCJKsc-Regular.otf)}body{font-family:Noto,sans-serif;font-size:12px;line-height:1.7;max-width:1050px;margin:35px auto;color:#17212f}h1,h2,h3{color:#12354a}table{border-collapse:collapse;font-size:10px;width:100%;margin:18px 0}td,th{border:1px solid #ccc;padding:5px}img{max-width:100%}h2,h3{break-after:avoid}tr{break-inside:avoid}@page{size:A4;margin:15mm}''';doc=f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>E4-A实验报告</title><style>{css}</style><body>{body}</body></html>';(ROOT/'REPORT.html').write_text(doc);HTML(string=doc,base_url=str(ROOT)).write_pdf(ROOT/'REPORT.pdf');print(json.dumps(stats,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

import json,numpy as np,pandas as pd
from settings import *

def table(df):
 rows=['|'+'|'.join(df.columns)+'|','|'+'|'.join(['---']*len(df.columns))+'|'];rows+=['|'+'|'.join(str(x) for x in row)+'|' for row in df.itertuples(index=False,name=None)];return '\n'.join(rows)

def main():
 con=json.loads((OUT/'FINAL_CONCLUSION.json').read_text());O=pd.read_csv(OUT/'contrasts_overall.csv');S=pd.read_csv(OUT/'summary.csv');C=pd.read_csv(OUT/'contrasts_condition.csv');R=pd.read_csv(OUT/'state_readout.csv.gz');IC=pd.read_csv(OUT/'internal_calibration.csv');OP=pd.read_csv(OUT/'operational_states.csv.gz');au=json.loads((ROOT/'AUDIT.json').read_text());TM=pd.read_csv(OUT/'timing.csv');names={'update_direct':'直接路线在线更新','update_recursive':'递进路线在线更新','recursive_no_update':'无更新时递进贡献','recursive_update':'有更新时递进贡献','interaction':'递进×更新交互','structure_direct':'直接结构反馈','structure_recursive':'递进结构反馈','vs_strong_direct':'相对滚动直接强参照'};gn={'A':'直接不更新','B':'直接在线更新','C':'递进不更新','D':'递进在线更新','E':'直接固定结构','F':'递进固定结构','G':'滚动直接强参照'}
 p=O[O.n==100];main=p[['batch','contrast','gain','bootstrap_low95','bootstrap_high95','p_holm8','practical_success']].copy();main.columns=['测试批','比较','收益','95%下界','95%上界','Holm校正p','实用通过'];main['比较']=main['比较'].map(names)
 for key in ['收益','95%下界','95%上界']:main[key]=main[key].map(lambda v:f'{v:.6f}')
 main['Holm校正p']=main['Holm校正p'].map(lambda v:f'{v:.3g}');main['实用通过']=main['实用通过'].map({True:'是',False:'否'});metric=S[S.n==100].groupby(['batch','group'])[['NLL','CRPS','Energy','Variogram','cover90','expected_K']].mean().reset_index();metric['group']=metric['group'].map(gn);metric.columns=['测试批','组别','20步联合NLL/步','边际CRPS','Energy','Variogram','边际90%覆盖','后验平均维数']
 for key in metric.columns[2:]:metric[key]=metric[key].map(lambda v:f'{v:.6f}')
 heading='支持本轮预设的递进与更新额外协同标准' if con['unique_synergy_demonstrated'] else '没有通过本轮预设的递进与在线更新独特协同标准'
 rows=[]
 for label in ['update_direct','update_recursive','recursive_update','interaction','structure_recursive','vs_strong_direct']:
  z=p[p.contrast==label];rows.append({'问题/机制':names[label],'两批均实用通过':'是' if z.practical_success.all() else '否','测试A收益':f'{z[z.batch=="A"].gain.iloc[0]:.6f}','测试B收益':f'{z[z.batch=="B"].gain.iloc[0]:.6f}'})
 doc=[];add=doc.append
 add(f'''# E3：X-only 概率状态发现、双线见证与在线结构概率更新

日期：2026年11月23日。**{heading}。** 这是有限候选图下的线性高斯概率原型，包含非线性及非平稳失配测试；不运行E4因果失效预警。

本轮先冻结协议，再冻结实现。共17个独立X条件×20个训练种子，嵌套20/50/100/200/400条训练规模，1700个训练组合，每组合三套独立/对照候选库、各K=0..3，合计20400个状态空间候选。两批测试各13600条独立轨迹，模型与规则在测试前冻结。相同轨迹的多模型、多规模及重叠窗口评分不是额外独立样本。

{table(pd.DataFrame(rows))}

收益均为对应旧方案NLL减新方案NLL，正值支持该机制；单位nats/步。主要判据为未来20步联合NLL/步，最小实用收益0.005，主规模100条/路线。实用通过要求95%区间下界超过0.005且Holm校正通过；“没有通过”不自动等于证明无差异。协同成功还必须同时满足有更新时递进贡献、交互项及相对滚动直接参照收益，两批复现，并满足错误扩展约束。
''')
 add('''## 本轮完成内容与边界

|阶段|实际实现|边界|
|---|---|---|
|E3.1 概率初始发现|X-only EM状态空间模型，完整均值、协方差和跨时刻协方差|未知维数只在统一上限3内，不是无限开放发现|
|E3.2 双线发现|从概率后验路径递进提案；另用独立训练池从原始X拟合见证库|两路线共享观测类型，一致性不是独立物理证据|
|E3.3 在线更新|每步预测、评分、更新；联合未来概率积分|得到模型条件后验，不是未知真实系统的精确后验|
|E3.4 概率反馈|后验不确定性参与训练提案，确认图内维数和路线权重在线变化|测试不新拟合；物理增加、拆分和合并身份未唯一识别|
|E3.5 可识别性|无隐藏、单隐藏、串联、并联、共同驱动及旋转等价反例|等价检查是构造性拒判，不是通用因果图可识别性分类器|
|E3.6 公平比较|完整2×2、固定结构E/F、滚动直接G、20种子和两批测试|两批共享冻结训练模型；第二批不是新增训练种子|

没有把统计候选直接命名为真实Y、Z，也没有给出缺乏识别依据的因果边准确率或提前预警率。
''')
 add('''## 数学模型与数据规则

候选模型为 z_t=Fz_(t−1)+η_t，X_t=aX_(t−1)+Cz_t+ε_t。F、Q、C、a、R只从X学习；滤波q_t(z)=N(m_t,P_t)，P_t保留非对角相关。不同路线、不同K的解释组成q_t(路线,K,z)，未来20步对潜在状态积分，得到完整联合高斯混合预测。

单步新X到达时按各解释的更新前观测似然调整权重，再更新连续状态。候选切换概率固定0.01，属于动态模型平均近似，不宣称任意切换机制的精确Bayes后验。两路线解释作混合，不将同一个X的两份似然相乘。

递进路线先拟合概率状态，用后验协方差和未充分解释动态选择探索父状态，抽取2条完整FFBS后验路径，从其20步历史提出下一方向，再进行5次EM联合估计。直接路线用完整40步原始X历史初始化，随后同样5次EM。递进主要改变提案与初始化，最终概率模型族统一；若有收益，也要区分搜索/优化归纳偏置与新物理信息。

主训练池R与见证池D严格独立。直接对照另在R训练，所以直接和递进主线使用相同数据；所有组都使用独立D见证，不给递进组额外独占训练数据。每路线400条训练池，选择40、确认40；每规模等量取R、D前20/50/100/200/400。所有轨迹长度160。

初始K由选择集评分、复杂度惩罚和单标准误差规则决定。独立确认集用双路线预测非劣筛候选图；明确新增信息需两路线相邻维数预测收益都达到0.005并通过统一Holm校正。概率不确定候选与已确认新增证据分开登记。测试只在冻结图内推断，不用测试评分重新拟合或调门槛。

每时刻在线更新，未来联合预测在40、50、…、130发布，分别评价未来1/2/5/10/20步。原点重叠只在轨迹内平均。统计以20个种子为独立块，置信区间用10000次种子/轨迹分层bootstrap。完整协议、精确方程及实现细节见 PROTOCOL.md、BASE_SYSTEMS.json、IMPLEMENTATION_NOTES.md。协议文字列出8个对照却误计为7，开测前按完整列表更正为Holm8、逐条件Holm136，保留原协议及COUNT_CLARIFICATION.json。
''')
 add('''## 七组比较如何理解

A/C的无更新组仍保留概率分布，并使用当前X的观测AR项，但40步前缀后不再做测量后验更新。它们是机制消融，状态误差累积不能作为新方法创新证据。

B/D分别开启直接/递进模型的状态与结构概率更新；E/F也在线更新状态和路线权重，但K固定为独立选择结果。G每个预测起点使用最近40个滞后加当前X重建直接概率状态，固定初始K、两路线等权，拥有最新观测信息。D必须对B、G也有收益，不能只胜过A/C。

交互项I=(C−D)−(A−B)。即使I为正，也可能因为C的无更新起点更差；因此必须与B−D、G−D共同判断，而非单独宣布双线协同。
''')
 add('## 主规模全部对照\n\n'+table(main)+'\n\n逐条件校正与全部规模结果在 results/contrasts_condition.csv、contrasts_overall.csv、contrasts_seed.csv。')
 add('## 概率评分原始数值\n\n'+table(metric)+'\n\nCRPS与覆盖在1/2/5/10/20步边际平均；Energy使用两套各16个联合样本的配对估计；Variogram对五个时距间10对做有限样本近似，受Monte Carlo误差影响。辅助评分不是新的成功指标；CRPS、Energy原始单位不同条件量级不同，应结合逐条件表解释。')
 captions=[
 ('机制贡献分开检验','八个差值分别回答更新、递进、结构反馈和交互。误差线为预注册的分层95%区间，红线为0.005。每个机制须看两批是否复现及Holm8，不能把左侧的普通滤波收益自动归于递进发现。'),
 ('所有条件的架构及强基线比较','左图为B−D，右图为G−D，正值才支持递进在线方案。所有17条件均保留，含负收益。普通逐条件95%区间不是同时区间，局部实用成功还需Holm136；不从最好条件推断总体有效。'),
 ('小样本学习曲线','左图比较有更新模型相对滚动直接参照，右图展示更新和交互收益随训练规模变化。横轴是每路线轨迹数，总训练预算为两独立路线的两倍。嵌套规模相关，曲线平台或交叉不等于额外独立复现，也未证明20条以下的数据效率。'),
 ('预测时距和观测概率校准','联合1/2/5/10/20步评分与90%边际覆盖分开显示。短期有收益不保证远期也有，覆盖接近90%也不保证尾部或物理隐藏状态正确。混合边际覆盖用实际观测的混合CDF检验，不用16个样本粗糙估计区间。'),
 ('固定轨迹上的完整概率状态','事先固定并联强组、种子3011、测试A第一条轨迹。第一行只展示观测X；第二行是四种候选维数后验质量；第三行是预先按组件顺序选取的连续状态均值与90%区间。阴影来自协方差，不是人为误差棒。坐标的符号、尺度和混合均不具有物理变量身份。'),
 ('状态数、过扩展与不确定性','候选维数的后验期望不是自动发现真实变量数。右图特别保留无隐藏/单隐藏系统上后验MAP维数超过真值代理、且概率≥0.8的时间比例；它与独立确认的新增错误率不同。延迟条件真隐藏维数4而候选上限3，包含预设容量失配。'),
 ('隐藏状态质量的两种证据','前两图是在全部X预测冻结后，额外独立40条真值轨迹拟合读出、另一40条评价的校准读出分数及覆盖。读出加入单独估计的残差，不等于原模型直接给出的真实Y/Z后验。右图为各已拟合模型自模拟的条件校准，仅检验模型内滤波概率实现；即使接近90%，也不解决真实系统的状态身份问题。'),
 ('观测等价机制保持拒判','对固定候选做可逆正交状态旋转，F、C、Q、P同时变换。新坐标的隐藏图可以不同，X似然完全相同，等先验解释的后验应保持1/2。此构造允许相关创新，不能外推到具有额外独立创新假设的所有模型。它说明预测和概率更新不能独自消除观测等价性。'),
 ('单因素压力条件','从同一个并联中等条件分别只改变持续性、观测创新噪声、传递缓存、创新相关、观测非线性或作用强度渐变。图中保留线性概率族在非线性/非平稳生成器上的失配结果。渐变条件不用于预警时刻结论，不把误差变化直接命名为因果失效报警。'),
 ('独立原始X见证','每点为一个独立种子的确认集第二方向增量，横轴递进主线、纵轴独立训练原始X见证。两轴都达到实用门槛仍需多重检验；只有递进一轴有收益不能确认新状态。两路线参数和训练池独立，但都来自同一类型X，不能把一致性升级为独立物理因果证据。')]
 for i,(title,caption) in enumerate(captions,1):add(f'## 图{i}：{title}\n\n![{title}](results/figures/{i:02}.png)\n\n{caption}\n')
 add('## 错误扩展与维数代理\n\n'+table(pd.DataFrame(con['false_expansion_diagnostics']).rename(columns={'condition':'条件','direct_errors':'直接确认错误','recursive_errors':'递进确认错误','seeds':'种子数','excess':'递进多出次数'}))+'\n\n这些是强新增证据的确认次数；概率候选的不确定维数还需看图6和operational_states.csv.gz，不能用确认0次隐藏在线过扩展。20次仍不足以证明真实错误率为零或方法间严格相等。')
 op=OP.copy();op['condition']=op.case.str.rsplit('_',n=1).str[0];z=op[(op.batch=='A')&op.condition.isin(['S0','S1'])].groupby(['condition','group'])[['expected_K','overextended_fraction','confident_overextended_fraction','uncertain_fraction']].mean().reset_index();z.columns=['条件','组','期望维数','MAP过扩展比例','高置信过扩展比例','低置信时间比例']
 for col in z.columns[2:]:z[col]=z[col].map(lambda v:f'{v:.4f}')
 add(table(z))
 rr=R.groupby('group')[['readout_gain','readout_CRPS','readout_R2','marginal_cover90','moment_ellipsoid_cover90']].mean().reset_index();rr.columns=['组','读出NLL收益/维','读出CRPS','读出R2','边际90%覆盖','矩匹配椭球90%覆盖']
 for col in rr.columns[1:]:rr[col]=rr[col].map(lambda v:f'{v:.5f}')
 add('''## 真状态审计的正确解释

真实Y/Z的直接后验评分：未定义，因为没有识别唯一对应坐标。下面是独立校准读出代理；读出矩阵只用额外审计轨迹拟合，不使用正式测试真值、不回写预测模型。其残差噪声单独记录，因此覆盖好不代表原始潜在后验直接识别了物理状态。联合覆盖是矩匹配椭球诊断，不是混合后验精确HPD区域。

'''+table(rr))
 it=IC.groupby('bank')[['marginal_cover90','joint_cover90']].mean().reset_index();it['bank']=it.bank.map({'direct':'直接主线','recursive':'递进主线','witness':'独立原始X见证'});it.columns=['候选库','模型内边际90%覆盖','模型内联合90%覆盖'];add('## 模型内条件概率校准\n\n'+table(it)+'\n\n这是从已拟合模型自身模拟状态后，条件于固定参数的校准；没有包含参数估计不确定性，也没有证明未知真实系统中的Y/Z后验校准。')
 add(f'''## 审计、计算与归档

所有冻结散列已复核。重新生成{au['input_cases_regenerated']}个输入组合，数值重放覆盖{au['prefix_and_replay_case_batches']}个固定条件/批次。最大未来扰动前缀误差{au['max_prefix_error']:.3g}，最大评分重放差{au['max_score_replay_error']:.3g}，权重归一化误差{au['max_weight_normalization_error']:.3g}。状态旋转后预测/似然误差{au['rotation_equivalence_error']:.3g}，等价解释权重差{au['equivalent_explanation_weight_error']:.3g}。

有限5次EM且带协方差下限、稳定性投影，不保证全局最优；审计记录{au['nonmonotone_EM_fits_due_to_floors_and_stability_projection']}个出现轻微非单调对数似然的拟合，不能据此把比较当模型族全局最优定理。主规模100每组合平均训练CPU {TM[TM.n==100].cpu.mean():.3f}秒；全部规模、进程峰值RSS见timing.csv。峰值RSS是进程累计值，不是单模型独占内存。

每个训练组合保存三套K0..3模型、选择评分、后验提案日志及确认图。两批全部逐轨迹指标在压缩CSV。仅每条件种子3011、规模100的首条测试轨迹保存完整代表后验均值、协方差和模型权重，不宣称所有轨迹的完整概率数组均归档。数据由固定生成器按阶段独立重建，不重复提交原始大数组。

原始附件、协议、补充计数澄清、源码、冻结记录、模型、图表、审计均随目录提供。复现方式见RUN.md。原始E1–E2.5结果保留，新实验使用独立分支。

## 回答研究目标

本轮实际检验了模型定义的概率状态解释、新观测反馈、有限图内结构不确定性，以及递进相对完整直接概率方法的增量价值。是否通过以开头的配对判据为准；普通滤波有效不能自动说明递进发现有独特价值。

概率分布让模型能够保留多种解释和不确定性，但不能创造X中尚未出现的信息，也不能自动确定真实隐藏因果链。只有在进一步增加可识别假设或额外观测/干预并验证后，才有依据进入物理因果关系和预测失效提前预警。
''')
 (ROOT/'REPORT.md').write_text('\n\n'.join(doc));print(heading)
if __name__=='__main__':main()

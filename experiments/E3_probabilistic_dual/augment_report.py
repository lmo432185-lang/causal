"""Append separately registered and exploratory diagnostics, preserving main claims."""
import json,pandas as pd
from settings import *
from make_report import table

def main():
 p=ROOT/'REPORT.md';s=p.read_text();marker='## 追加独立训练种子与收敛诊断'
 if marker in s:raise RuntimeError('Regenerate base report before augmentation')
 R=pd.read_csv(OUT/'replication_contrasts.csv');Q=pd.read_csv(OUT/'crossline_fresh_check.csv');assessment=json.loads((OUT/'STUDY_ASSESSMENT.json').read_text());s=s.replace('**支持本轮预设的递进与更新额外协同标准。**','**固定5次EM的预测协同通过，可靠隐藏状态发现尚未通过。**');intro='''

**完整结论需要同时看预测收益和状态可靠性。** 固定5次EM的正向收益在新增10个独立训练种子中复现，主规模合计30个种子。但延长到25次EM后，额外交互收益低于预设实用门槛；单隐藏系统存在维数过扩展，真状态读出区间偏宽。因此本轮完成并检验了概率原型，没有证明恢复唯一、可靠的真实隐藏状态或因果链。
''';s=s.replace('## 本轮完成内容与边界',intro+'\n## 本轮完成内容与边界',1)
 names={'recursive_update':'递进在线相对直接在线','interaction':'递进×更新交互','vs_strong_direct':'相对滚动直接'};z=R[R.analysis.str.startswith('new10')&R.contrast.isin(names)].copy();z['analysis']=z['analysis'].str.replace('new10_em25_','25次EM·测试',regex=False).str.replace('new10_em5_','5次EM·测试',regex=False);z['contrast']=z.contrast.map(names);z=z[['analysis','contrast','gain','bootstrap_low95','bootstrap_high95','practical_success']];z.columns=['独立新10种子','比较','收益','95%下界','95%上界','实用通过']
 for k in ['收益','95%下界','95%上界']:z[k]=z[k].map(lambda v:f'{v:.6f}')
 z['实用通过']=z['实用通过'].map({True:'是',False:'否'});combined=R[R.analysis.str.startswith('combined30')&R.contrast.isin(['recursive_update','interaction'])][['analysis','contrast','gain','bootstrap_low95','bootstrap_high95']].copy();combined['contrast']=combined.contrast.map(names);combined['analysis']=combined.analysis.str.replace('combined30_em5_','合计30·5次EM·测试',regex=False);combined.columns=['合计30种子描述','比较','收益','95%下界','95%上界']
 for k in ['收益','95%下界','95%上界']:combined[k]=combined[k].map(lambda v:f'{v:.6f}')
 new='''## 追加独立训练种子与收敛诊断

此部分在原20种子第一批正向信号后另行登记，属于后续复核，不能冒称原始预注册内容。保留全部17条件、只复核原主规模100，新增种子3031..3040。EM5严格复现原流程；EM25从同一EM5候选出发，对直接、递进、见证全部额外做20次EM，再用独立选择/确认数据确定图，之后冻结并测试。两种预算使用相同新增X轨迹，规则和阈值不变。

'''+table(z)+'''

5次EM下，递进在线相对直接在线的收益约0.026，交互约0.0073，新增10种子两批均复现。25次EM下，递进相对直接仍有约0.0109的收益，**不能写成优势完全消失**；但交互约0.0039，95%区间整体低于0.005，额外协同的实用判据不再通过。直接模型延长优化的改善大于递进模型，递进相对差距缩小约58%。

这支持“有限优化预算下的概率递进初始化价值”，尚不支持“直接模型原则上无法取得同样信息”。当前直接基线只采用一个CVA初值，25次EM仍非全局收敛证明，尚未运行多起点、按实际CPU严格匹配的最优直接搜索。不能只凭迭代次数推断计算效率，也不能把初始化优势解释成新物理变量发现。

合计30种子的综合描述如下；独立新增10种子的表才是后续复核证据，不能用合并平均掩盖复现失败。

'''+table(combined)+'''

## 图11：独立新种子与优化预算

![独立种子与收敛诊断](results/figures/11.png)

每点是独立新增10个训练种子上的配对收益，两种测试批均展示；误差线为分层95%区间。绿色为5次EM，紫色为25次。相对直接的残余收益与交互收益是不同问题：延长优化后前者仍为正，交互则低于实用线。原20种子结果和追加结果全部保留，不只展示有利预算。

## 状态可靠性未通过的具体证据

原20种子在单隐藏条件中，递进强新增确认错误2/20，直接1/20。与此不同，在线后验的高置信维数过扩展时间比例约为递进30.1%、直接22.8%；递进平均候选维数约2.37，而生成器物理隐藏维数为1。预测表示维数与物理原因数并不等价，这些数不能当真实原因恢复成功。

在无隐藏S0中，动态模型平均的维数期望约1.47，但高置信过扩展比例为0，维数解释始终低置信；它保留了不确定性，却没有稳定收敛到唯一零状态解释。固定结构组则在S0选择了0。确认新增错误与在线多解释质量必须同时报告。

独立真值读出中，递进在线组平均R2约0.53，直接在线约0.39，说明确实含部分隐藏动态信息；但递进读出的90%区间实际覆盖约96.4%，偏保守。固定结构递进的读出覆盖约92.8%、读出NLL收益还高于动态结构递进，未证明结构反馈让真实状态解释更可靠。无物理隐藏状态的S0不计算真状态读出，未以0值冒充准确。

预注册的结构反馈F−D在两批约0.004，低于0.005，说明本轮尚未证明动态拆分/合并具有预设的实用增益。主预测协同通过不代表所有E3子目标均通过。

## 图12：新增独立X中的同一线索检验

![新X中的候选一致性](results/figures/12.png)

主确认检查两路线各自增加一维是否改善预测，这不足以保证找到同一条Wy。为此主实验结束后另行登记探索性审计：完全新阶段4的40条X拟合旧预测控制回归，阶段5的另40条X检查递进与独立见证的10步预测增量残差对应，并同时要求两路线在新X中有0.005的联合预测收益。它不修改模型、候选图或主结论。

'''+f'''指定的观测效果一致性检查仅{assessment['fresh_crossline_coherent_cases']}/{assessment['fresh_crossline_total']}通过，强并联条件为11/20，其余大多较弱。这个比例是特定探索性诊断，不是通用物理识别率。它表明本轮未稳定建立同一新候选状态的双路线对应，不能把两路都有预测增量自动当成同一隐藏变量被独立发现。

这些限制与观测等价反例一致：概率更新能调整解释可信度，但不能独自消除状态坐标混合、维数冗余和因果机制的不可识别性。

'''
 s=s.replace('## 错误扩展与维数代理',new+'\n## 错误扩展与维数代理',1);s=s.replace('每时刻在线更新，未来联合预测','弱/中/强标签沿用E2.5的生成方程，未按E3的未来20步目标重新标定0.005边界；不能把继承的中档名称当作本轮精确可检测界限。\n\n每时刻在线更新，未来联合预测',1);s=s.replace('## 七组比较如何理解','这些分布条件于已估计的固定参数；解释权重使用预测似然、复杂度先验和0.01复位，不是对参数完整积分后的模型证据。未来20步内各解释独立传播，用当前混合权重形成联合预测，没有模拟预测区间内部的模型切换。维数熵受先验与复位影响，低置信度不等于学会了通用不可识别性判别。\n\n## 七组比较如何理解',1);s=s.replace('本轮实际检验了模型定义的概率状态解释','本轮原始预测协同判据通过且新增种子复现，但长迭代敏感性、状态维数及真值读出质量使完整可靠发现目标尚未通过。\n\n本轮实际检验了模型定义的概率状态解释');p.write_text(s);print('Qualified follow-up and state limitations inserted')
if __name__=='__main__':main()

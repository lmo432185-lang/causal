# E3 数据与结果复核

入口REPORT.md / REPORT.pdf。结果分开保存：results为原20种子全学习曲线；replication/em5及em25为新增10种子主规模复核和收敛诊断。两批测试为独立轨迹，使用同一冻结训练模型。12张中文图和解释见报告。

需要Linux、Python3.12、g++和requirements.txt版本。设置单线程，避免改变数值环境：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit_replication.py
```

audit.py按固定种子重建X，不保存大数组、不重训练。native.py首次自动从native.cpp编译本地内核，编译产物不提交。内核上限3，请勿仅改settings.KMAX而不同时调整内核容量并重新验证。

需要重建图和报告：

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/e3-mpl python plot.py
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/e3-mpl python plot_followup.py
python make_report.py
python augment_report.py
python render_report.py
```

重跑拟合请使用独立副本与新登记记录，不覆盖原始冻结结果。数学模型和流程以PROTOCOL.md为主，IMPLEMENTATION_NOTES.md说明数值实现；COUNT_CLARIFICATION.json把原协议列表的8个对照计数更正为Holm8/136，未删去任何对照。REPLICATION_PROTOCOL.md及其冻结记录明确标记追加实验；crossline_check是主实验后的探索性新X审计，不冒称预注册主结果。

原始附件保存在REQUEST.txt。所有真实状态只在两批X预测冻结后用于独立审计读出；这些读出不是直接恢复真实Y/Z的后验。概率递进提案在训练阶段使用完整训练轨迹的平滑后验抽样，正式在线测试只使用当前及过去X滤波，不做新参数拟合。

补充元数据：新增复核模型的peak_RSS_kib=0是未采集的占位值，不表示零内存。EM25中的cpu保留EM5初始拟合耗时，额外继续训练耗时在continued_EM_cpu；合计需相加。报告timing.csv仅汇总原20种子实测，不用于声称两种方法逐一CPU完全相同。

每个K维SSM的名义生成参数数目为 K²+K(K+1)/2+K+2，K=0/1/2/3时为2/5/11/20。每路线完整候选库共38，预测用两路线最多76；所有主组容量上限相同，独立确认筛掉的候选数可以不同。概率权重和尺度等派生量另存，不将它们当真实物理状态参数。

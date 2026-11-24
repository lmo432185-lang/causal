# E4-A4／A5复现说明

本目录为2026-11-24实际运行档案。先读REPORT.md的限制、事故和Go／No-Go；不要把未收敛结果当可靠物理后验。

**最终范围调整**：用户随后明确要求小数据验证。5000主两批与主审计已完成；1000前缀仅完整测试A，原全量B取消，改为small_validation的6条件×3种子×5条、复用模型、不新增训练。原pipeline.py记录原计划，不能当全部已执行。实际短流程为short_audit_A_only.py、short_analysis_A_only.py及small_validation.py；USER_SMALL_SCALE_AMENDMENT.md记录此调整。后续默认先做小原型。

## 环境

Linux、Python3.12、g++支持C++17。依赖版本见requirements.txt。计算时固定OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1；主任务2核配额、每阶段3个进程。native.py与bayes.py首次导入会在源码旁编译共享库，二进制不随档案保存。

原工作路径为`/workspace/work/E4A4_joint`，解释器为`/workspace/causal-venv/bin/python`；short为其子目录。冻结源码中settings.OLD仍为`/workspace/work/E4A_probabilistic`，pipeline解释器路径也保留原值。移动位置时可用这些路径的符号链接或容器挂载，不要修改档案源文件后仍声称与原SHA一致。报告绘图优先使用同目录fonts，兼容原旧目录字体路径。

PNG中文已栅格化、PDF已嵌入中文字体，可以直接阅读。为节约磁盘，不再重复归档16MB字体文件；`fonts/FONT_SOURCE.json`提供原固定分支下载地址和SHA256，LICENSE已附。重新作图或离线HTML显示时，按该索引将字体还原到fonts目录。参数与数据不依赖该字体。

旧基础模型来自GitHub lmo432185-lang/causal分支experiment/e4a-independent-state-20261123，提交`38cda63b1ee470cdc53efa5abd86b2208028d921`，路径`experiments/E4A_independent_state/results/<case>/model_5000.json`或model_1000.json。主workflow复用5000模型；short/workflow.py用git show读取1000模型。重新拟合前将40字节提交所指对象取回，并从该提交恢复上述旧结果到settings.OLD/results。模型中old_model_sha256记录逐个来源。本轮模型已内含需要的完整参数，重放和作图不必重新拟合旧模型。

## 重放现有结果

保留档案原样，先用`sha256sum -c MANIFEST.sha256`核验；清单自身不列入。生成器按SYSTEMS.json、固定SeedSequence(index,seed,stage)重建X，不需下载原始浮点数组。模型json.gz保留float64全部参数，逐轨迹指标保留12位有效数字。

为节约云端磁盘，较大的CSV.gz使用`compact/INDEX.json`索引的CSV.xz载荷归档。先运行`python restore_scores.py --verify-only`验证能逐字节恢复原gzip SHA，再运行`python restore_scores.py`恢复原路径，然后执行下述分析／审计。恢复不重跑实验、不修改任何数值或冻结记录；索引保留原gzip头尾、压缩flush与zlib版本及CSV本体哈希。仅查看报告、图和小样本结果不需要恢复。不同zlib若不能产生相同deflate字节，应使用INDEX记录的版本；CSV本体哈希同时独立核验数值载荷。

1. 在独立可写副本运行`audit.py`、`short/short_audit.py`，会核对冻结记录、代表性冷重放、未来扰动和精确等价证书。
2. `analysis.py`与`short/short_analysis.py`只汇总已有评分，不拟合模型。它们会写汇总与完成标记，故不要在只读原档案中执行。
3. `diagnostic_summary.py`汇总优化和MCMC诊断；`figures.py`输出15张中文图；`report.py`生成REPORT.md/html/pdf。绘图与报告脚本是测试后描述性展示，不是冻结的模型选择代码。

## 从头运行

必须使用新的输出副本，不能删除旧结果冒充未见测试。完整顺序：

1. 还原旧基础模型及字体；固定SYSTEMS、jobs、协议、代码和输入哈希。原PROTOCOL_REGISTERED／CODE_REGISTERED等记录是历史证据，不伪造新时间替代它们。
2. `run_fit.py`：480组合模型，读取原X-only训练分区；`preflight.py`验证输入、旧基线和模型。
3. `calibration_setup.py`、`bayes_fit.py`、`extra_plans.py`：仅训练和选择集建立已登记方案；原bayes_fit.py固定80任务、两链及诊断阈值。
4. `confirm.py`：全新stage10双确认块、Holm5760，冻结最终模型；`interpretation.py`记录坐标解释。
5. `evaluate.py`：首次打开stage11/12，分别冻结A/B评分、X输入哈希和示例；不能在已存在test_A/B文件时重跑覆盖。
6. `forecast_joint.py`：未来20维联合覆盖；`state_audit.py`：两批预测完成后才开放真值；`sensitivity_audit.py`、`parameter_audit.py`、`audit.py`：只读固定模型和规定数据；`analysis.py`汇总。
7. short按其pipeline执行同样扩展、确认和预测，训练／选择／确认只有1000点，测试5000点；不重复主物理概率和MCMC诊断。

pipeline.py是按完成标记续跑的原脚本；不能仅复制旧完成标记到新目录后声称重新执行。register_final.py仅用于尚未打开新测试时冻结评价源，本轮已经完成，不能再次登记以追认变动。restore_registered_evaluation.py是一次事故恢复取证程序，日常复现不用运行。

## 指标口径

主NLL是联合未来20步概率密度除以20；每条轨迹多个预测起点先平均，再配对比较。主bootstrap以20训练种子为最高独立层。H0改善≤0.005，主6／次要4／机制3／容量4分别校正，不能跨家族挑有利结论。R3/J1/D及J2/E为有意别名，非额外复现。两个测试批独立，两个训练前缀嵌套且同一测试，不能把长度当独立复制。

贡献坐标的physical_scores采用有限高斯混合精确CDF、密度和CRPS；联合覆盖为固定U后V的Rosenblatt区域，并非HPD。连续MCMC后验目标有显式先验，有限8/16样本和17/33节点仍有近似误差。任何新算法、更多迭代或不同先验都是新实验，需独立登记和新验证，不能替换本档案负结果。

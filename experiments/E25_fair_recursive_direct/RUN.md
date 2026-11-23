# 复现评分（不重新训练）

在本目录安装 requirements.txt 对应依赖，使用单线程 BLAS，恢复输入并核对冻结结果：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python restore_inputs.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit.py
```

输入约325 MiB。restore_inputs.py 不改原始协议、模型、确认结果及冻结散列；已有输入时只验证。AUDIT.json 是复核输出。不同数值库或平台可能影响逐位再现；本轮使用 Linux、Python 3.12 和 requirements.txt 固定版本。

结果入口为 REPORT.md、REPORT.pdf；各条件和全部种子指标位于 results，主比较为 overall_comparisons.csv。Delta=直接NLL−递进NLL，正值代表递进更好；不能把仅优于固定字典读作胜过完整直接模型。所有数据效率结论仅涉及100/200/400条训练规模。

需要重新拟合时请使用独立副本，先保存原始归档，移走副本中旧data/results和运行生成的冻结记录，保留预注册源文件，再按 generator.py → fit.py → confirm.py → evaluate.py 执行。新运行须有新登记时间，不能冒用本轮原始冻结记录。

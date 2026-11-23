# E4-A运行与复核

Python 3.11，Linux／g++。依赖见requirements.txt。每个进程设置OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1以避免过量并行。native.py在首次导入时编译native.cpp，不需要上传二进制。建议至少2CPU和1GB内存。

## 保留现有档案，只复核

现有冻结文件属于历史运行。不要覆盖模型或测试。audit.py只读现有参数和测试表，重建代表轨迹并验证；不调用拟合。analysis.py／figures.py／report.py重建派生统计与报告，可在档案副本运行。运行路径应包含完整results目录；仓库中为正常目录，云端工作目录为指向/tmp的符号链接。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python audit.py
python analysis.py
python figures.py
python report.py
```

## 从头重跑

先复制源码、协议、参数、fonts到一个新目录，排除results、fit_checkpoints、所有运行产物和FROZEN／COMPLETE文件；保留预注册记录作参考，给新运行重新登记时间与代码哈希。禁止在历史档案原位置重跑并覆盖结果。

```bash
python development.py
# 为新运行登记PROTOCOL_REGISTERED、CODE_REGISTERED、ANALYSIS_REGISTERED、POSTCHECK_REGISTERED
python fit.py
python confirm.py
python evaluate.py
python state_evaluate.py
python audit.py
python analysis.py
python figures.py
python report.py
```

fit.py生成400组合×2长度，共800个模型组合；每组合20训练R＋20独立训练D＋10选择＋2×10确认＋2×20测试。状态真值只在两批测试冻结之后开放。两个长度是嵌套前缀，不能作为独立种子。pipeline.py可等待拟合完成后依次执行确认、测试、真值审计和统计，但不会替代协议登记。

## 断点

checkpoint.py只按字节备份完整模型，不读取测试、不调整优化；resume_from_checkpoints.py可恢复丢失的临时results。未冻结前fit.py跳过已经完成的合法模型。冻结后恢复只核对哈希，不重新拟合。最终仓库无需附带冗余fit_checkpoints。

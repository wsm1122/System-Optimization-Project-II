# 已公开的逐条实验证据

本目录已补入 R8-B 的 12 份完整 dev 预测及对应逐条评分（每份 14,326 条）、worker 汇总、统计分析、SVG 图表和脱敏资产审计。R9 目前只公开第一份完整 test 预测（7,176 条），其余预测、逐条评分、原始逐请求日志及逐音频配对表仍未全部公开。

预测与评分使用 gzip 无损压缩；每个文件的原始、解压后和存储 SHA-256 见 PUBLISHED_MANIFEST.json。JSON 中删除 PID 和 GPU UUID，绝对路径用文件名标记替代，数值结果保留。

在仓库根目录执行：

```bash
python3 course_submission/evidence/full/verify_published.py
```

只依赖 Python 标准库，不需要 GPU、PyTorch、NumPy 或 SciPy。校验器检查清单中所有公开文件哈希、12 份 dev 预测逐字节一致、逐条评分 S/D/I/N、根据评分日志中的 lab/rec 用编辑距离重新计算错误数，并复算进程级均值及 E2-E1 的 95% t 区间。它会明确显示 R9 明细覆盖不完整、raw 日志和 paired_audio 表尚未公开。

这些是历史实验的证据复核，不是新的 GPU 推理实验。音频级和说话人级 bootstrap 结果可以在 analysis.json 查看，但缺少配对表时不能仅凭此公开子集复算。源模型、音频、Datastore、索引和中间张量不在本次上传中。

自动审批已获操作者具体授权后，仍拒绝了部分预测、原始日志和配对表的上传。相关文件没有通过其他形式绕过拦截。

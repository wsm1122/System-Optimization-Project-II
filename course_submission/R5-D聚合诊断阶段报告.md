# R5-D：GPU概率聚合逐算子诊断

日期：2026-11-22。状态：**BLOCKED，实际GPU诊断未启动；没有实施新聚合优化。**

R5-B/C均已独立完成正确性和完整dev验证。随后两次启动 `R5_ALLOW_GPU=1 R5_GPU=4 python r5/launch_r5.py --stage D --mode diagnose` 的宿主执行权限自动审查均超过截止时间；执行工具明确返回进程创建被拒绝。不是CUDA、FAISS或模型异常，不存在可汇报的新GPU数据。

证据：`r5/results/D_launch_review_timeout.json`、`D_launch_review_timeout_retry.json`。依工具规则仅重试一次，之后不再自动提升权限或改用其他设备。

已准备 `r5/runner_d.py`：计划在真实dev邻居上拆分温度除法、softmax、gather、CPU零张量分配、H2D、scatter_add、log、nan_to_num，并校验分解结果与原聚合一致；另保存CPU/CUDA profiler时间线。准备代码不等于实际执行通过。

尚不能断言哪个聚合算子是真正热点，也不能把R4已有GPU直接分配当成本轮新增优化。当前候选没有新的性能、显存或正确性结论。恢复宿主执行许可后须重新检查GPU占用再执行，任何CUDA/FAISS或正确性失败均停止对应候选。

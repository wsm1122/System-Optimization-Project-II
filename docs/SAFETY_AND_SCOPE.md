# 安全、正确性与适用范围

## Blank-aware 检索

`blank_only_process` 按帧 CTC argmax 非 blank 条件构造 Query 子集，检索这些 Query，并将返回概率放回对应帧后执行原融合函数。该方法保留邻居数、nprobe、距离定义、融合权重与温度。

启用条件：batch size 为 1、`decode_skip_blank=true`、冻结 greedy 配置，且 `scale_lmbda` 与 `recompute_dists` 关闭。超出条件会拒绝执行，而不静默改变语义。

**边界：**跳过 blank 帧意味着这些帧不触发 FAISS 搜索。因此若被跳过帧本应因 FAISS 非法结果而报错，其异常路径与原始“先检索后跳过”不同。已验证范围不应扩大解释为任意损坏索引下无条件等价。

## GPU 聚合候选

`device_allocation_aggregate` 仅改变输出聚合张量的设备分配位置；算法仍用 softmax、scatter_add 和相同的标签。此候选可独立关闭/启用，但最终冻结配置关闭它。局部聚合微基准不能替代完整推理端到端测量。

## 检索守卫

`validate_neighbors` 在标签索引/概率聚合前检查距离有限性和 ID 范围，不会过滤、填充或重检索。绕过该守卫不是本项目的优化方式。

## 不属于证明的内容

单元测试、历史 GPU 对照和 Z3 索引范围抽象均不能证明完整 FAISS、CUDA 浮点内核或 ASR 系统对所有输入正确。正式复现应记录运行环境、资产哈希、Run ID 和原始结果。

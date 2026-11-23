# kNN-CTC 检索执行优化（AISHELL-1）

本仓库沉淀 AISHELL-1 kNN-CTC 课程项目中经过验证的检索执行优化代码与实验摘要。它不包含可直接推理所需的大型数据和模型资产。

## 方案

最终采用 E2：**Blank-aware 查询调度开启，GPU 直接分配聚合关闭**。Blank-aware 模式在 FAISS 查询前根据 CTC argmax 选择非 blank 帧，避免对最终被跳过的 blank 帧执行检索。它没有修改模型、FAISS 索引、距离公式或概率融合公式。

实现位于 `src/selective_retrieval.py`。可通过 `selective_mode(engine, enabled=True/False)` 显式启用或回退。该原型限制为 batch size 1，并要求 `decode_skip_blank=true`、greedy 配置且未启用距离重算/缩放路径。

`src/gpu_aggregation.py` 提供 GPU 上直接创建聚合输出张量的可选实验实现，使用 `aggregation_mode(model, enabled=True/False)` 开关。**最终 E2 配置关闭此候选**：局部微基准收益不等于完整系统端到端收益。

## 已有实验摘要

| 数据划分 | E1 基线 | E2 Blank-aware | 观察结果 |
|---|---:|---:|---|
| AISHELL-1 dev | CER 4.340097691%；请求均值 15.765006 ms | CER 4.340097691%；请求均值 14.939531 ms | 质量一致；请求均值约降低 5.24% |
| AISHELL-1 test | CER 4.740132678%；请求均值 16.436934 ms | CER 4.740132678%；请求均值 15.197961 ms | 7,176 条预测逐条一致；请求均值降低 7.5377%，全任务耗时降低 7.5262% |

测试结果为历史冻结实验的汇总，不是本仓库创建时重新执行的实验。test 仅用于冻结方案的最终评估，没有用于参数选择。跨三个独立进程的请求延迟配对差 95% CI 为 [-1.865437, -0.612509] ms；统计细节和限制见 `docs/EXPERIMENTS.md`。

## 冻结条件

`configs/frozen_aishell.json` 记录最终 E1/E2 参数和开关。外部模型、字典、音频、Datastore 与 FAISS 索引必须由使用者按 AISHELL-1/官方来源单独获取，并自行校验。缺少资产时不得用其他数据集或模型替代。

## 运行要求

- Python 3.10；PyTorch/torchaudio、FAISS GPU 版本需与官方 kNN-CTC/WeNet 环境兼容。
- 这些模块需要在官方 kNN-CTC 推理对象上集成；代码片段本身不是独立 ASR 应用。
- 本仓库不包含一键真实 GPU 复现所需的音频、checkpoint、Datastore 或 FAISS 索引，也不声称在干净机器完成过完整 GPU 重跑。

代码只读地展示优化入口；请先阅读 `docs/SAFETY_AND_SCOPE.md`。不建议对未经核验的环境直接运行全量 AISHELL-1。

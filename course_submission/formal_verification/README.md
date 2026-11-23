# AISHELL-1 整数索引抽象验证

本独立实验重新安装并运行 `z3-solver==4.13.3.0`，使用当前 AISHELL Datastore 的 N1798000、词表 V4233。脚本 `verify.py` 不读取任何语音或模型，不调用 CUDA。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python verify.py
```

每个新 Run 保存 12 项实际 SAT/UNSAT 查询及 SMT-LIB。`solver.log` 和 `results/` 是本项目新运行证据；不能用其他项目的 solver 结果替代。

源码映射：`retrieval_guards.py` 在 get_knns 返回前拒绝负数/越界 ID，聚合路径在 `engine.py` 安装该 guard；`gpu_aggregation.py` 先 `vals[knns]`，再以标签索引 scatter。源码没有显式标签范围 guard，故必须区分 ID guard 的性质与额外标签范围契约。

ID 范围 guard 下没有 gather 行号反例（UNSAT）；加入标签范围契约后没有 scatter 下标反例（UNSAT）。不加入标签契约、删 ID 下界或放松上界时保存 SAT 反例。`id=-1/N`、`label=-1/V` 明确建模；N=0 和 K=0 不满足非空/正大小 guard 前提。合法 blank 标签允许存在。

标签契约的有限实证来自 `audit/assets.json` 对当前所有 values 的只读检查；这不是源码里存在标签 guard 的证明。证明仅针对所列整数范围抽象，不能覆盖浮点 softmax/scatter、FAISS 实现、CUDA、内存损坏、并发或 ASR 正确性。实际结果以本目录新 Run 的 result.json 为准。

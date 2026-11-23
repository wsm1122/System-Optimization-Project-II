# AI 复现操作说明

本说明针对 AISHELL-1 项目最终冻结方案 E2：Blank-aware 选择性检索开启、GPU 直接分配聚合关闭，`k=1024`、`nprobe=32`、`lambda=0.5`、`temperature=1.0`、`decode_skip_blank=true`。不要从旧 R4 配置推断最终方案。

## 安全默认操作

在项目根目录执行：

```bash
bash scripts/reproduce.sh --mode static
bash scripts/reproduce.sh --mode report-only
bash scripts/reproduce.sh --mode dev-gpu --dry-run
```

前两项是本地工程检查和历史结果只读核验；第三项仅打印 GPU 阶段计划。缺少模型、AISHELL-1 数据、Datastore、FAISS 索引、兼容 CUDA/FAISS 环境或获准 GPU 时，应返回 `BLOCKED`，不得自动下载替代资产或改用 CPU 冒充 GPU 实验。

## 真实 GPU 重跑

完整 dev 与冻结 test 会运行较长时间，test 最多 7,176 条音频。只有操作者在确认资源、数据和冻结 manifest 后，明确设置脚本要求的双授权环境变量，才可执行：

```bash
AISHELL_ALLOW_GPU=1 AISHELL_ALLOW_FINAL_TEST=1 AISHELL_GPU=4 \\
  bash scripts/reproduce.sh --mode all
```

上例中的物理 GPU 编号 `4` 仅是格式示意；操作者应先确认该物理卡获准且空闲，再替换编号。AI 助手不得代替操作者设置授权变量。一次完整运行可能包含 AISHELL-1 test；不要为了试跑或调试反复执行 test。若要单独运行阶段，先查阅 `README.md`、`REPRODUCE_PROMPT.md` 和 `r10/reproduce.py` 的门禁定义。

## 从零构建

从零下载/预处理数据、获取模型、构建 Datastore/索引或训练模型不是上述只读报告重建，也不是完整重跑。必须使用独立目录和新的 Run ID，先核实公开来源、资源预算、训练/开发/测试隔离与资产哈希；本交付包不含原始音频、权重、Datastore 或索引，也不宣称从零构建已验证。

## AI 执行准则

AI 可以读取本地源码和日志、解释门禁、运行静态/CPU 测试、核对已有 CSV/JSON、生成报告草稿；不得虚构缺失结果、改变冻结方案、覆盖历史记录、访问未配置的私有数据或将模拟/报告重建标为真实 GPU 复现。每个执行动作应保存命令、Run ID、日志和退出状态，最终由操作者审阅并决定是否提交。

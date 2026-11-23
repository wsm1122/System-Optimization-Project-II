# R8-B：AISHELL-1 双层优化组合验收报告

日期：2026-11-22  
结论：**BLOCKED（GPU 条件不可用，四组正式实验未启动）**  
运行编号：`combination_blocked_20261122T114555Z_c048b0ad`

## 1. 执行范围与结论

本阶段目标是在冻结配置下对 E1–E4 进行 AISHELL-1 dev 全量质量和性能对照。当前执行环境没有可访问的 NVIDIA GPU：`nvidia-smi` 返回 9 并报告无法连接驱动，`/dev/nvidia*` 下没有设备节点，PyTorch `cuda.is_available()` 为 false、设备数为 0，FAISS GPU API 虽存在但报告 0 张 GPU。因此在 GPU 安全性预检处停止，没有启动模型推理、GPU 搜索或正式性能 worker。

CPU 控制流与聚合等价测试 **7/7 PASS**；冻结模型、Datastore、索引和 AISHELL-1 dev manifest 的 SHA-256 **PASS**。这些检查不替代真实 GPU 正确性或性能实验。四组实验、14,326 条 dev 推理、CER、完整预测、模块计时、资源统计及配对统计均为 **NOT RUN / BLOCKED**。本轮未打开 AISHELL-1 test 数据，也未将 R4–R7 历史结果冒充为本轮结果。

## 2. 冻结设置

|项目|冻结值|本轮状态|
|---|---|---|
|模型、权重、字典|R4 AISHELL-1 Conformer|SHA-256 匹配|
|Datastore|1,798,000 × 256，FP16 keys|SHA-256 匹配|
|索引|IndexIVFPQ；nlist=4096，M=64，nbits=8|SHA-256 匹配|
|检索|k=1024，nprobe=32|配置文件值已核对|
|融合|lambda=0.5，temperature=1|配置文件值已核对|
|解码|decode_skip_blank=true|配置文件值已核对|
|数据|AISHELL-1 dev，14,326 条|dev 列表及参考 SHA-256 匹配|
|测试集|不使用|未打开、未运行|

配置来自 `configs/aishell.json`；本次实际资产和 dev 文件校验明细见 `r8/results/combination_blocked_20261122T114555Z_c048b0ad/preflight.json`。

本次四组设计对应已有实现：E2/E4 使用 `r7/selective_retrieval.py:14-24`，在原融合规则前筛掉 blank 查询；E3/E4 使用 `r7/snapshot/gpu_aggregation.py:7-16`，将聚合输出张量直接创建在目标设备。官方原聚合在 `${OFFICIAL_KNN_CTC_REPOSITORY}/wenet_knn_ctc_model.py:220-228`，其中输出先由 `torch.full` 创建，再 `.to(self.device)`。本轮没有修改这些推理实现；由于 GPU 不可用，组合的运行时行为未验证。

## 3. 设备与资源预检

|检查|结果|
|---|---|
|`nvidia-smi`|退出码 9：无法与 NVIDIA driver 通信|
|`/dev/nvidia*`|没有可用设备节点|
|`CUDA_VISIBLE_DEVICES`|未设置|
|PyTorch|1.13.0+cu117；CUDA runtime 11.7；CUDA 不可用；设备数 0|
|FAISS|1.7.2；GPU API 存在；GPU 数 0|
|并行项目 worker 检查|未发现正在运行的项目 Python/worker 进程|
|GPU 推理启动|否|

证据日志：`r8/results/combination_blocked_20261122T114555Z_c048b0ad/nvidia-smi.txt`、`device_nodes.txt`、`preflight.json`。不能在该环境确认物理 GPU 3/4 的占用或授权映射，故不选择设备、不尝试 CPU 替代正式 GPU 路径。

## 4. 四组实验状态

|组别|配置|GPU 正确性捕获|完整 dev 质量|性能与资源|结论|
|---|---|---|---|---|---|
|E1|原始检索 + 原始聚合|BLOCKED|14,326 条未运行；CER/预测无新结果|未测|BLOCKED|
|E2|Blank-aware 检索 + 原始聚合|BLOCKED|14,326 条未运行；CER/预测无新结果|未测|BLOCKED|
|E3|原始检索 + GPU 直接分配聚合|BLOCKED|14,326 条未运行；CER/预测无新结果|未测|BLOCKED|
|E4|Blank-aware 检索 + GPU 直接分配聚合|BLOCKED|14,326 条未运行；CER/预测无新结果|未测|BLOCKED|

请求延迟均值/P95、完整任务耗时、Search 查询帧数与次数、同步 Search/聚合耗时、吞吐量、RTF、CPU/GPU 资源和置信区间均留空为未测；不以历史结果填充。

## 5. CPU 侧验证

运行命令：

```bash
cd ${PROJECT_ROOT}
PYTHONDONTWRITEBYTECODE=1 ${ASR_ENV}/bin/python -B \
  -m unittest discover -s r8 -p 'test_*.py' -v
```

结果：7 项通过。

- 6 项来自 R8-A 控制流测试：混合 blank 查询裁剪、全 blank 跳过检索、全非 blank、选中邻居非法 ID 在 gather 前拒绝、异常输入下两路径行为差异、冻结配置保护。
- 1 项本轮新增 CPU 合成测试：抽取官方原聚合函数，与既有 GPU 直接分配实现以 CPU 张量执行；覆盖重复标签，输出张量及标签完全相同，最大绝对误差为 0。

该聚合测试只能证明 CPU 合成输入上的函数结果一致，不能证明 CUDA stream、GPU 分配器、GPU 数值、FAISS 邻居或真实音频端到端正确。

日志：`r8/results/combination_blocked_20261122T114555Z_c048b0ad/cpu_tests.log`。新增测试源码：`r8/test_aggregation_cpu.py`。

## 6. 历史证据的使用边界

R7 曾对 E1 与 Blank-aware + 原始聚合执行六个独立 GPU worker；R4 曾单独测试 GPU 直接分配聚合。它们属于不同阶段、不同运行批次，不能替代本轮要求的同一冻结设置下 E1–E4 平衡交替组合实验。R8-A 报告中复核的 R7 CER、预测一致性和约 5.2% 延迟变化只作为既有历史结论，不列入本轮结果。

故本轮无法回答 Blank-aware 是否在当前运行批次维持既有收益、GPU 直接分配在新负载下是否改善聚合、E4 是否有额外端到端收益、两项优化是否存在交互，或是否出现真实 GPU 中间张量差异。

## 7. 代码与资产校验

上游官方 commit：`b0f3e0f5d123b39e3c4134e76a9dbf104908b777`。

|代码/资产|SHA-256|
|---|---|
|官方 `wenet_knn_ctc_model.py`|`a9629983559e88a83edc7f4b7828fcd16c6bc42fb7987164850121e12bdb26e5`|
|R7 `selective_retrieval.py`|`b44b96ae18c4068316ff7f6278f57072f0e62742de3af817c94a10bafeacaa60`|
|R7 `gpu_aggregation.py`|`296e1b35ba09e243aac2b9e7ba68fe3bcdffb14ba1deb6cc29704d6e0aac050b`|
|R7 `worker.py`|`b2adba2f45b30a66636e11e2d44ead10d8eba76cc167b3570319f5e4b1c2f5d0`|
|本轮 R8-A CPU 控制流测试|`85baa9ad2ab96fd5b6008910bc8c352f48eac00df5ecf95123341b72dc451db5`|
|本轮 R8-B CPU 聚合测试|`30c4a76d9f51705d87b30ff32ad85c305269fa3afdf9ef8ab7923b3b1e99eb23`|
|checkpoint|`aa4cec2e2fc47b2061bcc47047df9d5a8782155145df3f61ff3d550e4568a606`|
|Datastore keys|`93e0974a8f3edac0076c2ae712088c2e17607e2078357175de285dbb8b5df980`|
|Datastore values|`6f9fe8c4ff01700e293fabf15222e5a93d50d644771be62efb8601e84d350815`|
|FAISS index|`c810f5ed569458a89fd090253cfd681cac9f583c67e95ea74eab8fca85bbdbeb`|
|dev 列表|`eb0b1fb68e66597f69ba9ea1b694dd6160b75d26b35202129aa1692e53d54569`|
|dev 参考|`c0ab7548c44fb553cf246196e6d4a068130106fee8ff99c3efcae41193ed533e`|

完整资产、配置、dev 文件和源码哈希校验值在本次 `preflight.json` 中。没有修改官方仓库、模型、Datastore、索引或 R4–R7 历史结果。

## 8. 验收判定与恢复条件

|验收项|状态|
|---|---|
|冻结配置与模型/索引/dev 哈希|PASS|
|CPU 单元测试|PASS，7/7|
|短/中/长真实音频中间张量捕获|BLOCKED|
|四组完整 dev 推理与逐条预测/CER|BLOCKED|
|平衡独立进程性能测试及配对统计|BLOCKED|
|R8-B 整体验收|**BLOCKED**|

解除阻塞需要在获准的 GPU 主机/作业环境中暴露 NVIDIA 设备节点，并确认指定物理 GPU 空闲、driver/CUDA/FAISS GPU 探针通过。恢复后应使用新的 Run ID，从四组的短/中/长真实 dev 正确性捕获开始；只有正确性通过后才运行完整 dev 和至少三组平衡独立进程性能复测。不得通过切换 CPU、改参数或采用旧结果将本阶段标成 PASS。

本轮停止于 GPU 预检；不运行 AISHELL test，不进入后续阶段。

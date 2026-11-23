# R6阶段一：真实性能瓶颈与Batch现状审计

日期：2026-11-22（北京时间）。真实阶段一执行PASS；后续优化未据此提前宣称成功。

新Run ID：`phase1_20261122T085413Z_cdf02bde`；执行日志与退出码见 `r6/results/launch_phase1_20261122T085413Z_46ae70e3/`。隔离快照位于r6/snapshot；原始R4/R5未修改，416个保护文件及全部固定资产摘要保持。

## 1. 工作负载和执行口径

同一RTX A6000物理GPU4（进程逻辑0），PyTorch1.13/cu117、FAISS1.7.2；固定原权重、1798000×256库和IVF-PQ索引，k1024/nprobe32/lambda0.5/temp1/skip_blank=true。原始GPU聚合实现、R5-B/C关闭；没有运行test。

整个dev14326条，先完整无模块插桩U一轮，再完整同步模块P一轮，均读取原始音频、单一loader；两轮预测逐条相同，S8494/D209/I209/N205341，CER4.340097691%。模型和索引初始化单列，评分不混入推理任务时间。两轮不是多次独立任务重复，不据此报告稳定总体P95或性能CI。

## 2. 完整dev新基线

|指标|无模块插桩U|同步模块P|
|---|---:|---:|
|完整dev推理任务秒|224.744828|228.141747|
|请求均值ms|15.659346|15.886276|
|dev单轮请求P95 ms|17.248989|17.416003|
|音频条数/秒|63.743402|62.794294|
|音频时长秒/墙钟秒|289.755472|285.441154|
|PyTorch allocated峰值bytes|355772928|355772928|
|PyTorch reserved峰值bytes|406847488|406847488|

共享模型/索引初始化 3.834427秒。CPU RSS与进程显存采样见summary与launcher，不混为PyTorch峰值。任务时间包含loader创建、输入、推理、预测/计时记录写入；请求不含模型加载与评分。

## 3. 非重叠同步模块分析

|模块|同步均值ms|同一P请求内比例%|
|---|---:|---:|
|preprocessing_ms|1.715103|10.796|
|input_h2d_ms|0.055660|0.350|
|encoder_ms|11.334904|71.350|
|query_access_ms|0.005427|0.034|
|ctc_ms|0.059655|0.376|
|search_guard_ms|1.229255|7.738|
|aggregation_ms|0.940687|5.921|
|fusion_ms|0.265734|1.673|
|cpu_collapse_ms|0.017031|0.107|
|unassigned_wall_ms|0.262820|1.654|

这里的比例只相对同一同步P请求，不除以U。没有同时记录嵌套retrieval_fusion复合时间；unassigned包含Python、解码其余操作以及插桩同步边界成本，不是纯业务残差。Search包含原返回设备处理与合法性守卫，不能称为纯FAISS kernel；CTC仅log_softmax，其他准备并入残差。前处理是同一loader.next复合区间，H2D在P中单独同步；P会增加或改变同步，不能代替真实U。

## 4. 条件理论上限

|方向|串行P份额|该部分加速2倍时系统因子|该部分完全消除时系统因子|
|---|---:|---:|---:|
|encoder|0.713503|1.554609|3.490437|
|search_and_aggregation|0.136592|1.073302|1.158201|
|preprocessing|0.107961|1.057061|1.121028|

采用S=1/[(1−f)+f/s]。假设原串行结构、其他成本固定、没有新开销和重叠；只是P测量代理，不是U严格硬上界，也不限定批处理吞吐。检索加聚合全部消除的代理上限约1.158倍，不能预设选择性检索单项达到20%以上。Encoder约71.35%的P份额，明显比只优化守卫或临时小操作更值得优先探索。

## 5. Batch现状与真实限制

`engine.py:68`强制Dataset静态batch1，运行时原始batch_config也为static/1；完整dev实际逐条断言batch_size=1。DataLoader(batch_size=None)表示Dataset已组batch，不是跨音频batch。

现有FAISS已经批量检索一个话语的所有帧，所以“逐帧改批查询”不是新增优化。官方CLI参数默认16不代表课程正式评测启用16，也不代表kNN数学路径能处理B>1。

原代码get_knns只squeeze第0维，B>1保留三维输入；融合硬编码interpolated[0,mask]；Engine只返回hypotheses[0]。原Dataset.padding按长度重排批内样本，kNN解码缺少有效帧裁剪。因此不能只改batch_size后统计吞吐，否则可能形状失败、丢输出或把padding当查询。

两项CPU原路径回归实测通过：B1搜索成功、B2搜索按预期拒绝；B2原融合按预期异常。这证明需要适配，不等于GPU批处理实验通过。证据r6/results/batch_audit_cpu.log。

## 6. 后续优先方案

首选批量Encoder、逐音频复用原检索/融合。先adapter batch1对齐原始路径，再2/4/8/16；按mask裁剪encoded和FFN key，按ID恢复顺序，保留所有邻居守卫。小规模严格对比失败先分析，不自动放宽精度、质量或继续完整dev扫描。

第二候选是FP32 Encoder执行后端。当前Attention为相对位置实现；环境没有ORT/ONNX/TensorRT，现有ONNX脚本导出forward_chunk且未直接输出kNN所需最后FFN key。不能只比较encoder_out就声称ASR相同。优先调查兼容环境内TorchScript显式双输出，若需Torch2编译或新后端必须隔离环境，不能升级生产环境。该候选尚未实测。

选择性检索先研究融合恒等保留的blank帧：现有skip_blank只在融合处筛选，没有减少Q。可独立测试blank-only数学等价裁剪，再考虑pmax/熵近似规则。调参规则需dev说话人隔离子集选择和验证，必须记录真实search输入Q、CER错误数及请求/任务收益。

## 7. 文件和复现

原始逐请求记录、预测、评分、资源与模块CSV：`r6/results/phase1_20261122T085413Z_cdf02bde/`。

`R6_ALLOW_GPU=1 R6_GPU=4 python r6/launch.py` 为本阶段新完整dev入口；必须先获许可和确认GPU空闲，任何失败保留回执且停止。报告重建使用 `python r6/report_phase1.py --run-id phase1_20261122T085413Z_cdf02bde`，不会启动GPU。

限制：只有各一轮全dev任务；未测在线队列/成批等待、并发容量、硬件利用率或带宽上限。P95是这轮实际dev请求的描述统计，不是稳定线上分布。后续所有优化必须重测同时间窗口基线。

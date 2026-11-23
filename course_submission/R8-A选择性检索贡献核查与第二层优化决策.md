# R8-A：选择性检索贡献核查与第二层优化决策

日期：2026-11-22。**本轮只读源码审计、历史数据重算与CPU测试PASS；没有启动GPU、读取AISHELL test或运行新ASR推理，没有实施候选优化。** 本轮按最新授权替代此前的Encoder Profiler任务；新的GPU算子级分析未执行。

项目根目录：`${PROJECT_ROOT}`。

## 一、官方与R7执行路径对照

### 1. 固定版本与实际源码

只读核实官方仓库 `${OFFICIAL_KNN_CTC_REPOSITORY}` 的HEAD为：

```text
b0f3e0f5d123b39e3c4134e76a9dbf104908b777
```

官方 `${OFFICIAL_KNN_CTC_REPOSITORY}/wenet_knn_ctc_model.py` 与R7实际运行快照 `${PROJECT_ROOT}/r7/snapshot/vendor/wenet_knn_ctc_model.py` 的SHA-256相同：

```text
a9629983559e88a83edc7f4b7828fcd16c6bc42fb7987164850121e12bdb26e5
```

R7选择性包装 `${PROJECT_ROOT}/r7/selective_retrieval.py` 的SHA-256：

```text
b44b96ae18c4068316ff7f6278f57072f0e62742de3af817c94a10bafeacaa60
```

官方仓库检查前已有 `.gitignore` 跟踪修改，本轮前后状态一致；不声称仓库最初完全干净，也没有修改它。

### 2. 最小调用链

**官方原始推理：**

```text
knn_ctc_greedy_search
  ├─ Encoder → 最后FFN输入作为key
  ├─ CTC log_softmax → 每帧CTC概率
  └─ wrapper.process(全部话语帧key, CTC概率)
       ├─ get_knns(全部话语帧key)
       │    └─ index.search([T,256], k=1024)
       ├─ knns_to_log_prob(全部T帧邻居)
       │    └─ softmax → 标签gather → scatter_add → log → nan_to_num
       └─ interpolate(..., decode_skip_blank=true)
            ├─ argmax(CTC) != 0 得到非blank mask
            ├─ clone原CTC
            └─ 只更新非blank行，blank行保持原CTC
```

**R7新执行路径：**

```text
同一个Encoder与CTC
  └─ 由context开关替换wrapper.process
       ├─ selected = nonzero(argmax(CTC) != 0)
       ├─ query = 原key.index_select(帧维, selected)
       ├─ 原get_knns(query) → 原FAISS搜索，只处理S个非blank帧
       │    └─ 课程既有合法性守卫：拒绝非法邻居后才能gather
       ├─ 原knns_to_log_prob：只聚合S帧
       ├─ 将选中概率放回全帧中间张量
       └─ 复用原interpolate，保持原融合公式和blank输出
```

### 3. 逐函数证据

|步骤|实际代码位置|核查结论|
|---|---|---|
|原推理取得key和CTC|`${PROJECT_ROOT}/r7/snapshot/vendor/wenet/transformer/asr_model.py:381`、`:390`、`:396`|key、CTC仍按完整话语生成；R7没有减少Encoder帧计算|
|原推理进入kNN|`${PROJECT_ROOT}/r7/snapshot/vendor/wenet/transformer/asr_model.py:407`|调用wrapper.process，后续从融合概率产生文本|
|原process|`${OFFICIAL_KNN_CTC_REPOSITORY}/wenet_knn_ctc_model.py:151`，关键调用`:158`、`:162`、`:169`、`:179`|先全帧检索、聚合，再融合；process内没有提前查询mask|
|原FAISS调用|`${OFFICIAL_KNN_CTC_REPOSITORY}/wenet_knn_ctc_model.py:208`、`:216`|只squeeze话语维，整批[T,D]进入search；不是每个blank帧单独调用一次FAISS|
|原邻居聚合|`${OFFICIAL_KNN_CTC_REPOSITORY}/wenet_knn_ctc_model.py:220`|对get_knns返回的全部行聚合，未按CTC blank提前裁剪|
|原blank跳过|`${OFFICIAL_KNN_CTC_REPOSITORY}/wenet_knn_ctc_model.py:245`，具体`:247`至`:252`|在interpolate内判定argmax!=0，只跳过blank的融合更新；不是跳过其搜索|
|R7提前过滤|`${PROJECT_ROOT}/r7/selective_retrieval.py:14`、`:18`、`:19`、`:20`|在get_knns之前筛选key，再复用原search和聚合|
|R7复用原融合|`${PROJECT_ROOT}/r7/selective_retrieval.py:21`至`:24`|重新构造全帧中间张量，调用原interpolate；blank数学规则不是新提出的|
|R7开关|`${PROJECT_ROOT}/r7/selective_retrieval.py:29`至`:34`|默认关闭，context内替换process，可回退|
|查询计数器|`${PROJECT_ROOT}/r7/worker.py:23`至`:26`|在实际index.search调用前累计queries.shape[0]，不是统计事后被mask的输出|
|排除热身计数|`${PROJECT_ROOT}/r7/worker.py:52`至`:57`|6条热身后清零，完整dev正式运行后保存Q与调用数|

冻结字典实际存在 `<blank> 0`，与两条路径使用的blank ID一致。

### 4. 查询减少的来源及贡献边界

每个E1 worker累计 **1,604,729** 个search输入向量，每个R7 worker为 **211,871**。减少 **1,392,858** 个，即 **86.7970853646%**。减少的是被原CTC判定为blank的帧所对应的搜索行和聚合行；没有修改k、nprobe、索引、标签、模型或融合权重。

两配置每轮都调用FAISS **14,326次**，与dev音频条数相同。因此不能写成“FAISS调用次数减少86.8%”。这是缩小话语内search batch的行数，不是首次引入话语内帧批量检索。合成全blank用例下R7确实可以零次search，但现有完整dev每条均有非blank查询，未观察到调用次数下降。

本轮可直接复核聚合计数器记录、清零位置、真实调用入口，并用CPU替身验证控制流；**历史raw.jsonl没有逐请求Q字段，不能从该文件独立重新推导每条blank帧数量**。聚合Q事实来自六个真实GPU worker保存的计数器结果，而不是本轮重新运行GPU得出。

准确的贡献名称建议为：**“复用原blank融合判定，将其前移为查询与聚合工作的执行裁剪”**，或“blank-aware检索执行调度优化”。新增的是执行位置及实际工作量变化，不是blank判定规则、CTC blank pruning理论、FAISS算法或概率公式的创新。这里仅核查相对指定官方commit的差异，没有进行全球文献首创性证明。

还需区分官方已有的建库blank过滤：`${OFFICIAL_KNN_CTC_REPOSITORY}/wenet_knn_ctc_model.py:337`调用null_mask，`:344`至`:349`过滤建库标签0，之后reservoir sampling。这是在**Datastore构建阶段**剔除blank标签，不等于推理时不搜索CTC blank查询。该功能及原GPU FAISS、FP16 key、话语内batch均不能包装为本项目新增工作。

### 5. 等价性范围：不能无条件推广

- 原数学公式在blank行不使用检索概率；固定B1 greedy、skip_blank=true、scale_lmbda=false、recompute_dists=false时，跳过这些行的计算有明确依据。
- 已有完整dev逐条预测相同，以及R6小规模浮点、ID和标签校验，是实际证据，不是对所有输入的形式化证明。
- R7辅助距离/标签返回从全T行改为S行，不能声称其他使用这些返回值的调用者接口无条件不变。
- **异常行为不完全等价**：原路径若在blank行得到非法邻居，仍会被课程守卫拒绝；R7未搜索该行，不产生该非法结果，也不执行其守卫。所有实际搜索的非blank邻居仍必须在gather前校验。不能表述为“原异常路径也完全一致”。

本轮CPU实测覆盖上述范围，包括特意构造“全blank且FAISS替身返回非法ID”的差异，不隐瞒该限制。

## 二、R7性能与质量复核

### 1. 本轮实际执行的方法

输入为 `${PROJECT_ROOT}/r7/results/processes_20261122T101641Z_19f5dbfb/` 的六个worker。

本轮从每份dev/raw.jsonl重新计算请求均值及线性插值P95，核对音频ID、顺序、14,326条完整性和summary；读取worker中的实际Q、calls与PID。全部预测文件重新做逐字节比较。本轮还用冻结的compute-wer.py对已有预测及原dev参考在CPU重新评分6次，并核对历史逐条评分的计数和，未运行声学模型。

独立输出：`${PROJECT_ROOT}/r8/results/audit_20261122T112406Z_844212e0/`。

### 2. 六个worker复核表

|顺序|配置|PID|请求均值ms|P95 ms|完整dev任务s|Q|calls|
|---|---|---:|---:|---:|---:|---:|---:|
|0|E1|1603825|15.789269|17.328705|226.597623|1604729|14326|
|1|R7|1604432|14.816657|15.799625|212.654327|211871|14326|
|2|R7|1605149|14.860773|15.811224|213.284983|211871|14326|
|3|E1|1605668|15.627275|17.171951|224.274628|1604729|14326|
|4|E1|1606303|15.630205|17.174673|224.315138|1604729|14326|
|5|R7|1606825|14.907602|15.897076|213.957858|211871|14326|

**任务耗时的复核边界：** task_seconds是当时保存的任务墙钟计时，不是请求耗时简单求和。该字段在worker与dev/summary.json中一致；各轮任务比请求耗时之和多约0.3895–0.4006秒，与loader初始化、循环和写盘等外部区间相容，但没有原始起止时间戳，不能本轮独立重新测量过去的墙钟时长，也不能把差额精确归因于某个操作。

### 3. 均值、P95和质量

|指标|E1，3个独立进程|R7，3个独立进程|
|---|---:|---:|
|请求均值ms，进程均值±SD|15.682250 ± 0.092693|14.861677 ± 0.045479|
|每进程P95的均值±SD，ms|17.225109 ± 0.089727|15.835975 ± 0.053232|
|完整dev任务s，均值±SD|225.062463 ± 1.329642|213.299056 ± 0.651880|
|每轮S/D/I/N|8494/209/209/205341|8494/209/209/205341|
|每轮错误数|8912|8912|
|CER %|4.340097691|4.340097691|

六份预测逐字节一致；CPU重新评分与每轮历史计数均一致。请求均值降低 **5.2324905245%**；完整dev任务均值降低 **5.2267299304%**，两个百分比不能混写成相同的精确值。P95指完整dev有限请求分布，每进程单独计算后再平均，不是在线服务等待延迟，也不是先把三个进程当作一个大独立样本池。

### 4. 三对差值与区间

差值统一为“R7−E1”，负值表示更快；按预先固定连续两进程组成AB、BA、AB。

|配对|请求均值差ms|完整任务差s|
|---|---:|---:|
|0|−0.972611445|−13.943296635|
|1|−0.766502173|−10.989644838|
|2|−0.722603057|−10.357279873|

均值请求差约−0.820572225ms；使用df=2的t临界值4.3026527299，三对进程差值的95%CI为 **[−1.1521718244, −0.4889726257]ms**。任务差区间为 **[−16.5183803041, −7.0084339266]s**。本轮独立重算与原analysis.json一致。

另按每条音频先平均各配置3个进程，seed=7做2000次音频bootstrap，重算区间 **[−0.8264470051, −0.8153773044]ms**，也与原记录一致。该区间不处理说话人相关性，不能替代跨进程区间，更不能把42,978次同配置请求当作42,978条独立音频。

因此可以确认**本次设备、dev负载和测量协议下有跨进程可重复的约5.2%改善**。但只有3对进程、AB两对BA一对且顺序不能完全对半，t区间依赖独立近似正态配对差假设；不可外推为任意时间、设备或并发负载的收益保证。

## 三、第二层优化候选审查与决策

### 1. 已有热点证据及排除项

R6完整dev同步分项原始数据 `${PROJECT_ROOT}/r6/results/phase1_20261122T085413Z_cdf02bde/profile_serialized/summary.json`：Encoder11.334904ms、Search及守卫1.229255ms、聚合0.940687ms、融合0.265734ms。这些属于**同一同步插桩运行**，不能除以R7独立未插桩请求时间计算严格占比或相加推导真实端到端耗时。

本轮重新汇总16条真实dev查询、每条5次的历史独立微基准 `${PROJECT_ROOT}/r6/results/selective_micro_20261122T092557Z_76498490/result.json`：

|配置|纯Search ms|Search及守卫ms|聚合ms|
|---|---:|---:|---:|
|E1|1.083878|1.156244|0.986400|
|blank-only|0.487929|0.537723|0.634557|

每配置80次测量只有16个音频独立单位。纯Search为同步FAISS API墙钟，不是单kernel；Search及守卫包含Search，二者不能相加。聚合仍约0.6346ms是复合耗时，不能直接说都是CPU拷贝、scatter或内存带宽。

其他源码审查结果：

- 原wrapper已导入faiss.contrib.torch_utils（官方文件`:12`）；get_knns的`.to(self.device)`（`:217`）在同设备、同dtype情况下可能是无拷贝返回。当前没有证据支持“FAISS结果每次经CPU NumPy往返”，因此不把删除这些转换列为候选。
- labels在初始化时即`.long().to(self.device)`（官方文件`:138`至`:141`），不是每请求重读CPU标签库。
- 冗余CTC假设和守卫主机同步已有R5单独实验，未确认可靠端到端收益；不能重新包装成尚未做过的新发现。
- R7无Encoder算子级profiler证据，Batch2、ORT均未满足严格数值/邻居合同。本轮不因Encoder大就直接指定其改写，也不把Batch/ORT写成PASS。

### 2. 候选一：GPU直接分配聚合输出，组合复测既有技术【首选】

**源码位置：** `${OFFICIAL_KNN_CTC_REPOSITORY}/wenet_knn_ctc_model.py:225`先在CPU创建零输出再`.to(self.device)`。既有实现 `${PROJECT_ROOT}/r7/snapshot/gpu_aggregation.py:7`至`:16`改为直接在目标GPU分配，保留softmax、标签读取、scatter_add、log和nan_to_num。开关位于该文件`:19`。R7 worker经phase1.py:91传入optimized=False，因此R7正式复测没有启用这项优化。

**层次：** GPU张量分配与CPU→GPU数据传输执行层；与blank-only缩减搜索/聚合查询行的工作量调度不同。**这是R4已有技术，不能宣称为R8新实现；本轮建议的是在R7新负载下独立组合复测。**

**已有真实证据：** 本轮从R4原始combine/raw.jsonl重新计算，并核对combine/summary.csv：E1聚合1.264964ms、E3聚合0.919309ms；对应无插桩请求24.210192ms、24.846008ms。每配置36次请求、6个音频。历史局部改善没有支持整体加速，不能只选局部数字宣告成功。数据源为 `${PROJECT_ROOT}/results/course_20261121T153154Z_147dfac4/combine/`，本轮未读取test结果来做选择。

**为何仍首选：** 当前R7仍走CPU零张量→GPU路径；已有独立开关、修改范围最小、同数据集存在聚合级测量证据，适合以低风险回答能否成为有效第二层。**不是因为已证明R7组合会加速。** Q缩小后该收益可能更小或落入噪声，应允许负结果。

**正确性风险：** dtype/default-device、零初始化、scatter重复标签归约、CUDA stream/分配器行为及空查询路径。已有R4验证不能替代R7组合验证；不能顺带改变概率精度、k或守卫。

**最小验证方案（仅计划，未执行）：** 先真实dev短/中/长样本捕获并比较R7与R7+GPU分配的输入、Encoder、CTC logits、选中query、邻居ID/距离/labels、检索概率、融合、scores与文本；保持float容差和离散严格合同，测试重复标签及非法邻居在gather前拒绝。通过后完整dev预测/CER验证，再用同时间独立进程平衡配对性能测量，单列初始化、任务、请求和资源。正式性能无CPU张量捕获；CI跨0则报告不显著。

### 3. 候选二：消除R7融合前全帧中间概率张量和重复mask计算【备选】

**源码位置：** `${PROJECT_ROOT}/r7/selective_retrieval.py:14`已计算selected，但`:21`又分配[1,T,V]的full_probabilities、`:22`写回S行、`:23`进入原interpolate。官方`:248`至`:252`再次argmax并用mask从这个全帧中间张量取回同样S行。

**建议行为：** 后续若授权，可仅在冻结greedy分支直接以现有selected行的[S,V] kNN概率和对应CTC概率计算原logaddexp，写入CTC clone；避免“先扩为T行再取回S行”的中间物化，并复用已计算mask。最终输出仍必须是完整[1,T,V]；不能把最终输出本身也裁掉。

**层次：** GPU中间张量物化、索引与算子执行优化，而不是改变检索策略、邻居数量或融合数学公式。区别于blank-only提前裁剪实际search输入。

**开销证据与不足：** 源码明确存在全帧分配、S行写入和重复argmax/索引；R6旧同步融合整体0.265734ms说明有可测复合阶段，但**没有R7该中间张量/重复argmax的独立时间**，不能把0.265734ms视为可全部消除的收益，也不能提前给出加速比。

**正确性风险：** 空selected、行顺序、blank原值、in-place别名、标量log常数及dtype、浮点执行次序、辅助距离/标签返回合同；尤其不能擅自把原np.log常数替换成精度不同的表达式。其他动态lambda/recompute或非greedy分支必须维持原拒绝/回退边界。

**最小验证方案（仅计划，未执行）：** 先在独立目录确认分配/argmax/索引的复合时间及临时内存，再实施单开关候选；全blank/全非blank/混合、重复标签、不同帧数和真实邻居检查，逐字段对照原R7；通过才进行完整dev和独立进程性能测试。若分配确实减少但请求收益不显著，必须保留这个结论。

### 4. 最终决策

1. **首选候选一，定位为“复用已实现GPU直接分配，在R7选择性负载下做组合复测”，不虚构新增创新。**
2. 候选二为有源码依据、但缺少单项时间证据的新执行优化备选；先测量后授权实施。
3. 本轮没有实施任何候选，也没有确认新的第二层端到端收益。若需要“全新优化”而非复用现有第二层，候选一不能满足新颖性要求，候选二也尚未验收，必须如实说明。

## 四、本轮验收、证据与复现

审计Run：`audit_20261122T112406Z_844212e0`。

|项目|状态|证据|
|---|---|---|
|固定官方commit及wrapper对照|PASS|receipt.json中的版本与SHA|
|六轮请求统计与覆盖复算|PASS|recomputed.json、workers.csv|
|CPU重新评分与历史逐条计数复核|PASS|rescore_0.log至rescore_5.log|
|六份完整预测逐字节一致|PASS|workers.csv中的相同prediction_sha256|
|三对进程及音频bootstrap区间复算|PASS|recomputed.json，与原analysis.json核对|
|CPU控制流/边界测试|PASS，6项|cpu_tests.log|
|历史文件不变性|PASS，850文件|protected_before.json及receipt.json|
|新GPU性能、Encoder profiler、候选实现|NOT RUN|本轮授权明确禁止|

全部新证据在 `${PROJECT_ROOT}/r8/results/audit_20261122T112406Z_844212e0/`；候选历史数据复核在该目录candidate_evidence.json。

可复制CPU复核命令：

```bash
cd ${PROJECT_ROOT}
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  ${ASR_ENV}/bin/python -B r8/audit_r7.py
PYTHONDONTWRITEBYTECODE=1 \
  ${ASR_ENV}/bin/python -B r8/candidate_evidence.py
```

每次audit生成新的结果目录，不覆盖R4–R7。CPU测试通过AST提取官方原始process/get_knns/聚合/融合函数并使用确定性搜索替身，避免直接import官方模块时其logging.FileHandler(mode='w')覆盖日志。这些测试证明控制流和合成整数/概率用例，**不冒充真实FAISS GPU测量**。

测试覆盖：混合blank裁剪真实调用输入行数、全blank零搜索/零聚合、全非blank不缩减、非法选中邻居在gather前被拒绝、不支持的配置被拒绝，以及blank异常行为并非无条件等价。没有实现两个候选，没有修改生产代码或冻结资产。

**完成本轮审计后停止，等待下一阶段授权。**

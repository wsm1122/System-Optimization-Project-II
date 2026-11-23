# AISHELL-1 资产来源与放置说明

运行仅访问 `configs/aishell.json` 指定的资产别名；缺文件或 SHA-256 不符即停止。服务器当前采用只读符号链接复用公开资产，链接不随提交包分发。

| 项目内路径 | 内容和来源 |
|---|---|
| `assets/audio/{train,dev,test}/` | OpenSLR SLR33 AISHELL-1 原始 16 kHz 单声道 PCM16 wav；保留原说话人目录 |
| `assets/model/final.pt` | 官方 WeNet U2++ Conformer AISHELL checkpoint（20210601 模型）；不修改权重 |
| `assets/model/train.reproduction.yaml` | 已有对应模型配置；运行时只重定位 CMVN 路径 |
| `assets/model/units.txt` | 4233 项字典，ID 与模型输出一致 |
| `assets/model/global_cmvn` | 对应原模型 CMVN |
| `assets/store/dstore_1798000_256_keys.npy` | raw FP16 `[1798000,256]`，不是带 NPY header 的数组 |
| `assets/store/dstore_1798000_256_vals.npy` | raw int32 `[1798000,1]`，标签来自 CTC pseudo labels |
| `assets/store/index_1798000_256.indexed` | L2 `IndexIVFPQ`，nlist4096、PQ64×8 |
| `assets/store/reproduction_manifest.json` | official sampler、seed1、train-only 构建来源和模型/字典/配置摘要 |

现有服务器公共资产相对上游项目的位置分别是 `data/aishell_source/data_aishell/wav`、`exp/20210601_u2++_conformer_exp` 和 `runs/full_r1_official_20261109T065922Z_543dc128/datastore`。不要求该上游目录在目标机器存在，只要求将同内容文件放到上述项目别名并通过摘要校验。模型/数据/索引均不随小型提交包分发。

资产大小、SHA-256、结构和标签范围的实际检查见 `audit/assets.json`；全部固定摘要也在 `configs/aishell.json`。该库是按官方配方、同一模型完成的既有公开数据复现产物，不能描述为作者直接发布的二进制检索库。源码对应 commit 记在配置中；`audit/source_migration.json` 保存选择性迁移文件摘要。

既有 official sampler 的边界批次存在原实现回放/截断细节：构建清单记录有效接收帧 1798064、内部计数 1798074，实际存储和索引容量均为 1798000。内部计数不是唯一帧数或库大小；本项目保留原资产，不修复/重采样，不将这部分既有行为包装为新增优化。

AISHELL 原始录音数量与有效转录匹配清单不同：有效 train/dev/test 分别为 120098/14326/7176；排除缺少官方转录的 train320、dev5，不删原 wav。新清单只改为项目相对资产路径并附加时长，不改转录；`audit/list_sources.json` 记录原始清单摘要。train/dev/test 说话人与音频 ID 互斥，构建 manifest 的输入摘要对应 train。

## 从零资产准备

官方配方位于 kNN-CTC README 与 `knn_run_aishell-1.sh`。先从 OpenSLR 准备 AISHELL-1，再按 WeNet 官方 pretrained models 文档获取对应 checkpoint，保持词表/CMVN/配置一致。pruned 库参数为 N1798000、D256、nlist4096、PQ64×8、official sampler/seed1，训练集仅用于库构建；dev/test 不进入库构建。

理论 keys 约 878 MiB、values 约 6.86 MiB、已有索引约 128 MiB；还需模型、音频、索引训练临时内存与构建日志。历史同服务器全流程（含建库及两次 test 推理）约 43 分钟，属于历史资源参照，不保证新硬件耗时。重新生成的索引若摘要不同，应另行审计与命名，不能跳过当前冻结校验假称相同资产。本轮没有重新训练或建库。

原始公开来源：[AISHELL-1](https://www.openslr.org/33/)、[kNN-CTC 论文](https://arxiv.org/abs/2312.13560)。

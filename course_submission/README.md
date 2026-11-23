# AISHELL-1 课程提交材料

本目录包含课程报告、AI 辅助说明、复现 Prompt、阶段报告、冻结配置、资产校验和 Z3 材料。核心优化代码位于仓库根目录 src/、tests/、configs/。

## 实验证据

[evidence/full/README.md](evidence/full/README.md) 提供已获准发布的逐条材料：12 份 R8-B dev 预测和逐条评分，以及 worker 汇总、分析、SVG 和脱敏资产审计。R9 当前仅公开第一份 test 预测。原始逐请求日志、逐音频配对表及其余测试预测和评分仍受上传审批限制。

可执行 `python3 course_submission/evidence/full/verify_published.py` 复核公开证据；脚本会列明尚未公开的范围。它不启动 GPU 推理，也不会把未公开文件视作校验通过。

大型模型、AISHELL 音频、Datastore、FAISS 索引、中间张量、虚拟环境均未包含。当前公开镜像尚不能代替本地完整证据包，也尚未验证干净服务器上的完整 GPU 实验复现。

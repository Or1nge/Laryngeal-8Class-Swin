# CHANGELOG

## v6.35 (2026-06-02)

### Patient ID parsing
- **裸 DICOM 文件名识别**：ROI-cropped glottis gate 分支的 binary split 生成逻辑现在把 `13.0000000082503...jpg` 解析为 `00082503`，避免无姓名前缀 LDP 图像被当成逐图独立病人。

---

## v6.34 (2026-05-18)

### ROI 裁切声门 Gate 实验分支
- **新增 ROI 裁切二分类 split 生成脚本**：`图像识别/glottis_binary/build_roi_cropped_split.py` 复用现有患者级 split，把正类限制在当前 8 分类任务的 7 个 VOC 类中，并按“半数 ROI 裁切正类 + 半数原图正类 + 同数量非声门负类”生成平衡训练/验证/测试清单。
- **ROI 裁切工具支持清单输入**：`图像识别/roi_reflection/crop_project_rois.py` 新增 `--include-list`，可只裁切实验选中的图像，避免为了 gate 鲁棒性实验复制整个原始数据树。
- **文档同步实验命令**：`图像识别/glottis_binary/README.md` 记录 ROI 裁切、SupCon+CE 训练和结果目录约定；所有裁切图、split 与 checkpoint 默认写入 `Results/roi_cropped_glottis_gate/`。
- **本分支实测结论**：`roi_cropped_supcon_swin/supcon_swin_base` 在平衡 test split 上推荐阈值 0.96，Accuracy 0.9784、Specificity 0.9948、Glottis recall 0.9619、AUROC 0.9989。插回 main 视频管线且沿用阈值 0.94 后，全量 `classified_videos` 从旧 gate 的 22/43 提升到 23/43，但 mean gate frames 从 43.6 降到 33.7，说明它没有整体放宽召回，不应直接视为默认替代模型。

---

## v6.33 (2026-05-18)

### BAGLS 视频流程合并到 main
- **main 视频入口改用 BAGLS 标准管线**：`视频识别/video_inference.py` 已从 BAGLS ROI/reflection 分支合并，诊断口径固定为质量 gate -> ROI validity/reflection gate -> `glottis_binary` 声门 gate -> 8 分类疾病模型 -> top-fraction 投票聚合。
- **补入 ROI/reflection sidecar 模块**：`图像识别/roi_reflection/` 进入 main worktree；已验证的 8 类 ROI-soft checkpoint、ROI localizer 和 reflection gate 已迁入 `Results/main/roi_reflection/`，视频推理默认优先读取 main 结果树。
- **移除旧多规则诊断作为主口径**：`voc_sum_gt_nonvoc`、`nonvoc_lt_*`、`all_frames` 等历史规则不再用于病人级诊断；相关 Non-VOC/VOC 概率仅保留为逐帧审计字段。
- **文档同步**：根 README 与 `视频识别/README.md` 已改为 main 当前标准流程，并说明 `--render-video` 诊断叠加输出和默认 checkpoint 回退关系。

---

## v6.28 (2026-05-06)

### 视频白斑反光误判保护
- **新增强反光质量 gate**：`视频识别/video_inference.py` 默认要求 `white_ratio <= 0.12`，剔除大面积饱和白色反光帧，降低反光被识别为 `声带白斑` 证据的风险。
- **低证据时长标记**：新增 `--min-evidence-duration-sec`，默认少于 1 秒阈值以上证据时在诊断表标为 `low_confidence`，避免单帧或极短片段被误读为稳定视频诊断。
- **低置信诊断调用降级**：诊断表新增 `diagnosis_call` / `diagnosis_call_zh`，低置信病例显示为 `review_required` / `待人工复核`，保留原始 `pred_label` 供追溯。

---

## v6.27 (2026-05-05)

### Grad-CAM 显示简化
- **移除白色轮廓线**：视频诊断 Grad-CAM overlay 不再额外绘制 CAM 高响应区域的白色 contour，只保留热力图叠加，减少对医生复核的视觉干扰。

---

## v6.26 (2026-05-05)

### 视频推理抽帧与输出留痕调整
- **提高默认抽帧密度**：`视频识别/video_inference.py` 默认 `--sample-fps` 从 4 调整为 8，用更密的时间采样支持声门 gate 和病灶秒段定位。
- **普通 keyframe 默认关闭**：`keyframes/` 只是每个规则的最高分 JPG 原图，留痕价值低于病人级 `diagnosis_gradcam/`，默认不再保存；调试时可显式传入 `--save-keyframes`。

---

## v6.25 (2026-05-05)

### 视频识别接入声门二分类 Gate
- **前置二分类 gate**：`视频识别/video_inference.py` 默认加载 `glottis_binary` Swin checkpoint，先输出 `prob_glottis` 并筛出有效声带/声门帧，再送入 8 分类疾病模型。
- **高特异性阈值优先**：默认主阈值为 `0.94`，且不自动降阈值，优先减少非声门帧进入疾病模型；如需提高召回，可显式传入 `--glottis-gate-fallback-threshold 0.62`。
- **短暂中断合并**：新增 `--max-segment-gap-sec`，默认合并 1 秒内的闪光、瞬时模糊或遮挡中断，使声门出现区间更符合视频连续性。
- **输出字段补充**：`frame_predictions.csv`、`video_predictions.csv`、`diagnosis_summary.csv` 和 `summary.json` 记录二分类 gate 概率、阈值、保留帧数、是否使用回退，以及低置信原因。

---

## v6.24 (2026-05-05)

### 声门区 / 非声门区二分类 Gate 基准模块
- **新增 `图像识别/glottis_binary/`**：独立生成 VOC / Non-VOC 二分类 manifest、患者级 split，并训练 ResNet、ViT、Swin、SupCon+Swin gate 基准。
- **新增并行 launcher**：`launch_parallel_benchmarks.py` 可在单卡上同时启动四个模型训练，并按 profile 给不同 backbone 分配 batch，降低 OOM 风险。
- **结果继续离开源码目录**：二分类 checkpoint、日志、metrics、图表和 provenance 默认归档到 `Results/main/glottis_binary_benchmarks/<timestamp>/`。
- **面向视频前置 gate 的阈值输出**：每个模型会保存 val 选择出的推荐声门概率阈值、test 混淆矩阵、ROC/PR、错误样本清单，便于串到 `视频识别/video_inference.py` 前。

---

## v6.23 (2026-05-05)

### 移除根目录兼容包装脚本
- **删除外层包装入口**：移除根目录 `train_phase*.py`、`train_pipeline.py` 和 `tools/*.py`，避免误以为根目录仍有独立训练/推理实现。
- **入口收敛**：图像任务统一从 `图像识别/` 运行，视频任务统一从 `视频识别/` 运行。

---

## v6.22 (2026-05-05)

### Selective Unfreeze 合入图像识别
- **合入显式 block 解冻**：`图像识别/shared.py` 支持 `unfreeze_blocks`，可直接指定 Swin stage/block，避免连续解冻末尾 block 时额外打开不需要的层。
- **更新默认实验配置**：`图像识别/config_phase1.json` 显式解冻 `stage2.block[-1]` 与 `stage3.block[-1]`；`图像识别/config_phase3.json` 显式解冻 `stage3.block[-1]`，恢复 `10%` 混淆阈值，并保留 Phase 2 checkpoint 中的 projector。
- **训练曲线标注阶段**：Phase 2 / Phase 4 生成曲线时会分别标注 CE 阶段，避免完整 pipeline 后图标题误写为 Phase 2。

---

## v6.21 (2026-05-05)

### 图像识别与视频识别目录分层
- **新增 `图像识别/`**：静态图像训练、评估、配置、冻结切分、文献图表脚本和图像 Grad-CAM 辅助工具已归入该目录。
- **新增 `视频识别/`**：视频弱监督推理脚手架归入 `视频识别/video_inference.py`，继续复用图像 checkpoint，不训练时序模型。
- **入口先行兼容**：本版本曾短暂保留根目录包装脚本；后续 v6.23 已移除这些包装入口，统一从新目录运行。

---

## v6.20 (2026-05-05)

### TBR 文件名与 Grad-CAM 标题
- **TBR 视频文件名补充病人名**：`/mnt/data/LarynxData/videos/tbr` 下的 mp4 文件已统一加上中文病人名前缀，便于离线查看原始视频和诊断输出。
- **Grad-CAM 文件名和标题显示病人名与中文病种**：`diagnosis_gradcam/` 的对照图文件名和标题现在显示中文病人名、病人编号和中文预测病种，并显式配置 Noto CJK 字体以避免中文缺字。

---

## v6.19 (2026-05-05)

### 视频诊断输出与 Grad-CAM 证据图
- **新增病人级诊断总表**：`tools/video_inference.py` 现在输出 `diagnosis_summary.csv`，按 `--diagnosis-variant`（默认 `voc_sum_gt_nonvoc`）汇总每个病人的预测病种、证据秒段、最高概率帧时间和 Grad-CAM 路径。
- **新增视频级证据表**：输出 `diagnosis_evidence.csv`，记录每个视频的预测病种、连续高概率片段、最高概率帧和低置信标记；多视频病人会在总表中选择最高概率帧作为代表证据。
- **新增 Grad-CAM 对照图**：输出 `diagnosis_gradcam/`，每个病人保存原始视频帧、模型输入帧和 Grad-CAM overlay；overlay 的白色轮廓表示 CAM 高响应区域，仅作为模型关注区域提示。

---

## v6.18 (2026-05-05)

### 视频推理低风险加速
- **顺序解码默认启用**：`tools/video_inference.py` 默认从逐采样点 seek 改为 `--decode-mode sequential`，顺序读视频并按帧间隔采样，减少 mp4 反复定位造成的 CPU 解码开销；保留 `--decode-mode seek` 便于回退对照。
- **推理吞吐提升**：默认 `--batch-size` 从 64 调到 256，并默认启用 CUDA AMP；可用 `--no-amp` 关闭混合精度。
- **结果元数据补充**：`summary.json` 记录 `decode_mode`、`batch_size` 和 `amp`，便于比较不同推理配置的速度和结果。

---

## v6.17 (2026-05-05)

### 视频数据盘迁移与 TBR 批次
- **视频目录移出 workspace**：`data/videos` 已搬到 `/mnt/data/LarynxData/videos`，避免把 1G+ 视频数据继续放在代码 workspace 下。
- **新增 TBR 待复核视频批次**：从 12 个压缩包中只抽取视频文件，整理到 `/mnt/data/LarynxData/videos/tbr/<原压缩包名>/`，共 13 个 mp4；txt 等非视频文件未放入 `tbr`。
- **视频推理默认路径更新**：`tools/video_inference.py` 默认读取 `/mnt/data/LarynxData/videos/classified_videos`，并支持 `LARYNX_VIDEO_ROOT` 覆盖。
- **支持未标注视频推理**：新增 `--allow-unlabeled`，用于 `tbr` 这类没有真实类别标签的视频；当 `--candidate-labels auto` 且没有已知标签时，会自动以所有 VOC 类作为候选类别，summary 只对已知标签计算准确率。

---

## v6.16 (2026-05-05)

### 视频帧级推理脚手架
- **新增视频弱监督推理工具**：加入 `tools/video_inference.py`，将视频按固定 fps 抽帧后复用现有图像 checkpoint，输出帧级概率、视频级聚合、候选片段和关键帧，不训练时序模型。
- **Non-VOC 只作弱 gate**：脚本同时比较全质量合格帧、`sum(VOC 概率) > Non-VOC 概率`、VOC margin、低 Non-VOC 阈值等规则，避免把 `1 - P(Non-VOC)` 当作严格声带概率。
- **支持医生标注有效秒段**：新增 manifest CSV 入口，可限定 `valid_start_sec` / `valid_end_sec` 只在清晰有效声带区间内抽帧和聚合。
- **阿里云视频 smoke test**：从 `ali:/usr/OconutWebApp/app/static/videos/datasets/larynx` 同步 5 个模型可用视频标签目录共 6 个 mp4 到 workspace 数据目录；发现 `zhang-qi-fei.mp4` 原放在 `vocal-cord-polyp` 下，但同目录 TXT 写有“诊断：任克氏水肿”，已按 Reinke-Edema 口径移动到 `benign/reinke-edema/`。修正标签后，`Results/selective_unfreeze_confusion10/best_model.pth` 在这 6 个视频上为 6/6，`Results/main/best_model.pth` 在修正前为 4/6。

---

## v6.15 (2026-05-04)

### Phase 3 阈值与 Pipeline 早停
- **混淆阈值下调**：Phase 3 默认 `phase3_confusion_threshold` 从 `10%` 改为 `5%`，更容易触发训练集混淆类对的 focused SupCon。
- **无类对时直接停在 Phase 2 结果**：pipeline 会读取 `phase3_history.json`；如果 Phase 3 因没有超过阈值的训练集混淆类对而跳过，则直接打印并归档 Phase 2 最终结果，不再运行 Phase 4。
- **Phase 2 指标落盘**：Phase 2 新增 `phase2_final_metrics.json`，记录最终 train / val / test 指标，供 pipeline 在 Phase 3 无可训练类对时输出和归档。

---

## v6.14 (2026-05-04)

### Phase 3/4 混淆聚焦训练与完成运行归档
- **新增 Phase 3**：基于 Phase 2 在 train split 上的预测错误样本统计混淆方向，超过阈值的混淆类对会触发 focused SupCon refinement；val / test 不参与 Phase 3 的错样本选择、类对选择或对比学习训练。
- **新增 Phase 4**：从 Phase 3 checkpoint 重新初始化并训练 CE classifier，默认冻结 backbone/projector，只训练分类头；`phase4_train_backbone=true` 可恢复全模型 CE 微调。
- **新增配置文件**：加入 `config_phase3.json` 与 `config_phase4.json`，分别控制混淆阈值、pair margin、Phase 3 epoch，以及 Phase 4 classifier 重训策略。
- **pipeline 扩展**：`train_pipeline.py` 默认串行运行 Phase 1 -> Phase 4，并支持 `--through-phase` 只跑到指定阶段；完整 pipeline 中 Phase 2 不做中间归档，最终归档由 Phase 4 完成。
- **每次训练自动归档**：Phase 4 完成最终 best checkpoint 评估后，会把当前活动输出目录完整复制到 `Results/runs/<完成时间>_<分支>_<commit>_testacc<ACC>_testauc<AUC>/`；单独运行 Phase 2 时仍可按 Phase 2 结果归档。
- **归档元数据**：新增 `run_metadata.json`，记录运行命令、配置、Git 分支/commit、dirty 状态、best epoch 以及最终 train / val / test 指标。
- **路径可控**：保留 `LARYNX_RESULTS_DIR` 作为活动输出目录覆盖，并新增 `LARYNX_RUNS_DIR` 控制归档根目录、`LARYNX_ARCHIVE_RUNS=0` 临时关闭归档。
- **简化 nohup 启动**：新增 `train_pipeline.py`，自动创建 `Results/<worktree>/pipeline_<时间>.log` 并串行运行 Phase 1 -> Phase 4，启动命令不再需要手写时间戳和输出重定向。
- **VRAM 评估修复**：修复 hierarchical metrics 在 VRAM loader 下对 CUDA tensor 直接调用 `.numpy()` 导致 Phase 2 收尾崩溃的问题。

---

## v6.13 (2026-05-03)

### 无空格 workspace 与 worktree/Results 分层
- **新 workspace 约定**：项目整理到 `laryngeal_multiclass/worktrees/<name>/`，代码方法版本由 Git worktree 管理，训练产物默认写到 `laryngeal_multiclass/Results/<worktree名>/`。
- **训练输出移出代码目录**：`shared.py` 自动识别 workspace 根目录，`best_model.pth`、Phase 1 checkpoint、history、metrics、图表、TensorBoard 日志和 ONNX 默认都输出到集中 Results 根目录。
- **路径可覆盖**：新增 `LARYNX_IMAGE_DIR`、`LARYNX_RESULTS_ROOT`、`LARYNX_RESULTS_DIR`、`LARYNX_WORKSPACE_DIR` 环境变量支持；旧配置里的 `Results/phase1_checkpoint.pth` 会解析到当前 worktree 对应的 Results 目录。
- **文档置顶提醒**：README 第一行明确说明代码改动和训练产物的归档位置，避免把 checkpoint 或运行日志写进 worktree。

---

## v6.12 (2026-05-03)

### 连续显存缓存与 GPU 索引路径收敛
- **Split 连续写入显存**：`preload_image_cache` 不再按路径排序打散样本，而是按传入的 train / val / test 顺序写入同一块连续 `uint8 CHW` 缓存，使每个 split 在缓存中天然成为连续区间。
- **评估直切连续显存**：`VRAMDataLoader` 会检测 split 是否连续；无 shuffle / sampler 的 train_eval、val、test 路径直接使用显存切片，减少每个 batch 的索引 gather。
- **训练采样权重常驻 GPU**：balanced sampler 权重在 loader 初始化时搬到目标设备并转为 `float32`，每个 epoch 直接用 GPU `multinomial` 生成本轮索引，避免重复从 CPU 取权重。
- **跳过无意义 CUDA 预取包装**：当 batch 已经常驻 GPU 时，训练/评估循环不再额外套 `CUDAPrefetcher`；CPU-backed fallback 仍保留原预取路径。
- **Normalize 常量缓存**：ImageNet mean/std 按 device + dtype 缓存，避免每个 batch 重复创建小 GPU tensor。

---

## v6.11 (2026-05-01)

### VRAM 数据预加载与无多进程加载优化
- **全显存数据缓存**：重构 `preload_image_cache`，根据 JSON 将所有预处理后的图片直接转化为 `uint8 CHW` 张量并常驻 GPU 显存 (VRAM)。在大规模下占用极小且完全消除了训练期间的主内存-显存拷贝瓶颈。
- **关闭 num_workers 并行加载**：由于数据已全部加载至显存，废除 `num_workers` 多进程加载与 `pin_memory` 机制，强制设为 0，避免 CUDA 张量跨进程共享报错及调度开销。
- **纯 GPU 数据增强保留**：在显存中直接调度数据，并完美衔接原有的 `GPUAugment` 流水线，实现真正意义上“显存直接调取数据 -> 显存内做数据增强 -> 显存内前向反向传播”的极致训练效率。

---

## v6.10 (2026-05-01)

### 模型对比实验与论文式对照图
- **新增 baseline 对比脚本**：加入 `tools/train_comparison_models.py`，按当前 8 类冻结 split 训练 `MobileNetV2`、`DenseNet121`、`ResNet50` 等经典 ImageNet 预训练基线。
- **新增模型对比目录**：`Results/model_comparison/` 按模型子文件夹保存 checkpoint、history、summary/per-class metrics 和 image-level predictions，并生成 combined comparison CSV。
- **新增顶刊风格对比图**：用点线小面板、分组 overview、逐类 F1 热力图和 VOC 二分类指标图替代大色块柱状图，并同时导出 PNG/PDF。
- **区分证据范围**：内部测试集结果与外部文献 reference 分开呈现；文献 SOTA 行仅作 context，不参与直接排名。
- **当前 checkpoint 口径提示**：检测到 `best_model.pth` 早于 v6.9 split 文件，README 标注当前模型行可作视觉审阅，正式 strict same-split 结论需在 v6.9 split 上重新训练 Phase 1/2。

---

## v6.9 (2026-05-01)

### 冻结切分改为 8 类口径
- **切分生成阶段排除声带固定**：`build_multiclass_split.py` 默认排除 `Vocal-Cord-Fixation` / `声带固定`，重新生成的 `dataset_split.json` 不再包含该类。
- **切分审计同步**：新 split 记录 `声带固定` 为显式排除来源文件夹，8 类总计 12,631 张图片，患者组 train/val/test 无重叠。
- **配置残留清理**：移除两份配置中的声带固定类别、显式排除项，以及 Phase 1 知识图谱中声带固定相关系数；`shared.py` 默认类别同步为 8 类。
- **文档同步**：README 更新为当前 8 类冻结切分口径。

---

## v6.8 (2026-05-01)

### Q1 文献对齐评估与论文式图表
- **新增 Q1 文献矩阵**：下载并归档 3 篇 2025 下半年至 2026 上半年医学视觉模型相关开放原文，记录期刊分区依据、模型、评价指标和图表类型。
- **新增论文式评估脚本**：加入 `tools/literature_aligned_metrics.py`，基于当前 `best_model.pth` 与冻结患者级切分导出 Accuracy、Balanced Accuracy、Precision、Recall/Sensitivity、Specificity、F1、AUROC、AUPRC、VOC/Non-VOC 二分类指标及 bootstrap 95% CI。
- **新增论文式图表输出**：生成混淆矩阵、one-vs-rest ROC/PR 曲线、VOC-vs-NonVOC ROC/PR/混淆矩阵、逐类指标条形图、类别支持数、训练曲线、测试集特征 t-SNE，并归档现有 Grad-CAM 图。
- **新增类别均衡 Grad-CAM**：加入 `tools/generate_class_balanced_gradcam.py`，按类别随机抽取测试样本，并将原图与 Grad-CAM 成对展示；标题和边框用绿色/红色区分预测对错。
- **优化论文图表工具性能**：评估脚本复用配置化 eval batch、多进程 DataLoader、预取和图像缓存；t-SNE 特征采集避免重复 backbone 前向，误分类收集脚本避免 hierarchical 默认路径重复推理。
- **文档同步**：README 新增文献对齐评估图表说明，便于后续换模型后一键复跑。

---

## v6.7 (2026-04-28)

### Training Curve 坐标与口径澄清
- **概率类指标统一坐标**：Phase 2 的 Macro-F1 / Accuracy / AUROC 子图统一使用 0–1 纵轴，避免自动缩放把正常的局部差距视觉放大。
- **Train 曲线口径标注**：图例明确 train 曲线来自训练增强与 balanced-sampler batch；README 补充说明，过拟合判断优先看 `Results/metrics.csv` 中无随机增强的 train_eval / val / test 最终评估。

---

## v6.6 (2026-04-28)

### Training Curve 显示口径修正
- **默认隐藏 Test 曲线**：`Results/history.csv` 仍保留每轮 test 诊断指标，但 `Results/training_curves.png` 只绘制 train / val 曲线，避免在训练过程图里展示测试集表现。

---

## v6.5 (2026-04-28)

### 静态单图训练口径修正
- **默认排除 Vocal-Cord-Fixation**：旧 9 类运行中 `Vocal-Cord-Fixation` 在 train/val/test 都显著低 recall，并主要与 `Non-Vocal-Cord` 互相混淆；该标签更依赖动态声带运动证据，不适合作为当前静态单帧图像的默认分类目标。
- **配置化冻结切分类别过滤**：新增 `excluded_classes_from_split`，训练入口仍读取患者级冻结 `dataset_split.json`，但会在运行时过滤不参与当前任务的类别，不改原始数据和 split 文件。
- **文档同步为 8 类默认任务**：README 明确当前默认启用 8 类，原始 `dataset_split.json` 仍保留 9 类切分；如需恢复声带固定类别，可清空 `excluded_classes_from_split` 后完整重跑 Phase 1/2。

---

## v6.4 (2026-04-28)

### 九分类过拟合与指标诊断修正
- **Phase 1 checkpoint 改按验证 SupCon loss 选择**：KG bilevel 训练启用 `supcon_monitor=val_loss`，避免训练 loss 后期继续下降时把验证损失已经反弹的表征保存给 Phase 2。
- **Phase 2 选择指标偏向 macro-F1**：val composite score 改为 `0.7 × macro-F1 + 0.3 × AUROC`，并加入 `early_stopping_min_delta=0.001`，减少 AUROC 平台期对少数类 F1 问题的掩盖。
- **温和类别均衡采样**：Phase 2 新增 `sampler_balance_alpha=0.65`，从九类完全等权采样改为温和逆频率采样，降低小样本类别被重复抽样记忆的风险。
- **喉镜方向与细节友好增强**：关闭垂直翻转，降低随机裁剪、仿射、模糊和锐化强度，避免破坏上下方向、病灶边缘和黏膜纹理。
- **测试曲线对齐当前 epoch**：Phase 2 每轮同步记录 test 诊断指标，保证 `Results/history.csv` 与 `training_curves.png` 中的 test 曲线不再滞后一轮。

---

## v6.3 (2026-04-28)

### 两阶段 GPU 增强流水线
- **GPU 图像增强**：Phase 1 SupCon 与 Phase 2 CE 训练共享新的 `GPUAugment`，随机裁剪、翻转、仿射、颜色扰动、模糊、锐化和 ImageNet normalize 都在 batch 到达 GPU 后执行。
- **CPU 基础预处理缓存**：CPU 侧只做黑边裁切、Resize、CenterCrop，并缓存未归一化 `uint8 CHW` 张量，减少每个 epoch 的 PIL/torchvision 随机增强开销。
- **DataLoader 同步配置化**：两阶段显式复用 `prefetch_factor`、`persistent_workers` 与 CUDA-aware `pin_memory` 设置。
- **Batch 策略调整**：Phase 1 SupCon batch 提升到 256 并关闭梯度累积；Phase 2 训练 batch 提升到 256、评估 batch 提升到 512，同样使用真实 batch 更新而不再累积梯度。

---

## v6.2 (2026-04-28)

### 极简两阶段训练包
- **最小运行文件**：当时项目根目录只保留 `train_phase1.py`、`train_phase2.py`、`shared.py`、两份 phase 配置、`dataset_split.json`、README 和 CHANGELOG；当前入口已在 v6.23 收敛到 `图像识别/` 与 `视频识别/`。
- **删除训练产物与非训练工具**：移除 `Results/`、旧 `best_model.pth`、ONNX 导出/量化脚本、独立 GradCAM 脚本、重新切分脚本、部署文档和 LAST-ViT 归档文档。
- **文档同步**：README 改为当前极简训练包说明，不再引用已删除的部署、ONNX、重新切分或历史产物文件。

---

## v6.1 (2026-04-28)

### 两阶段训练清理与切分类别同步
- **冻结切分类别优先**：训练、导出与可视化加载配置时会优先使用 `dataset_split.json` 中的 `class_folders`，当前训练口径与冻结切分保持为 9 类。
- **配置同步**：`config_phase1.json` / `config_phase2.json` 已移除当前冻结切分不包含的声带黏连、功能性声带运动不全、声带小结；Phase 1 知识图谱也同步删去这些类别相关边。
- **无用文件清理**：移除旧单脚本 `config.json`、历史日志、`nohup.out`、项目内误放的 fan 配置和 Python 缓存文件；当时训练入口只保留 `python train_phase1.py` 与 `python train_phase2.py`，当前入口已在 v6.23 收敛到 `图像识别/` 与 `视频识别/`。

---

## v6.0 (2026-04-28)

### 良性病变拆分为多分类任务
- **多分类标签体系**：将原 `Lesion` 聚合类拆分为声带任克水肿、囊肿、固定、息肉、白斑、肉芽肿、黏连、功能性声带运动不全、小结等具体疾病类别。
- **动态层级逻辑**：`shared.py` 不再写死四分类与 3 个 VOC 子类；除 `non_voc_class` 外的配置类会自动作为 VOC 类参与层级推理、平衡采样、指标统计与 ONNX 元数据导出。
- **知识图谱更新**：`config_phase1.json` 的 `knowledge_graph.class_similarity` 改为面向具体疾病类别的相关性矩阵，用于 Knowledge-Guided SupCon 预训练。
- **文档与部署同步**：README 与 DEPLOYMENT 改为多分类说明，并明确旧四分类 checkpoint 需要重新训练后才能用于 Phase 2 / ONNX 导出。

---

## v5.0 (2026-04-17)
*(包含旧版本 v4.5, v4.6)*

### 数据泄露审计、统计学方法修复与数据划分冻结
这标志着项目进入了一个更加严谨和可重复的阶段，彻底解决了先前的隐式环境依赖和数据泄露隐患，此前由于数据跨集泄露，需要进行完整的重训。
- **患者级数据切分冻结**：新增 `freeze_split.py` 将 `split_patients()` 生成结果一次性冻结到 `dataset_split.json`，分离划分逻辑，两阶段训练直接读取 JSON 以确保一致性，解决环境多变引发的数据漂移可能。
- **Phase 1/2 种子对齐**：修复了 Phase 1 (`seed:42`) 和 Phase 2 (`seed:43`) 种子不一致导致跨训练与测试集交集干扰严重数据泄露问题。现统一为 `seed: 42`。
- **验证集增强隔离**：修复 Phase 1 中 bilevel KG 验证集 DataLoader 误用训练增强 (`train_tf`) 的问题，严格应用评估无参增强 (`eval_tf`)。
- **实时监控说明**：澄清并在训练循环中加入异步测试集监控用于观察。测试集只用于绘图诊断，不参与最佳模型决策与 early stopping。
- **清理重复项**：删除了 `config_phase1.json` 与 `config_phase2.json` 中的多余的 `"声带白斑"` 标签。
- **理论文献归档**：新增 `last_vit.md` 文档，总结了 LAST-ViT 的注意力频域修改机制与核心理论实现。

---

## v4.4 (2026-04-02)
*(包含旧版本 v4.2, v4.3, v4.4)*

### 部署导出流水线优化及学习率调整
- **ONNX 导出解耦并升级**：将 ONNX 导出逻辑从训练流程拆离至独立脚本 `export_onnx.py`。新增 FP16 单文件导出支持，大幅缩减模型体积（约 171.2 MB）并在保留 INT8 支持的同时规避了旧方案多文件分散、体积受限的问题。
- **学习率上下限优化 (Phase 1)**：提高全局学习率至 1e-3 并增强余弦退火学习率下限（`supcon_min_lr`=3e-4），成功解决 Phase 1 后期 loss 停滞的问题，充分激发后期优化效果。
- **GradCAM 逻辑提升**：允许独立指定 checkpoint、config 路径与自定义参数名称输出，并在图片上通过色彩编码明确标注预测结果是正确或错误。

---

## v4.1 (2026-04-02)

### 相似度矩阵坍缩修复
- **KG 坍缩修复**：解决学习模块优化失灵问题；将 `kg_lr_multiplier` 降至 0.5 并引入 `kg_anchor_strength=5.0` L2 锚定正则计算，防止知识图谱（Knowledge Graph）自由偏离解剖逻辑先验跌至 0，保障模型不采取类似同类同拉的捷径。

---

## v4 (2026-04-01)

### 知识图谱引导的 SupCon与CUDA并行流水线优化
- **Knowledge-Guided SupCon (Bilevel Optimization)**：重构传统的对比学习损失定义。允许类间提供软对齐差异（如 Normal-Lesion 为 0.3 的距离）。采取 Bilevel 异步内/外循环更新，同时学到模型权重及最佳知识矩阵。
- **CUDA 流水线提速**：加入了独立的 `CUDAPrefetcher` 隐藏张量拷贝流等待。增加了基于独立流的 `AsyncCheckpointSaver` 使得模型写入不阻断 GPU 推进。扩大 batch size 以提升硬件饱和度。

---

## v3 (2026-03-26 - 2026-03-28)
*(包含旧版本 v3.0, v3.1, v3.2)*

### 两阶段流程拆解、服务部署方案及量化
- **管线架构彻底解耦**：解决多段管线在同一个代码里的隐患。彻底拆分成 `train_phase1.py` 和 `train_phase2.py`。新增对应 `config` 与集成公有工具特性的 `shared.py`。
- **ONNX及服务端指北**：提供可供复线的完整 `DEPLOYMENT.md`，实现从 675MB (FP32) 通过 `quantize_model.py` 到 91MB (INT8) 的转换，大幅度适配低显存的廉价云端服务器部署。
- **GradCAM 并跑数据错乱修复**：修复因多线程评估引起的与 `generate_attention_maps` 对 TIMM 输出钩子的捕获干扰。同时纠正了钩子点至最后一个块而不只是归一化层，根治注意力矩阵产生的"竖形条纹"废旧数据。

---

## v2 (2026-03-25 - 2026-03-26)
*(包含旧版本 v2.0, v2.1, v2.2)*

### CE 微调抗过拟合设计
- **强化正则防御网络**：Phase 2 中调整 Classifier `dropout_rate` 加大至 0.4 并同时将内块丢包深度 `drop_path_rate` 升级到 0.25；对解冻末尾 block 数降为 1。特别新增 `Feature Dropout` 在核心分类特征抽取前进行特征截断（0.2 的率），防止高维表示通道产生过拟合依赖。
- **参数收敛与余弦退火**：额外加入 `min_lr`，提供有力的抗学习率过度坠落支持机制。

---

## v1 (2026-03-25)

### Swin-B 模型范式转换
- **Backbone 翻新演化**：舍弃原本依靠全局注意力的 ViT-B/16 (384) 引入 Shifted Window (局部 → 全局) 控制更精准、特征聚合更多元的 Swin-B (224 尺寸)。
- **向下的兼容适配**：顺利过渡层级推理（VOC vs Non-VOC）判断、层级平衡分类抽取机制。重置匹配了新网络深度层的学习递减惩罚系数。

# CHANGELOG

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
- **最小运行文件**：项目根目录只保留 `train_phase1.py`、`train_phase2.py`、`shared.py`、两份 phase 配置、`dataset_split.json`、README 和 CHANGELOG。
- **删除训练产物与非训练工具**：移除 `Results/`、旧 `best_model.pth`、ONNX 导出/量化脚本、独立 GradCAM 脚本、重新切分脚本、部署文档和 LAST-ViT 归档文档。
- **文档同步**：README 改为当前极简训练包说明，不再引用已删除的部署、ONNX、重新切分或历史产物文件。

---

## v6.1 (2026-04-28)

### 两阶段训练清理与切分类别同步
- **冻结切分类别优先**：训练、导出与可视化加载配置时会优先使用 `dataset_split.json` 中的 `class_folders`，当前训练口径与冻结切分保持为 9 类。
- **配置同步**：`config_phase1.json` / `config_phase2.json` 已移除当前冻结切分不包含的声带黏连、功能性声带运动不全、声带小结；Phase 1 知识图谱也同步删去这些类别相关边。
- **无用文件清理**：移除旧单脚本 `config.json`、历史日志、`nohup.out`、项目内误放的 fan 配置和 Python 缓存文件；当前训练入口只保留 `python train_phase1.py` 与 `python train_phase2.py`。

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

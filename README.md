**项目约定：代码/方法改动在 `laryngeal_multiclass/worktrees/<分支名>/` 中提交；图像任务入口放在 `图像识别/`，视频任务入口放在 `视频识别/`；训练、评估、图表和 checkpoint 默认输出到 `laryngeal_multiclass/Results/<worktree名>/`，不要写回 worktree 代码目录。**

# Laryngeal Multi-class Classification

基于 Swin-B (224) 的喉镜图像层级多分类项目。训练流程先使用医学知识图谱引导的 Supervised Contrastive Learning 预训练，让不同疾病之间的相关性矩阵参与表征学习，再使用 Cross-Entropy Learning 做最终分类微调。

本项目从原四分类版本复制而来，并将原先聚合的 `Lesion`（良性病变）拆分为当前数据集中的具体疾病类型。

## 目录结构

| 目录 / 文件 | 说明 |
|------|------|
| `图像识别/` | 静态图像 8 分类训练、评估、Grad-CAM 和论文式图表脚本 |
| `图像识别/glottis_binary/` | 面向视频前置 gate 的声门区 / 非声门区二分类基准模块 |
| `图像识别/roi_reflection/` | BAGLS 驱动的 ROI 有效性 / 反光干扰 sidecar 与 gate 工作流 |
| `视频识别/` | 抽帧复用图像 checkpoint 的视频弱监督推理脚手架 |
| `README.md` / `CHANGELOG.md` | 项目总说明与变更记录 |

根目录不再保留训练或推理包装脚本；请直接使用 `python 图像识别/...` 或 `python 视频识别/...`。

## 层级分类逻辑

```
输入: 喉镜图片
        ↓
  Level 1: 是否声带图片 (VOC vs Non-VOC)
        ↓
  Level 2 (仅VOC): Normal / 具体良性病种 / Cancer
        ↓
输出: {Non-Vocal-Cord} 或 {Normal, Reinke-Edema, Vocal-Cord-Cyst, Vocal-Cord-Polyp, Vocal-Cord-Leukoplakia, Vocal-Cord-Granuloma, Cancer}
```

## 类别映射

训练时以 `图像识别/dataset_split.json` 中冻结的 `class_folders` 为基础；当前冻结切分按静态单图任务启用 8 类：

| 标签 | 默认来源文件夹 | 是否声带 |
|------|---------------|---------|
| Non-Vocal-Cord | 混杂图片 | ✗ |
| Normal | 正常 | ✓ |
| Reinke-Edema | 声带任克水肿 | ✓ |
| Vocal-Cord-Cyst | 声带囊肿 | ✓ |
| Vocal-Cord-Polyp | 声带息肉 | ✓ |
| Vocal-Cord-Leukoplakia | 声带白斑 | ✓ |
| Vocal-Cord-Granuloma | 声带肉芽肿 | ✓ |
| Cancer | 喉癌 | ✓ |

`图像识别/dataset_split.json` 已在生成阶段排除 `Vocal-Cord-Fixation`，因为声带固定更依赖发声/呼吸过程中的运动受限证据，单张静态喉镜图像容易把它混入 Non-VOC 或普通 VOC 外观。当前 `声带固定`、`声带黏连`、`功能性声带运动不全`、`声带小结` 已被本次冻结切分排除，不参与训练。

## 与 ViT 版本的区别

| 特性 | 本项目 (Swin-B) | laryngeal_4class_voc_vs_nonvoc (ViT-B/16) |
|------|-----------------|-------------------------------------------|
| Backbone | swin_base_patch4_window7_224 | vit_base_patch16_384 |
| 输入尺寸 | 224×224 | 384×384 |
| 注意力机制 | Shifted Window (局部 → 全局) | Global Self-Attention |
| Block 结构 | 4 stages [2, 2, 18, 2] = 24 blocks | 12 uniform blocks |
| 参数冻结 | 解冻最后 N 个 block（跨 stage） | 解冻最后 N 个 block |
| 预训练权重 | ImageNet-22k → ImageNet-1k | ImageNet-22k → ImageNet-1k |

## 方法论

### SupCon 预训练设计

默认启用 **KnowledgeGuidedSupConLoss**；关闭 `knowledge_graph.enabled` 时退回 **HierarchicalSupConLoss**。两者都会在投影空间中显式建模层级结构：

1. **Non-Vocal-Cord 样本**：聚成一簇，与 VOC 区域分离
2. **Vocal-Cord 样本**：在另一区域进一步细分
3. **VOC 内部**：Normal、各个良性病种、Cancer 保持适度分离
4. **VOC vs Non-VOC Margin 约束**：对每个 Non-VOC 样本，强制其与所有 VOC 样本的相似度低于 `voc_margin`（默认 0.3），使两类在投影空间中保持明确边界

### 推理阶段层级决策

```
1. 计算多分类 softmax 概率
2. 判断: sum(VOC 类概率) vs Non-VOC 概率
3. 若 VOC 获胜: 在所有 VOC 类中取 argmax
4. 若 Non-VOC 获胜: 输出 Non-Vocal-Cord
```

## 数据集

- 来源：默认自动查找 `../../Laryngeal_Dataset_Processed/` 或上级 `Larynx/Laryngeal_Dataset_Processed/`；也可用 `LARYNX_IMAGE_DIR=/path/to/data` 显式指定（不做任何改动）
- **患者级** 8:1:1 切分（train / val / test），避免同一患者图像跨集泄露
- 切分结果以 JSON 形式**冻结**在 `图像识别/dataset_split.json` 中，各阶段训练脚本只读不重算 —— 杜绝因运行时随机环境变化导致训练/评估划分漂移
- 自动黑边裁切（`CropBlackBorders`）在训练与推理中均生效

`图像识别/` 是极简训练包，只保留现成的 `dataset_split.json`。如果以后要改变类别集合或重新切分数据，需要重新引入/编写切分脚本并重新训练图像 checkpoint。

## 训练流程

当前默认入口是**四阶段 pipeline**：

1. **Phase 1 — Knowledge-Guided Hierarchical SupCon 预训练**：将同类样本拉近、异类样本推远，同时用 `knowledge_graph.class_similarity` 中的疾病相关性矩阵为医学上相近的类别提供软正样本权重。只训练 backbone + projection head。
2. **Phase 2 — CE 微调**：用 Phase 1 学到的 backbone 作为起点，训练分类头做标准交叉熵分类。
3. **Phase 3 — Train-confusion-focused SupCon**：只基于 Phase 2 在 train split 上的错分方向选择混淆类对，做 focused SupCon refinement。
4. **Phase 4 — Classifier retraining**：从 Phase 3 checkpoint 出发重置并重训分类头，最终 checkpoint 仍只按 val composite score 选择。

`图像识别/train_pipeline.py` 默认串行运行 Phase 1 -> Phase 4；如果 Phase 3 没有超过阈值的混淆类对，pipeline 会停在 Phase 2 并归档 Phase 2 结果。

### 正则化策略

| 手段 | 说明 | 当前值 |
|------|------|--------|
| Feature Dropout | backbone 输出到 classifier 前的 dropout | dropout_rate × 0.5 |
| Classifier Dropout | 分类头隐藏层后的 dropout | dropout_rate |
| Drop Path | Swin block 中的随机深度 | drop_path_rate |
| Label Smoothing | 标签平滑 | label_smoothing |
| Weight Decay | AdamW L2 正则 | weight_decay |
| Layer-wise LR Decay | 浅层使用更小的学习率 | layer_decay |
| 参数冻结 | 解冻末尾 N 个 block，或用 `unfreeze_blocks` 显式指定 stage/block | unfreeze_last_n_blocks / unfreeze_blocks |
| Early Stopping | 监控 val composite score = 0.7 × macro-F1 + 0.3 × AUROC，并使用 min_delta 抑制平台期噪声 | early_stopping_patience / early_stopping_min_delta |

### 本轮过拟合诊断口径

当前切分中 `Non-Vocal-Cord` 接近测试集一半，Accuracy 容易被大类与 VOC/Non-VOC 层级判断抬高；因此主要用 macro-F1、逐类 recall/F1 与 AUROC train-vs-val/test gap 判断是否泛化。旧 9 类运行里 `Vocal-Cord-Fixation` 在训练集、验证集和测试集均长期低 recall，且主要与 Non-VOC 互相混淆；这更像静态单图任务定义问题，而不是单纯训练轮数不够。Phase 1 checkpoint 继续优先按 `val_loss` 选择，避免把过拟合表征传入 Phase 2。

### 类别平衡采样

基础训练阶段通过 `WeightedRandomSampler` 实现采样平衡：

| 阶段 | 策略 | 有效采样比例 |
|------|------|-------------|
| Phase 1 SupCon | 层级平衡：VOC=Non-VOC 各 50%，VOC 内所有细分类等权 | Non-VOC 50%，VOC 类共享 50% |
| Phase 2 CE | 温和逆频率采样，`sampler_balance_alpha=0.65`；不再叠加 `class_weights` 加权 loss | 少数类被提升，但不再像类别完全等权那样重复抽样过多 |

## 配置

超参数分别集中在 `图像识别/config_phase1.json` 和 `图像识别/config_phase2.json` 中，无需修改 Python 代码即可调参。类别映射会在运行时与 `图像识别/dataset_split.json` 同步，并应用 `excluded_classes_from_split`。

### CE 微调参数

| 参数 | 说明 | 默认值 |
|------|------|----------|
| `epochs` | CE 最大训练轮数 | 80 |
| `batch_size` | 训练 batch | 256 |
| `eval_batch_size` | 评估 batch | 512 |
| `grad_accum` | 梯度累积步数 | 1 |
| `learning_rate` | 峰值学习率 | 5.5e-5 |
| `unfreeze_last_n_blocks` | Swin 解冻的末尾 block 数 | 1 |
| `unfreeze_blocks` | 可选：显式解冻指定 Swin block；设置后覆盖 `unfreeze_last_n_blocks`，例如 `{"stage": 2, "block": -1}` 表示 stage 2 最后一个 block | null |
| `layer_decay` | 层级学习率衰减系数 | 0.65 |
| `dropout_rate` | 分类头 Dropout（feature dropout = rate × 0.5） | 0.4 |
| `drop_path_rate` | Swin 随机深度 | 0.25 |
| `label_smoothing` | 标签平滑 | 0.1 |
| `weight_decay` | AdamW 权重衰减 | 0.08 |
| `sampler_balance_alpha` | Phase 2 逆频率采样强度，1.0 为当前启用类别等权，0.0 为自然分布 | 0.65 |
| `excluded_classes_from_split` | 可选：从冻结切分中临时排除类别；当前默认无需显式排除 | [] |
| `selection_f1_weight` / `selection_auc_weight` | val composite score 权重 | 0.7 / 0.3 |
| `early_stopping_patience` | 早停耐心 | 6 |
| `early_stopping_min_delta` | 早停最小改进幅度 | 0.001 |

### SupCon 预训练参数

| 参数 | 说明 | 默认值 |
|------|------|----------|
| `supcon_enabled` | 是否启用 SupCon 预训练 | true |
| `supcon_batch_size` | SupCon 阶段 batch size | 256 |
| `supcon_epochs` | SupCon 训练轮数 | 150 |
| `supcon_early_stopping_patience` | SupCon 早停耐心 | 10 |
| `supcon_early_stopping_min_delta` | SupCon 最小改进幅度 | 0.001 |
| `supcon_monitor` | Phase 1 checkpoint / early stopping 监控指标 | val_loss |
| `supcon_learning_rate` | SupCon 峰值学习率 | 1e-3 |
| `supcon_temperature` | 对比损失温度系数 | 0.1 |
| `supcon_projection_dim` | Projection head 输出维度 | 128 |
| `supcon_warmup_epochs` | SupCon warmup 轮数 | 3 |
| `supcon_voc_margin` | VOC/Non-VOC 分离边界 | 0.3 |

### 数据增强参数

图像训练现在采用极速的显存直读流水线：根据 `图像识别/dataset_split.json` 将所有基础预处理（黑边裁切、Resize、CenterCrop）后的图像，以未归一化的 `uint8 CHW` 张量形式**按 train / val / test 顺序写入一块连续 GPU 显存 (VRAM) 缓存**。训练时 GPU 直接从显存调用张量批次，均衡采样权重常驻 GPU，废除 `num_workers` 多进程加载开销，并在 GPU 内直接执行随机增强和 ImageNet normalize。评估、测试、GradCAM 同样享受显存直读速度；顺序 split 会直接用连续显存切片，不注入随机增强。

| 参数 | 说明 | 默认值 |
|------|------|----------|
| `gpu_augment_enabled` | 是否在 GPU 上执行训练随机增强 | true |
| `prefetch_factor` / `persistent_workers` | 已废除（由于数据常驻显存，已强制设置 num_workers=0） | - |
| `random_affine_degrees` | 随机仿射旋转角度 | 10 |
| `random_affine_translate` | 随机仿射平移比例 | [0.08, 0.08] |
| `random_affine_scale` | 随机仿射缩放比例 | [0.9, 1.1] |
| `random_horizontal_flip_prob` | 随机水平翻转概率 | 0.5 |
| `random_vertical_flip_prob` | 随机垂直翻转概率 | 0.0 |
| `gaussian_blur_prob` / `gaussian_blur_sigma_max` | 高斯模糊概率 / 最大 sigma | 0.2 / 2.0 |
| `random_adjust_sharpness_prob` / `random_adjust_sharpness_factor` | 随机锐化概率 / 强度 | 0.2 / 1.5 |
| `random_resized_crop_scale_min` | RandomResizedCrop 最小比例 | 0.85 |

## 训练

推荐使用 pipeline 脚本串行运行完整训练：

```bash
# 自动生成带时间戳的 pipeline 日志，并串行运行 Phase 1 -> Phase 4
nohup setsid python -u 图像识别/train_pipeline.py &
```

也可以只跑到指定阶段：

```bash
python 图像识别/train_pipeline.py --through-phase 2
python 图像识别/train_pipeline.py --through-phase 3
```

各阶段也可以单独运行：

```bash
# Phase 1: SupCon 预训练（配置：config_phase1.json）
python 图像识别/train_phase1.py

# Phase 2: CE 微调（配置：config_phase2.json，必须能加载 Phase 1 checkpoint）
python 图像识别/train_phase2.py

# Phase 3: 仅基于 train split 的 Phase 2 混淆，做 focused SupCon
python 图像识别/train_phase3.py

# Phase 4: 从 Phase 3 checkpoint 重新训练 CE classifier
python 图像识别/train_phase4.py
```

也支持自定义 config 路径：

```bash
python 图像识别/train_phase1.py --config my_phase1_config.json
python 图像识别/train_phase2.py --config my_phase2_config.json
python 图像识别/train_phase3.py --config my_phase3_config.json
python 图像识别/train_phase4.py --config my_phase4_config.json
```

自定义配置中的类别映射不会直接覆盖当前冻结切分；如需新增类别、合并类别或重做患者级划分，请先用目标配置重新生成 `图像识别/dataset_split.json`，再重新运行完整训练。

Phase 1 显式解冻 `stage2.block[-1]` 与 `stage3.block[-1]` 两个 Swin block；Phase 3 只解冻 `stage3.block[-1]`，并沿用 Phase 1 训练出的 projector，不重新初始化 projector。

Phase 3 严格只用 Phase 2 在 train split 上的预测错误样本统计混淆方向；默认阈值为 `10%`。若 A 类中超过阈值的样本被预测为 B，或 B 类中超过阈值的样本被预测为 A，就把 `{A, B}` 作为一个无序混淆类对纳入 focused SupCon；多个混淆类对会合并其涉及类别，并对每个选中的 pair 施加额外 pair margin。val / test 不参与 Phase 3 的错样本选择、类对选择或对比学习训练。若没有任何混淆类对超过阈值，pipeline 会停止在 Phase 3 后，保留并输出 Phase 2 结果，不再运行 Phase 4。

Phase 2 和 Phase 4 训练过程中同时监控 train / val / test 三套指标，**模型选择（early stopping + checkpoint）仅依据 val composite score = 0.7 × macro-F1 + 0.3 × AUROC**，测试集指标不参与任何训练决策。test 指标继续写入 `../../Results/<worktree名>/history.csv` 和 TensorBoard 便于离线诊断，但默认 `Training Curve` 只展示 train / val 曲线，避免把测试集表现放进训练过程图。图中的 train 曲线来自训练时的随机增强与 balanced-sampler batch，适合观察优化过程；判断 train/val 泛化差距时优先看 `../../Results/<worktree名>/metrics.csv` 中无随机增强的 train_eval / val / test 最终评估。

> 注意：旧四分类 checkpoint 以及本次排除 `Vocal-Cord-Fixation` 前的 9 类 checkpoint 都与当前 8 类分类头不兼容。调整类别口径后请重新运行完整图像训练 pipeline。

## 输出文件

当前 worktree 不保留训练产物；运行脚本时会先在 `../../Results/<worktree名>/` 写入本次活动输出。Phase 4 完成最终评估后，会自动把完整活动输出复制归档到 `../../Results/runs/<完成时间>_<分支>_<commit>_testacc<ACC>_testauc<AUC>/`，其中 `ACC` / `AUC` 来自 final best checkpoint 的最终 test 评估。可用 `LARYNX_RESULTS_DIR=/absolute/path` 覆盖活动输出目录，用 `LARYNX_RUNS_DIR=/absolute/path` 覆盖归档根目录；如临时不想归档，可设置 `LARYNX_ARCHIVE_RUNS=0`。单独运行 Phase 2 时仍会按 Phase 2 结果归档。

| 文件 | 来源 | 说明 |
|------|------|------|
| `../../Results/<worktree名>/phase1_checkpoint.pth` | Phase 1 | SupCon 预训练后的完整模型权重 |
| `../../Results/<worktree名>/phase1_history.json` | Phase 1 | SupCon 每 epoch 的 loss/lr 记录 |
| `../../Results/<worktree名>/logs_phase1/` | Phase 1 | TensorBoard 日志 |
| `../../Results/<worktree名>/pipeline_<时间>.log` | Pipeline | `图像识别/train_pipeline.py` 自动生成的完整 Phase 1 -> Phase 4 控制台日志 |
| `../../Results/<worktree名>/pipeline_latest.log` | Pipeline | 指向最新 pipeline 日志的便捷链接或路径记录 |
| `../../Results/<worktree名>/phase2_best_model.pth` | Phase 2 | Phase 2 最优权重副本，供 Phase 3 读取，避免被 Phase 4 最终模型覆盖 |
| `../../Results/<worktree名>/phase2_final_metrics.json` | Phase 2 | Phase 2 最终 train / val / test 指标，供 pipeline 在 Phase 3 无可训练类对时输出和归档 |
| `../../Results/<worktree名>/phase3_checkpoint.pth` | Phase 3 | 训练集混淆类对 focused SupCon 后的完整模型权重 |
| `../../Results/<worktree名>/phase3_train_confusion_matrix.csv` | Phase 3 | 仅基于 train split 的 Phase 2 混淆矩阵 |
| `../../Results/<worktree名>/phase3_train_confusion_pairs.csv` | Phase 3 | train split 中超过阈值的混淆方向与被选中的无序类对 |
| `../../Results/<worktree名>/phase3_train_misclassified_samples.csv` | Phase 3 | Phase 2 在 train split 上预测错误的样本清单 |
| `../../Results/<worktree名>/phase4_best_model.pth` | Phase 4 | Phase 4 classifier retraining 的最优权重副本 |
| `../../Results/<worktree名>/best_model.pth` | Phase 4 | 最终最优模型权重（按 val composite score = 0.7 × macro-F1 + 0.3 × AUROC 选取） |
| `../../Results/<worktree名>/history.csv` | Phase 4 | 每 epoch 的 train/val 指标与 test 诊断指标 |
| `../../Results/<worktree名>/metrics.csv` | Phase 4 | 最终三集上的分类报告（含层级 VOC 准确率） |
| `../../Results/<worktree名>/training_curves.png` | Phase 4 | F1 / Acc / AUC 训练曲线（含 Phase 1 曲线；默认仅显示 train / val） |
| `../../Results/<worktree名>/gradcam_maps.png` | Phase 4 | GradCAM 注意力可视化 |
| `../../Results/<worktree名>/logs_phase2/` | Phase 2 | TensorBoard 日志 |
| `../../Results/<worktree名>/logs_phase3/` | Phase 3 | TensorBoard 日志 |
| `../../Results/<worktree名>/logs_phase4/` | Phase 4 | TensorBoard 日志 |
| `../../Results/runs/<时间>_<分支>_<commit>_testacc..._testauc.../` | Phase 4 收尾 | 每次完整训练的自包含归档副本 |
| `../../Results/runs/<...>/run_metadata.json` | Phase 4 收尾 | 记录命令、配置、Git 分支/commit、best epoch 与最终 train / val / test 指标 |

## 声门二分类 Gate

`图像识别/glottis_binary/` 提供独立的声门区 / 非声门区二分类训练模块，用于减少非声门区域囊泡、气泡等误入疾病分类。它只读扫描 `Laryngeal_Dataset_Processed`，按文件夹语义映射为 `glottis` / `non_glottis`，并使用患者别名合并后的 patient-level split。训练产物默认归档到 `../../Results/main/glottis_binary_benchmarks/<timestamp>/`。

```bash
python 图像识别/glottis_binary/build_manifest_split.py --force
python -u 图像识别/glottis_binary/train_benchmarks.py --build-split --force-split
python -u 图像识别/glottis_binary/launch_parallel_benchmarks.py --build-split --force-split --profile balanced
```

基准脚本可串行或并行训练 ResNet、ViT、Swin 和 SupCon+Swin，并为每个模型保存 checkpoint、metrics、ROC/PR、混淆矩阵、错误样本、推荐 gate threshold 和 provenance。

## BAGLS ROI Reflection Gate

`图像识别/roi_reflection/` 是独立于 8 分类主训练的 sidecar 工作流，用公开 BAGLS glottis mask 构建 ROI 有效性与反光干扰信号。BAGLS 原始数据只读放在 `/mnt/data/LarynxData/BAGLS`；从 main worktree 运行时，新产物默认写到 `../../Results/main/roi_reflection/`。已验证的视频默认 checkpoint 已迁入 `../../Results/main/roi_reflection/`，历史 `../../Results/bagls_roi_reflection/` 只作为兼容回退。

该模块可用于构建 BAGLS manifest、训练 ROI localizer、训练 reflection gate、导出项目图像 `roi_scores.csv`，也可用 `crop_project_rois.py` 生成 ROI-cropped 项目图片根。仓库只保存代码、配置和轻量文档，不提交 BAGLS 原始图像、mask、大规模 crop、训练日志或大权重。

历史 corrected full run 证据：localizer corrected official test Dice 为 0.6853，reflection gate corrected official test F1 为 0.8507、specificity 为 0.9976、AUROC 为 0.9883；这些指标说明它适合做视频前置 ROI/反光过滤和复核提示，不应被表述为精细分割模型。

## 视频帧级推理脚手架

`视频识别/video_inference.py` 将喉镜视频当作抽帧后的图像集合处理，不训练时序模型。当前 main 的视频流程已合并 BAGLS 分支实现，固定为一条标准管线：质量 gate -> ROI validity/reflection gate -> `glottis_binary` 声门 gate -> 8 分类疾病模型 -> top-fraction 投票聚合。

视频数据已移出代码 workspace，统一放在 `/mnt/data/LarynxData/videos/`。脚本默认读取 `/mnt/data/LarynxData/videos/classified_videos`；也可用 `LARYNX_VIDEO_ROOT=/path/to/videos` 或 `--video-root` 覆盖。

旧的 `sum(VOC 概率) > Non-VOC 概率`、`nonvoc_lt_*`、`all_frames` 等多规则比较不再作为诊断口径；`frame_predictions.csv` 仍保留 `non_voc_prob` 和 `voc_sum_prob`，但它们只用于审计 8 分类模型的帧级倾向。病人级结论始终来自标准 ROI + glottis gate 管线，`variant` / `diagnosis_variant` 字段仅为兼容旧 CSV schema 保留。

默认 checkpoint：

```text
8-class model = Results/main/roi_reflection/eight_class_roi_soft/best_model.pth, fallback Results/main/best_model.pth
glottis gate = Results/main/glottis_binary_benchmarks/20260505_183333_parallel/swin_base/best_model.pth
ROI localizer = Results/main/roi_reflection/localizer_transformer_20260507_111744/roi_localizer_best.pth
reflection gate = Results/main/roi_reflection/reflection_full_corrected/roi_reflection_best.pth
```

`--max-segment-gap-sec` 默认 `1.0` 秒，用于把闪光、瞬时模糊等短暂中断前后的声门片段合并为同一候选区间；如确实需要提高召回，可显式设置 `--glottis-gate-fallback-threshold 0.62`。可用 `--no-roi-gate` 跳过 ROI/reflection gate，用 `--no-glottis-gate` 跳过声门 gate。

默认会从目录结构推断 5 个视频标签：`normal/healthy-larynx -> Normal`、`cancer/laryngeal-cancer -> Cancer`、`benign/reinke-edema -> Reinke-Edema`、`benign/vocal-cord-polyp -> Vocal-Cord-Polyp`、`benign/vocal-cord-leukoplakia -> Vocal-Cord-Leukoplakia`。如医生后续标注了有效秒段，可提供 CSV manifest：

```csv
video_path,true_label,video_id,valid_start_sec,valid_end_sec
benign/reinke-edema/example.mp4,Reinke-Edema,case001,3.0,12.5
```

常用命令：

```bash
python 视频识别/video_inference.py \
  --video-root /mnt/data/LarynxData/videos/classified_videos \
  --output-dir ../../Results/main/video_inference_classified_standard \
  --sample-fps 8 \
  --batch-size 256 \
  --glottis-gate-threshold 0.94 \
  --top-fraction 0.20 \
  --render-video \
  --min-evidence-duration-sec 0.5
```

未标注待复核视频放在 `/mnt/data/LarynxData/videos/tbr/`，每个压缩包一个子目录，目录内只保留视频文件。由于这些视频没有真实类别标签，运行时应显式启用未标注模式，并把候选类别设为所有 VOC 类：

```bash
python 视频识别/video_inference.py \
  --video-root /mnt/data/LarynxData/videos/tbr \
  --allow-unlabeled \
  --candidate-labels all-voc \
  --output-dir ../../Results/main/video_inference_tbr_standard \
  --sample-fps 8 \
  --batch-size 256 \
  --glottis-gate-threshold 0.94 \
  --top-fraction 0.20
```

传入 `--render-video` 时会额外在项目根目录 `Output_Video/` 下输出 H.264/yuv420p 诊断叠加视频：8fps 抽样帧按 1/8 秒写回视频，右侧图像叠加 ROI softweight 明暗遮罩和预测病种 Grad-CAM，左侧黑边使用中文字体显示声门 0/1、当前/最长证据时间、最终标注和近 5 秒投票趋势折线图。

速度相关默认值：视频抽帧默认使用 `--decode-mode sequential`，顺序读视频并按帧间隔采样，避免旧的每个采样点反复 seek；保留 `--decode-mode seek` 作为回退。CUDA 推理默认开启 AMP，可用 `--no-amp` 关闭；默认 `--batch-size 256`，显存充足时可继续调大。

输出文件：

| 文件 / 目录 | 说明 |
|------|------|
| `frame_predictions.csv` | 每个抽样帧的质量指标、ROI validity/reflection、二分类 `prob_glottis`/gate 结果、8 分类 argmax、Non-VOC 审计概率、VOC 概率和逐类概率 |
| `video_predictions.csv` | 每个视频在标准管线下的唯一预测类别、投票分数、候选片段长度和逐类 top-fraction 投票统计 |
| `video_segments.csv` | gate 片段与预测类别高置信证据片段 |
| `summary.json` | 标准管线的视频级准确率、预测分布、ROI / glottis gate 配置和保留帧统计 |
| `diagnosis_summary.csv` | 病人级诊断总表：预测病种、病灶证据时间段、最高概率帧时间、低置信原因和 Grad-CAM 路径 |
| `diagnosis_evidence.csv` | 视频级诊断证据表；多视频病人会在病人级总表中择最高概率帧 |
| `diagnosis_gradcam/` | 每个病人的原始视频帧、模型输入帧和 Grad-CAM overlay 对照图；文件名和标题都显示病人名与中文预测病种，热力图高响应区域仅表示模型关注位置，不等同于医生标注的真实病灶边界 |
| `Output_Video/*.mp4` | `--render-video` 生成的 H.264/yuv420p 诊断叠加视频 |

普通 `keyframes/` JPG 默认不再保存；如需调试每个规则的最高分原图，可显式加 `--save-keyframes`。

## 文献对齐评估图表

本项目额外提供 `图像识别/tools/literature_aligned_metrics.py`，用于按近期 Q1 医学视觉模型论文常见口径重新导出论文式评估表和图。默认读取 `../../Results/<worktree名>/best_model.pth` 与冻结患者级切分，不重新训练。

```bash
python 图像识别/tools/literature_aligned_metrics.py
```

该脚本默认复用 `图像识别/config_phase2.json` 里的 `eval_batch_size`、`num_workers`、`prefetch_factor` 与 `persistent_workers`，并预缓存基础预处理后的图像；如机器内存紧张，可加 `--no-image-cache`。

主要输出集中在 `../../Results/<worktree名>/literature_aligned_metrics/`：

| 文件 | 说明 |
|------|------|
| `summary_metrics.csv` | train / val / test 层面的 Accuracy、Balanced Accuracy、Macro/Weighted Precision、Recall、F1、AUROC、AUPRC |
| `per_class_metrics.csv` | 逐类 Precision、Sensitivity、Specificity、F1、one-vs-rest AUROC、AUPRC |
| `voc_binary_metrics.csv` | 将任务折叠为 VOC vs Non-VOC 后的二分类指标 |
| `bootstrap_ci_test.csv` | test 主指标的分层 bootstrap 95% 置信区间 |
| `confusion_matrix_test_counts.png` / `confusion_matrix_test_normalized.png` | 测试集计数与行归一化混淆矩阵 |
| `roc_curves_test.png` / `precision_recall_curves_test.png` | 测试集 one-vs-rest ROC 与 PR 曲线 |
| `voc_binary_roc_pr_confusion_test.png` | VOC vs Non-VOC 的混淆矩阵、ROC 与 PR 曲线 |
| `training_curves_literature_style.png` | 论文式训练/验证 Loss、F1、Accuracy、AUROC 曲线 |
| `per_class_metric_bars_test.png` / `support_distribution_test.png` | 逐类指标柱状图与类别样本量图 |
| `tsne_test_features.png` | 测试集 backbone 特征 t-SNE 可视化 |
| `gradcam_maps_existing.png` | 当前 Grad-CAM 输出的归档副本 |

如需生成类别均衡的 Grad-CAM 展示图，可运行：

```bash
python 图像识别/tools/generate_class_balanced_gradcam.py --samples-per-class 2
```

输出 `../../Results/<worktree名>/literature_aligned_metrics/gradcam_class_balanced_test.png`，默认每类随机展示 2 张测试集样本；每个样本左侧为原图、右侧为 Grad-CAM，标题与边框绿色代表预测正确、红色代表预测错误。

对应文献与图表口径记录在 `图像识别/References/q1_visual_model_papers/literature_matrix.md`，原文 PDF 放在同一目录下。

## 模型对比实验

为对齐近期医学视觉论文中的 baseline comparison 口径，项目提供 `图像识别/tools/train_comparison_models.py` 训练经典 ImageNet 预训练基线，并生成内部模型与文献参考结果的对比图表。当前已跑通 `MobileNetV2`、`DenseNet121`、`ResNet50` 三个 baseline，输出集中在 `../../Results/<worktree名>/model_comparison/`。

```bash
python 图像识别/tools/train_comparison_models.py --models mobilenetv2 densenet121 resnet50 --epochs 25 --patience 5
```

只重建图表和汇总表，不重新训练：

```bash
python 图像识别/tools/train_comparison_models.py --report-only
```

主要输出：

| 文件 / 目录 | 说明 |
|------|------|
| `../../Results/<worktree名>/model_comparison/internal_baselines/<model>/` | 每个 baseline 的 checkpoint、history、summary/per-class metrics、image-level predictions |
| `../../Results/<worktree名>/model_comparison/internal_same_split_comparison.csv` | 当前 checkpoint 与 baseline 在当前冻结测试集上的主指标对比 |
| `../../Results/<worktree名>/model_comparison/internal_per_class_f1.csv` | 逐类 F1 长表 |
| `../../Results/<worktree名>/model_comparison/literature_sota_reference.csv` | 文献报告值，仅作外部参考，不与本项目直接排名 |
| `../../Results/<worktree名>/model_comparison/internal_same_split_metrics.png/.pdf` | 多指标点线小面板 |
| `../../Results/<worktree名>/model_comparison/internal_per_class_f1_heatmap.png/.pdf` | 逐类 F1 热力图 |
| `../../Results/<worktree名>/model_comparison/internal_voc_binary_metrics.png/.pdf` | VOC vs Non-VOC 层级判断对比 |
| `../../Results/<worktree名>/model_comparison/model_comparison_overview.png/.pdf` | 内部测试集与文献 reference 分组概览 |

注意：当前 `../../Results/<worktree名>/best_model.pth` 早于 v6.9 split 文件生成时间。`../../Results/<worktree名>/model_comparison/current_checkpoint_v69_eval/` 中的当前模型行可用于视觉审阅和阶段性对比；正式论文中的严格 same-split 结论，建议在 v6.9 split 上重新运行 Phase 1/2 后再复跑 `--report-only`。

## 最小文件清单

当前项目只需要这些文件即可启动默认图像训练 pipeline：

| 文件 | 作用 |
|------|------|
| `图像识别/train_phase1.py` | Phase 1 SupCon 预训练入口 |
| `图像识别/train_phase2.py` | Phase 2 CE 微调入口 |
| `图像识别/train_phase3.py` | Phase 3 train-confusion-focused SupCon 入口 |
| `图像识别/train_phase4.py` | Phase 4 classifier retraining 入口 |
| `图像识别/train_pipeline.py` | Phase 1 -> Phase 4 串行控制入口 |
| `图像识别/shared.py` | 图像训练共享模型、数据、训练、指标与可视化逻辑 |
| `图像识别/config_phase1.json` | Phase 1 超参数与知识图谱配置 |
| `图像识别/config_phase2.json` | Phase 2 超参数配置 |
| `图像识别/config_phase3.json` | Phase 3 混淆聚焦 SupCon 配置 |
| `图像识别/config_phase4.json` | Phase 4 classifier retraining 配置 |
| `图像识别/dataset_split.json` | 当前 8 类患者级冻结切分与类别映射 |
| `README.md` / `CHANGELOG.md` | 当前运行说明与变更记录 |

## 设计约束

- **医学精细识别任务**：不使用 RandomErasing 等可能遮挡病灶的数据增强
- **移除 Mixup/CutMix**：避免局部病灶被挖掉或全局特征变得混浊
- **SupCon 单视图设计**：正样本关系完全由标签定义，不使用 SimCLR 风格的 two-view 增强
- **层级推理**：推理时通过概率比较实现 VOC vs Non-VOC 的动态决策

**项目约定：代码/方法改动在 `laryngeal_multiclass/worktrees/<分支名>/` 中提交；训练、评估、图表和 checkpoint 默认输出到 `laryngeal_multiclass/Results/<worktree名>/`，不要写回 worktree 代码目录。**

# Laryngeal Multi-class Classification

基于 Swin-B (224) 的喉镜图像层级多分类项目。训练流程先使用医学知识图谱引导的 Supervised Contrastive Learning 预训练，让不同疾病之间的相关性矩阵参与表征学习，再使用 Cross-Entropy Learning 做最终分类微调。

本项目从原四分类版本复制而来，并将原先聚合的 `Lesion`（良性病变）拆分为当前数据集中的具体疾病类型。

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

训练时以 `dataset_split.json` 中冻结的 `class_folders` 为基础；当前冻结切分按静态单图任务启用 8 类：

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

`dataset_split.json` 已在生成阶段排除 `Vocal-Cord-Fixation`，因为声带固定更依赖发声/呼吸过程中的运动受限证据，单张静态喉镜图像容易把它混入 Non-VOC 或普通 VOC 外观。当前 `声带固定`、`声带黏连`、`功能性声带运动不全`、`声带小结` 已被本次冻结切分排除，不参与训练。

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
- 切分结果以 JSON 形式**冻结**在 `dataset_split.json` 中，两阶段训练脚本只读不重算 —— 杜绝因运行时随机环境变化导致 Phase 1/2 划分漂移
- 自动黑边裁切（`CropBlackBorders`）在训练与推理中均生效

当前目录是极简训练包，只保留现成的 `dataset_split.json`。如果以后要改变类别集合或重新切分数据，需要重新引入/编写切分脚本并重新训练两阶段 checkpoint。

## 训练流程

采用**两阶段训练**：

1. **Phase 1 — Knowledge-Guided Hierarchical SupCon 预训练**：将同类样本拉近、异类样本推远，同时用 `knowledge_graph.class_similarity` 中的疾病相关性矩阵为医学上相近的类别提供软正样本权重。只训练 backbone + projection head。
2. **Phase 2 — CE 微调**：用 Phase 1 学到的 backbone 作为起点，训练分类头做标准交叉熵分类。

当前训练入口固定为两步制：先运行 Phase 1，再运行 Phase 2。

### 正则化策略

| 手段 | 说明 | 当前值 |
|------|------|--------|
| Feature Dropout | backbone 输出到 classifier 前的 dropout | dropout_rate × 0.5 |
| Classifier Dropout | 分类头隐藏层后的 dropout | dropout_rate |
| Drop Path | Swin block 中的随机深度 | drop_path_rate |
| Label Smoothing | 标签平滑 | label_smoothing |
| Weight Decay | AdamW L2 正则 | weight_decay |
| Layer-wise LR Decay | 浅层使用更小的学习率 | layer_decay |
| 参数冻结 | 仅解冻末尾 N 个 block | unfreeze_last_n_blocks |
| Early Stopping | 监控 val composite score = 0.7 × macro-F1 + 0.3 × AUROC，并使用 min_delta 抑制平台期噪声 | early_stopping_patience / early_stopping_min_delta |

### 本轮过拟合诊断口径

当前切分中 `Non-Vocal-Cord` 接近测试集一半，Accuracy 容易被大类与 VOC/Non-VOC 层级判断抬高；因此主要用 macro-F1、逐类 recall/F1 与 AUROC train-vs-val/test gap 判断是否泛化。旧 9 类运行里 `Vocal-Cord-Fixation` 在训练集、验证集和测试集均长期低 recall，且主要与 Non-VOC 互相混淆；这更像静态单图任务定义问题，而不是单纯训练轮数不够。Phase 1 checkpoint 继续优先按 `val_loss` 选择，避免把过拟合表征传入 Phase 2。

### 类别平衡采样

两阶段均通过 `WeightedRandomSampler` 实现采样平衡：

| 阶段 | 策略 | 有效采样比例 |
|------|------|-------------|
| Phase 1 SupCon | 层级平衡：VOC=Non-VOC 各 50%，VOC 内所有细分类等权 | Non-VOC 50%，VOC 类共享 50% |
| Phase 2 CE | 温和逆频率采样，`sampler_balance_alpha=0.65`；不再叠加 `class_weights` 加权 loss | 少数类被提升，但不再像类别完全等权那样重复抽样过多 |

## 配置

超参数分别集中在 `config_phase1.json` 和 `config_phase2.json` 中，无需修改 Python 代码即可调参。类别映射会在运行时与 `dataset_split.json` 同步，并应用 `excluded_classes_from_split`。

### CE 微调参数

| 参数 | 说明 | 默认值 |
|------|------|----------|
| `epochs` | CE 最大训练轮数 | 80 |
| `batch_size` | 训练 batch | 256 |
| `eval_batch_size` | 评估 batch | 512 |
| `grad_accum` | 梯度累积步数 | 1 |
| `learning_rate` | 峰值学习率 | 5.5e-5 |
| `unfreeze_last_n_blocks` | Swin 解冻的末尾 block 数 | 1 |
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

两阶段训练现在采用极速的显存直读流水线：根据 `dataset_split.json` 将所有基础预处理（黑边裁切、Resize、CenterCrop）后的图像，以未归一化的 `uint8 CHW` 张量形式**按 train / val / test 顺序写入一块连续 GPU 显存 (VRAM) 缓存**。训练时 GPU 直接从显存调用张量批次，均衡采样权重常驻 GPU，废除 `num_workers` 多进程加载开销，并在 GPU 内直接执行随机增强和 ImageNet normalize。评估、测试、GradCAM 同样享受显存直读速度；顺序 split 会直接用连续显存切片，不注入随机增强。

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

推荐使用拆分后的两阶段脚本，分别配置和运行：

```bash
# Phase 1: SupCon 预训练（配置：config_phase1.json）
python train_phase1.py

# Phase 2: CE 微调（配置：config_phase2.json，必须能加载 Phase 1 checkpoint）
python train_phase2.py
```

也支持自定义 config 路径：

```bash
python train_phase1.py --config my_phase1_config.json
python train_phase2.py --config my_phase2_config.json
```

自定义配置中的类别映射不会直接覆盖当前冻结切分；如需新增类别、合并类别或重做患者级划分，请先用目标配置重新生成 `dataset_split.json`，再重新运行两阶段训练。

训练过程中同时监控 train / val / test 三套指标，**模型选择（early stopping + checkpoint）仅依据 val composite score = 0.7 × macro-F1 + 0.3 × AUROC**，测试集指标不参与任何训练决策。test 指标继续写入 `../../Results/<worktree名>/history.csv` 和 TensorBoard 便于离线诊断，但默认 `Training Curve` 只展示 train / val 曲线，避免把测试集表现放进训练过程图。图中的 train 曲线来自训练时的随机增强与 balanced-sampler batch，适合观察优化过程；判断 train/val 泛化差距时优先看 `../../Results/<worktree名>/metrics.csv` 中无随机增强的 train_eval / val / test 最终评估。

> 注意：旧四分类 checkpoint 以及本次排除 `Vocal-Cord-Fixation` 前的 9 类 checkpoint 都与当前 8 类分类头不兼容。调整类别口径后请重新运行 Phase 1，再运行 Phase 2。

## 输出文件

当前 worktree 不保留训练产物；运行脚本后会在 `../../Results/<worktree名>/` 重新生成以下文件。可用 `LARYNX_RESULTS_DIR=/absolute/path` 覆盖本次输出目录。

| 文件 | 来源 | 说明 |
|------|------|------|
| `../../Results/<worktree名>/phase1_checkpoint.pth` | Phase 1 | SupCon 预训练后的完整模型权重 |
| `../../Results/<worktree名>/phase1_history.json` | Phase 1 | SupCon 每 epoch 的 loss/lr 记录 |
| `../../Results/<worktree名>/logs_phase1/` | Phase 1 | TensorBoard 日志 |
| `../../Results/<worktree名>/best_model.pth` | Phase 2 | 最优模型权重（按 val composite score = 0.7 × macro-F1 + 0.3 × AUROC 选取） |
| `../../Results/<worktree名>/history.csv` | Phase 2 | 每 epoch 的 train/val 指标与 test 诊断指标 |
| `../../Results/<worktree名>/metrics.csv` | Phase 2 | 最终三集上的分类报告（含层级 VOC 准确率） |
| `../../Results/<worktree名>/training_curves.png` | Phase 2 | F1 / Acc / AUC 训练曲线（含 Phase 1 曲线；默认仅显示 train / val） |
| `../../Results/<worktree名>/gradcam_maps.png` | Phase 2 | GradCAM 注意力可视化 |
| `../../Results/<worktree名>/logs_phase2/` | Phase 2 | TensorBoard 日志 |

## 文献对齐评估图表

本项目额外提供 `tools/literature_aligned_metrics.py`，用于按近期 Q1 医学视觉模型论文常见口径重新导出论文式评估表和图。默认读取 `../../Results/<worktree名>/best_model.pth` 与冻结患者级切分，不重新训练。

```bash
python tools/literature_aligned_metrics.py
```

该脚本默认复用 `config_phase2.json` 里的 `eval_batch_size`、`num_workers`、`prefetch_factor` 与 `persistent_workers`，并预缓存基础预处理后的图像；如机器内存紧张，可加 `--no-image-cache`。

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
python tools/generate_class_balanced_gradcam.py --samples-per-class 2
```

输出 `../../Results/<worktree名>/literature_aligned_metrics/gradcam_class_balanced_test.png`，默认每类随机展示 2 张测试集样本；每个样本左侧为原图、右侧为 Grad-CAM，标题与边框绿色代表预测正确、红色代表预测错误。

对应文献与图表口径记录在 `References/q1_visual_model_papers/literature_matrix.md`，原文 PDF 放在同一目录下。

## 模型对比实验

为对齐近期医学视觉论文中的 baseline comparison 口径，项目提供 `tools/train_comparison_models.py` 训练经典 ImageNet 预训练基线，并生成内部模型与文献参考结果的对比图表。当前已跑通 `MobileNetV2`、`DenseNet121`、`ResNet50` 三个 baseline，输出集中在 `../../Results/<worktree名>/model_comparison/`。

```bash
python tools/train_comparison_models.py --models mobilenetv2 densenet121 resnet50 --epochs 25 --patience 5
```

只重建图表和汇总表，不重新训练：

```bash
python tools/train_comparison_models.py --report-only
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

当前项目只需要这些文件即可启动两阶段训练：

| 文件 | 作用 |
|------|------|
| `train_phase1.py` | Phase 1 SupCon 预训练入口 |
| `train_phase2.py` | Phase 2 CE 微调入口 |
| `shared.py` | 两阶段共享模型、数据、训练、指标与可视化逻辑 |
| `config_phase1.json` | Phase 1 超参数与知识图谱配置 |
| `config_phase2.json` | Phase 2 超参数配置 |
| `dataset_split.json` | 当前 8 类患者级冻结切分与类别映射 |
| `README.md` / `CHANGELOG.md` | 当前运行说明与变更记录 |

## 设计约束

- **医学精细识别任务**：不使用 RandomErasing 等可能遮挡病灶的数据增强
- **移除 Mixup/CutMix**：避免局部病灶被挖掉或全局特征变得混浊
- **SupCon 单视图设计**：正样本关系完全由标签定义，不使用 SimCLR 风格的 two-view 增强
- **层级推理**：推理时通过概率比较实现 VOC vs Non-VOC 的动态决策

# 图像识别

这个目录保存静态喉镜图像的 8 分类训练、评估和论文图表脚本。

## 主要入口

| 文件 | 用途 |
|------|------|
| `train_pipeline.py` | 串行运行 Phase 1 -> Phase 4 |
| `train_phase1.py` | Knowledge-Guided SupCon 预训练 |
| `train_phase2.py` | CE 微调 |
| `train_phase3.py` | 基于 val split 混淆选对、再用 train 易混类样本训练的 focused SupCon |
| `train_phase4.py` | Phase 3 后的 classifier retraining |
| `shared.py` | 模型、数据、指标、路径和 Grad-CAM 共用逻辑 |
| `config_phase*.json` | 各阶段超参数 |
| `dataset_split.json` | 当前 8 类患者级冻结切分 |
| `glottis_binary/` | 声门区 / 非声门区二分类 gate 的 split、训练、评估与基准归档脚本 |

当前多分类 split 由 `Laryngeal_Dataset_Processed/build_multiclass_split.py` 生成。该脚本固定为 8 分类协议：混杂图片、正常、声带任克水肿、声带囊肿、声带息肉、声带白斑、声带肉芽肿、喉癌；新增或未配置的大文件夹不会自动进入 split。

## Selective Unfreeze

当前默认配置已合入 selective-unfreeze 口径：

- Phase 1: `config_phase1.json` 显式解冻 `stage3.block[-1]`，并训练 Swin final norm。默认 `knowledge_graph.learnable=true` 且 `knowledge_graph.activate_epoch=30`：epochs 1-29 使用固定零 KG similarity，epoch 30 开始激活 learnable KG 并在 train split 上做 bilevel KG 更新；val 不参与 KG 更新，early stopping 在 KG 激活前不生效。
- Phase 2: `config_phase2.json` 显式解冻 `stage3.block[-1]`，并训练 Swin final norm。
- Phase 3: `config_phase3.json` 显式解冻 `stage2.block[-1]`、`stage3.block[-1]` 与 Swin final norm，`phase3_confusion_threshold=0.08`，默认 `phase3_loss_mode=pair_margin`，也可切到 `direction`；Phase 2 原始 val score 会作为 epoch 0 baseline，只有超过 baseline 才会替换 checkpoint，`phase3_reinit_projector=false`。
- Phase 4: `config_phase4.json` 显式解冻 `stage3.block[-1]` 与 Swin final norm，重置并训练 classifier，projector 冻结；若 Phase 4 没有超过输入 checkpoint 的 val baseline，则最终保留输入 checkpoint。
- `shared.py` 支持 `unfreeze_blocks`；设置后覆盖连续末尾 `unfreeze_last_n_blocks`，并可用 `train_backbone_norm` 控制是否同步训练 Swin final norm。

## 辅助脚本

| 文件 | 用途 |
|------|------|
| `tools/literature_aligned_metrics.py` | 生成论文式评估表和图 |
| `tools/generate_class_balanced_gradcam.py` | 生成类别均衡 Grad-CAM 展示 |
| `tools/train_comparison_models.py` | 训练/汇总经典图像 baseline |
| `tools/collect_mispredictions.py` | 收集错分图像供复核 |
| `tools/train_hard_specialist.py` | 从现有 8 类 checkpoint 初始化 backbone，训练易混类别子模型并输出三集预测与混淆矩阵 |

## 声门二分类 Gate

`glottis_binary/` 只读扫描 `Laryngeal_Dataset_Processed`，按文件夹语义映射为 `glottis` / `non_glottis`，使用患者别名合并后的 patient-level split，并将 ResNet、ViT、Swin、SupCon+Swin 结果归档到 `../../Results/main/glottis_binary_benchmarks/<timestamp>/`。

```bash
python 图像识别/glottis_binary/build_manifest_split.py --force
python -u 图像识别/glottis_binary/train_benchmarks.py --build-split --force-split
python -u 图像识别/glottis_binary/launch_parallel_benchmarks.py --build-split --force-split --profile balanced
```

推荐从项目根目录运行，例如：

```bash
python 图像识别/train_pipeline.py
python 图像识别/tools/literature_aligned_metrics.py
```

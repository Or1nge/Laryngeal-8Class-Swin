# 图像识别

这个目录保存静态喉镜图像的 8 分类训练、评估和论文图表脚本。

## 主要入口

| 文件 | 用途 |
|------|------|
| `train_pipeline.py` | 串行运行 Phase 1 -> Phase 4 |
| `train_phase1.py` | Knowledge-Guided SupCon 预训练 |
| `train_phase2.py` | CE 微调 |
| `train_phase3.py` | 基于 train split 混淆的 focused SupCon |
| `train_phase4.py` | Phase 3 后的 classifier retraining |
| `shared.py` | 模型、数据、指标、路径和 Grad-CAM 共用逻辑 |
| `config_phase*.json` | 各阶段超参数 |
| `dataset_split.json` | 当前 8 类患者级冻结切分 |
| `glottis_binary/` | 声门区 / 非声门区二分类 gate 的 split、训练、评估与基准归档脚本 |

## Selective Unfreeze

当前默认配置已合入 selective-unfreeze 口径：

- Phase 1: `config_phase1.json` 显式解冻 `stage2.block[-1]` 与 `stage3.block[-1]`。
- Phase 3: `config_phase3.json` 显式解冻 `stage3.block[-1]`，`phase3_confusion_threshold=0.10`，`phase3_reinit_projector=false`。
- `shared.py` 支持 `unfreeze_blocks`；设置后覆盖连续末尾 `unfreeze_last_n_blocks`。

## 辅助脚本

| 文件 | 用途 |
|------|------|
| `tools/literature_aligned_metrics.py` | 生成论文式评估表和图 |
| `tools/generate_class_balanced_gradcam.py` | 生成类别均衡 Grad-CAM 展示 |
| `tools/train_comparison_models.py` | 训练/汇总经典图像 baseline |
| `tools/collect_mispredictions.py` | 收集错分图像供复核 |

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

# BAGLS ROI Reflection Gate

这个模块把公开 BAGLS 的 glottis mask 转成可复现的 ROI/reflection sidecar：

- `download_bagls.py` 下载 `training.zip` 和 `test.zip` 到 `/mnt/data/LarynxData/BAGLS`。
- `build_bagls_manifest.py` 生成 `bagls_roi_manifest.csv`、`review_queue.csv` 和 manifest summary。
- `train_localizer.py` 训练 glottis localizer；默认使用 ImageNet/timm encoder U-Net，旧 Tiny-UNet 可通过 `model_arch: "tiny_unet"` 回退。
- `train_reflection_gate.py` 训练 MobileNetV3-Small 双头 ROI valid / reflection gate。
- `evaluate_checkpoint.py` 复评 checkpoint。
- `export_project_scores.py` 对本地 8 类图像生成 `roi_scores.csv`，供 Phase 2/4 sidecar 使用。
- `crop_project_rois.py` 把 ROI localizer 输出实际裁剪成新的项目图片根；类别文件夹和相对文件名保持不变，供主 8 分类训练用 `LARYNX_IMAGE_DIR=/path/to/crops` 读取。

默认输出在：

```bash
/home/or1ngelinux/CVProjects/Larynx/laryngeal_multiclass/Results/main/roi_reflection/
```

## Quick Smoke

```bash
python 图像识别/roi_reflection/build_bagls_manifest.py \
  --bagls-root /mnt/data/LarynxData/BAGLS \
  --output-dir ../../Results/main/roi_reflection/manifest \
  --limit 200 --force

python 图像识别/roi_reflection/train_localizer.py \
  --manifest ../../Results/main/roi_reflection/manifest/bagls_roi_manifest.csv \
  --epochs 1 --cache-device cuda

python 图像识别/roi_reflection/train_reflection_gate.py \
  --manifest ../../Results/main/roi_reflection/manifest/bagls_roi_manifest.csv \
  --epochs 1 --cache-device cuda

python 图像识别/roi_reflection/crop_project_rois.py \
  --image-root /home/or1ngelinux/CVProjects/Larynx/Laryngeal_Dataset_Processed \
  --output-root ../../Results/main/roi_reflection/project_roi_crop_smoke_32 \
  --checkpoint swin256=../../Results/main/roi_reflection/localizer_transformer_20260507_111744/roi_localizer_best.pth \
  --combo-name swin256 \
  --threshold 0.55 \
  --limit 32 --batch-size 8 --device auto --postprocess-device auto
```

## Full Workflow

```bash
python 图像识别/roi_reflection/download_bagls.py --bagls-root /mnt/data/LarynxData/BAGLS
python 图像识别/roi_reflection/build_bagls_manifest.py --bagls-root /mnt/data/LarynxData/BAGLS --force
python 图像识别/roi_reflection/train_localizer.py --manifest ../../Results/main/roi_reflection/manifest/bagls_roi_manifest.csv --cache-device cuda
python 图像识别/roi_reflection/train_reflection_gate.py --manifest ../../Results/main/roi_reflection/manifest/bagls_roi_manifest.csv --cache-device cuda
python 图像识别/roi_reflection/export_project_scores.py \
  --image-root /home/or1ngelinux/CVProjects/Larynx/Laryngeal_Dataset_Processed \
  --localizer-checkpoint ../../Results/main/roi_reflection/<localizer_run>/roi_localizer_best.pth \
  --reflection-checkpoint ../../Results/main/roi_reflection/<reflection_run>/roi_reflection_best.pth \
  --scores-csv ../../Results/main/roi_reflection/roi_scores.csv \
  --cache-device cuda

python 图像识别/roi_reflection/crop_project_rois.py \
  --image-root /home/or1ngelinux/CVProjects/Larynx/Laryngeal_Dataset_Processed \
  --output-root ../../Results/main/roi_reflection/project_roi_crops_best \
  --checkpoint swin256=../../Results/main/roi_reflection/localizer_transformer_20260507_111744/roi_localizer_best.pth \
  --combo-name swin256 \
  --threshold 0.55 \
  --batch-size 64 --device auto --postprocess-device auto
```

`crop_project_rois.py` 是 8 分类训练前的数据预处理，不启动训练、不修改源数据，也不把 ROI 当作 sidecar weight 注入训练循环。裁切默认走分类安全模式：先用黑边检测得到有效图像区域，再以 ROI bbox 中心为参考，在有效区域内扩到最小上下文尺寸；默认 `--min-crop-width-ratio 0.60`、`--min-crop-height-ratio 0.60`、`--min-crop-area-ratio 0.50`、`--max-crop-area-ratio 1.00`。无 ROI、ROI 面积低于 `--min-roi-area-ratio`、或安全裁切仍低于阈值时，默认保存黑边裁切后的图像，避免把 1%-5% 的小条/小块送入主分类训练。

CUDA 可用时，`crop_project_rois.py` 默认用 `--postprocess-device auto` 在 GPU 上完成 ROI 概率图还原、ensemble 平均、阈值化、ROI 面积和 bbox 提取，只把最终 bbox 数字拿回 CPU 做裁剪与落盘。若要复现旧的 CPU 后处理路径，可显式加 `--postprocess-device cpu`。

`manifest.csv` 记录 `input_path`、`output_path`、`bbox`、`valid_prob`、`crop_status`、`safe_crop_status`、`class_folder`、原图尺寸、输出 crop 尺寸、`crop_area_ratio`、raw ROI bbox、expanded ROI bbox、safe ROI bbox 和 black-border 有效区域，便于审计 raw ROI、expanded safe ROI 与 fallback 的差异。

## Current Corrected Run

2026-05-07 的 corrected run 使用 `manifest_corrected/bagls_roi_manifest.csv`，修复了 official test 图像误配 training mask 的问题。归档指标：

- Manifest：59,250 张；train 50,175，val 5,575，official test 3,500。
- Localizer：train Dice 0.8704，val Dice 0.8692，corrected official test Dice 0.6853。
- Reflection gate：corrected official test Accuracy 0.9827，Precision 0.9557，Recall 0.7665，Specificity 0.9976，F1 0.8507，AUROC 0.9883，AUPRC 0.9459。
- Project sidecar：`roi_scores.csv` 13,402 行；`clean` 3,511，`reflection` 1,371，`severe_reflection` 7,590，`low_valid` 930。

## Localizer Upgrade Run

2026-05-07 `localizer_upgrade_20260507_102436` 使用同一个 corrected manifest 训练 BAGLS glottis ROI localizer。训练选择只看 validation split，不使用 official test 做 checkpoint 决策。

- 模型：`resnet34.a1_in1k` ImageNet/timm encoder U-Net；保留 `tiny_unet` 和 `residual_unet` 配置选项。
- Train-only 同步增强：水平翻转、保守仿射、gamma/亮度/对比度、轻微 blur/noise；val/test 不增强。
- 训练：AMP、AdamW、cosine scheduler、grad clipping、`min_delta=0.001`；best checkpoint 按 val Dice 选择。
- Best epoch：28；train Dice 0.8569，val Dice 0.8720，corrected official test Dice 0.7394，超过旧 Tiny-UNet corrected test Dice 0.6853。
- 产物：`Results/bagls_roi_reflection/roi_reflection/localizer_upgrade_20260507_102436/`；复评目录：`eval_corrected/`。
- 性能观察：本轮 train log 从 10:24:36 到 10:48:56，30 epochs 约 24.3 分钟；训练中采样到约 17GB 显存和约 73% GPU util，复评阶段约 2.7GB 显存且 GPU util 在 0-19% 波动，说明 per-sample Python/PIL 加载与同步增强仍是下一轮优先瓶颈。

## Transformer / Ensemble Localizer Run

2026-05-07 的 transformer localizer 使用 corrected manifest，不用 official test 做训练选择；针对 official test 与 train/val 的 aspect ratio、empty mask、mask area tail 差异，加入 letterbox/pad、batched GPU augmentation、cached batch iterator、area-aware sampling、Tversky/focal BCE、empty false-positive 监测、per-sample eval export 和 worst test overlay。

- 单模型候选：
  - `localizer_transformer_20260507_111744`：Swin-Tiny 256 letterbox，corrected train/val/test Dice 为 0.8426/0.8417/0.7840；test 比 resnet34 0.7394 提高 4.46 点，val-test gap 缩到 5.77 点。
  - `localizer_transformer_pvtv2b1_20260507_112932`：PVTv2-B1 256 letterbox，corrected train/val/test Dice 为 0.8386/0.8370/0.7303；val 高但 official test 退化，不作为最终路线。
  - `localizer_transformer_swin320_20260507_112932`：Swin-Tiny 320 letterbox partial run，corrected train/val/test Dice 为 0.8131/0.8152/0.7341；因慢且低于领先候选，在 epoch 10 主动停止并复评已有 best checkpoint。
- 性能证据：前几个 epoch 的 `nvidia-smi dmon` active SM 均值约 97%，Swin256 单进程 epoch 2 约 860 images/sec；三并发后总吞吐约 840-880 images/sec，停止慢速 Swin320 后 Swin256/PVT 分别约 450/570 images/sec。
- Ensemble/阈值：`localizer_ensemble_20260507` 在 val 上选择阈值后复评 corrected test；Swin256+PVT 概率平均取得 val/test Dice 0.8898/0.8451，Swin256+Swin320 为 0.8706/0.8403，三模型平均为 0.8854/0.8437。当前 best 为 Swin256+PVT ensemble。该目录早期 `metrics.csv` 的 train rows 由修复前评估脚本写出，因 `augment=False` 在非缓存 train 路径仍触发随机增强而偏低；val/test rows 不走 train 增强，仍可作为本轮结论。修复后 25,440 条 train sanity rows 的 mean Dice 为 0.8927，确认 train 低分是评估 bug。
- 误差定位：Swin256 corrected test 的主要剩余损失来自 empty mask 和 tiny mask；empty test mean Dice 0.5259，tiny mask mean Dice 0.6293；非空样本 mean Dice 0.8211，中大面积 mask 已达 0.91+。

不要把 BAGLS 原图、mask、衍生 crop 或基于 BAGLS 的大权重提交进源码仓库。许可说明见 `NOTICE`。

# 声门区 / 非声门区二分类 Gate

这个目录是独立于 8 分类训练文件的二分类模块，用于训练视频识别前置 gate：判断帧是否为有效声带 / 声门区。原始数据只读扫描，checkpoint 与训练证据默认输出到：

```bash
/home/or1ngelinux/CVProjects/Larynx/laryngeal_multiclass/Results/main/glottis_binary_benchmarks/<timestamp>/
```

## 数据口径

- `混杂图片` / `Non-Vocal-Cord`：`non_glottis`
- `室带膨隆`：`non_glottis`，因为它不是有效声带黏膜，后续如错误集中可再请医生确认
- `不用管—质量图片`：`non_glottis`，数量极少，作为不可用帧证据保留
- `正常`、`喉癌`、所有包含 `声带` / `Vocal-Cord` 的疾病文件夹：`glottis`

患者划分复用当前多分类数据规则：数字文件取前 8 位，`姓名_13...` 文件取下划线前姓名，同时合并文件名中暴露的数字别名，防止同一患者跨 split。

## 运行

```bash
# 只读生成 manifest + patient-level split
python 图像识别/glottis_binary/build_manifest_split.py --force

# 训练 ResNet / ViT / Swin / SupCon+Swin 基准
python -u 图像识别/glottis_binary/train_benchmarks.py --build-split --force-split

# 单卡四模型并行训练；balanced 会给四个进程分配不同 batch，降低 OOM 风险
python -u 图像识别/glottis_binary/launch_parallel_benchmarks.py --build-split --force-split --profile balanced

# 复评单个 checkpoint
python 图像识别/glottis_binary/evaluate_checkpoint.py \
  --checkpoint /home/or1ngelinux/CVProjects/Larynx/laryngeal_multiclass/Results/main/glottis_binary_benchmarks/<run>/<model>/best_model.pth
```

每个模型目录会保存 `config_effective.json`、`history.csv`、`best_model.pth`、`metrics.csv`、`confusion_matrix_test.*`、`roc_pr_test.png`、`predictions_*.csv`、`error_samples_test.csv`、`recommended_threshold.json`、`provenance.json` 和运行命令。

## ROI 裁切鲁棒性实验

`build_roi_cropped_split.py` 用于训练一个更能接受局部声门/声带画面的 gate。它复用现有患者级 split，只把正类限制在当前 8 分类任务的 7 个 VOC 类中；每个 split 里抽取一半正类指向 ROI localizer 生成的裁切副本，另一半保留原图，再抽取同数量非声门图片，形成平衡二分类数据。原始图片只读，裁切图和新 split 默认写到 `Results/<worktree>/glottis_binary_roi_crops/`。

```bash
python 图像识别/glottis_binary/build_roi_cropped_split.py

python 图像识别/roi_reflection/crop_project_rois.py \
  --image-root /home/or1ngelinux/CVProjects/Larynx/Laryngeal_Dataset_Processed \
  --output-root ../../Results/roi_cropped_glottis_gate/glottis_binary_roi_crops/images \
  --manifest-csv ../../Results/roi_cropped_glottis_gate/glottis_binary_roi_crops/roi_crop_manifest.csv \
  --include-list ../../Results/roi_cropped_glottis_gate/glottis_binary_roi_crops/roi_crop_input_list.csv \
  --checkpoint swin256=../../Results/main/roi_reflection/localizer_transformer_20260507_111744/roi_localizer_best.pth \
  --combo-name swin256 \
  --threshold 0.55 \
  --min-crop-width-ratio 0.40 \
  --min-crop-height-ratio 0.40 \
  --min-crop-area-ratio 0.25 \
  --max-crop-area-ratio 0.55 \
  --batch-size 64 --device auto --postprocess-device auto

python -u 图像识别/glottis_binary/train_benchmarks.py \
  --split ../../Results/roi_cropped_glottis_gate/glottis_binary_roi_crops/roi_cropped_glottis_split.json \
  --manifest ../../Results/roi_cropped_glottis_gate/glottis_binary_roi_crops/roi_cropped_glottis_manifest.csv \
  --output-root ../../Results/roi_cropped_glottis_gate/glottis_binary_benchmarks \
  --run-name roi_cropped_supcon_swin \
  --models supcon_swin_base \
  --batch-size 128 --eval-batch-size 384 \
  --cache-device cuda
```

本分支当前实测 checkpoint 为：

```text
../../Results/roi_cropped_glottis_gate/glottis_binary_benchmarks/roi_cropped_supcon_swin/supcon_swin_base/best_model.pth
```

它在 ROI-cropped 平衡 test split 上推荐阈值 0.96，Accuracy 0.9784、Specificity 0.9948、Glottis recall 0.9619、AUROC 0.9989。插回 main 视频管线且沿用原阈值 0.94 后，全量 `classified_videos` 为 23/43，旧 gate 同口径为 22/43；但 mean gate frames 从 43.6 降到 33.7，说明该模型更稳但不一定更宽松，暂不建议直接替换默认 gate。

## 当前推荐

当前视频 gate 默认使用：

```text
/home/or1ngelinux/CVProjects/Larynx/laryngeal_multiclass/Results/main/glottis_binary_benchmarks/20260505_183333_parallel/swin_base/best_model.pth
```

主阈值 `prob_glottis >= 0.94`，用于优先减少非声门帧进入 8 分类疾病模型。`视频识别/video_inference.py` 默认不自动降阈值；当某段视频保留帧过少且接受更高误放风险时，可以显式传入 `--glottis-gate-fallback-threshold 0.62`。

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

## 当前推荐

当前视频 gate 默认使用：

```text
/home/or1ngelinux/CVProjects/Larynx/laryngeal_multiclass/Results/main/glottis_binary_benchmarks/20260505_183333_parallel/swin_base/best_model.pth
```

主阈值 `prob_glottis >= 0.94`，用于优先减少非声门帧进入 8 分类疾病模型。`视频识别/video_inference.py` 默认不自动降阈值；当某段视频保留帧过少且接受更高误放风险时，可以显式传入 `--glottis-gate-fallback-threshold 0.62`。

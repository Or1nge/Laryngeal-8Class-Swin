# 视频识别

这个目录保存喉镜视频的弱监督推理脚手架。它不重新训练视频模型，而是抽帧后先使用 `图像识别/glottis_binary/` 的二分类 gate 筛出声门/有效声带帧，再复用 `图像识别/` 中的 8 分类 checkpoint 做疾病识别。

## 主要入口

| 文件 | 用途 |
|------|------|
| `video_inference.py` | 抽帧、质量 gate、声门二分类 gate、VOC/Non-VOC 弱筛选、病种概率聚合、证据秒段和 Grad-CAM 输出 |

默认声门 gate:

```text
checkpoint = Results/main/glottis_binary_benchmarks/20260505_183333_parallel/swin_base/best_model.pth
threshold = 0.94
fallback_threshold = disabled by default
```

`--max-segment-gap-sec` 默认 `1.0` 秒，会把闪光、瞬时模糊等短暂中断前后的声门片段合并。若临床视频有效帧太少，可显式加 `--glottis-gate-fallback-threshold 0.62` 提高召回，但这会增加非声门帧漏入疾病模型的风险。

推荐从项目根目录运行，例如：

```bash
python 视频识别/video_inference.py \
  --video-root /mnt/data/LarynxData/videos/tbr \
  --allow-unlabeled \
  --candidate-labels all-voc \
  --model ../../Results/selective_unfreeze_confusion10/best_model.pth \
  --output-dir ../../Results/main/video_inference_tbr \
  --sample-fps 8 \
  --batch-size 256 \
  --glottis-gate-threshold 0.94 \
  --top-fraction 0.20
```

默认每秒抽 8 帧，并只保存诊断表、逐帧概率、片段表和病人级 Grad-CAM。普通 `keyframes/` 原图默认关闭；需要调试规则最高分帧时再加 `--save-keyframes`。

反光控制：质量 gate 现在默认要求 `white_ratio <= 0.12`，会剔除大面积饱和白色反光帧，避免反光被 8 分类模型当作 `声带白斑` 证据。诊断表还会把阈值以上证据少于 `--min-evidence-duration-sec 1.0` 秒的病例标为 `low_confidence`，并把 `diagnosis_call_zh` 写成 `待人工复核`；这类输出只能用于复核，不应当直接当作稳定视频诊断。

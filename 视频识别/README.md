# 视频识别

这个目录保存喉镜视频的弱监督推理脚手架。它不重新训练视频模型，而是抽帧后按唯一标准管线筛选有效帧，再复用 `图像识别/` 中的 8 分类 checkpoint 做疾病识别。

## 主要入口

| 文件 | 用途 |
|------|------|
| `video_inference.py` | 抽帧、质量 gate、ROI validity/reflection gate、声门二分类 gate、8 分类病种 top-fraction 投票聚合、证据秒段、Grad-CAM 和诊断叠加视频输出 |

## 当前标准管线

1. 基础质量检查：亮度、清晰度、黑边比例和饱和白色反光比例。
2. ROI validity/reflection gate：保留 ROI 有效且非严重反光帧；轻度反光帧保留但写入审计字段。
3. 声门/非声门 glottis gate：使用二分类模型筛出声门/有效声带帧。
4. 8 分类疾病模型：只对质量、ROI、glottis 三个 gate 交集中的帧做疾病推理；传入 `--no-roi-gate` 时仅跳过 ROI 条件，传入 `--no-glottis-gate` 时仅跳过 glottis 条件。
5. 视频级聚合：对每个有效帧，在候选 VOC 病种中取概率位于 `--top-fraction` 的类别作为投票池；默认 `0.20` 且 7 个 VOC 病种时等价于每帧 top1 投票，最终选择票数最多的病种。证据秒段仍按预测病种概率 `>= --frame-threshold` 计算。

输出的 `variant` / `diagnosis_variant` 为兼容旧 CSV schema 保留，但现在固定为 `standard_roi_glottis_top_fraction`。`video_predictions.csv` 每个视频只输出一行，`summary.json` 也只统计这一套标准口径。

`frame_predictions.csv` 里的 `non_voc_prob` 和 `voc_sum_prob` 只用于审计 8 分类模型的帧级倾向，不再作为视频 gate 或二次筛选规则。

传入 `--render-video` 时会额外在项目根目录 `Output_Video/` 下输出 H.264/yuv420p 诊断叠加视频：8fps 抽样帧按 1/8 秒写回视频，右侧图像叠加 ROI softweight 明暗遮罩和预测病种 Grad-CAM，左侧黑边使用中文字体显示声门 0/1、当前/最长证据时间、最终标注和近 5 秒投票趋势折线图。若 `--candidate-labels` 仍为 `auto`，渲染模式会自动切到 `all-voc`，保证折线图展示 7 个 VOC 病种。

默认声门 gate:

```text
8-class model = Results/main/roi_reflection/eight_class_roi_soft/best_model.pth, fallback Results/main/best_model.pth
glottis gate = Results/main/glottis_binary_benchmarks/20260505_183333_parallel/swin_base/best_model.pth
ROI localizer = Results/main/roi_reflection/localizer_transformer_20260507_111744/roi_localizer_best.pth
reflection gate = Results/main/roi_reflection/reflection_full_corrected/roi_reflection_best.pth
glottis threshold = 0.94
fallback_threshold = disabled by default
```

`--max-segment-gap-sec` 默认 `1.0` 秒，会把闪光、瞬时模糊等短暂中断前后的声门片段合并。若临床视频有效帧太少，可显式加 `--glottis-gate-fallback-threshold 0.62` 提高召回，但这会增加非声门帧漏入疾病模型的风险。

推荐从项目根目录运行，例如：

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

默认 ROI checkpoint 已指向最新的 `localizer_transformer_20260507_111744` 和 `reflection_full_corrected`；如需复现实验，可显式传入 `--roi-localizer-model` / `--roi-reflection-model`。

未标注 `tbr` 批次可在同一命令基础上改 `--video-root /mnt/data/LarynxData/videos/tbr`，并加 `--allow-unlabeled --candidate-labels all-voc`。

默认每秒抽 8 帧，并只保存诊断表、逐帧概率、片段表和病人级 Grad-CAM。普通 `keyframes/` 原图默认关闭；需要调试标准管线最高分帧时再加 `--save-keyframes`。

反光控制：质量 gate 现在默认要求 `white_ratio <= 0.12`，会剔除大面积饱和白色反光帧，避免反光被 8 分类模型当作 `声带白斑` 证据。诊断表还会把阈值以上证据少于 `--min-evidence-duration-sec 1.0` 秒的病例标为 `low_confidence`，并把 `diagnosis_call_zh` 写成 `待人工复核`；这类输出只能用于复核，不应当直接当作稳定视频诊断。

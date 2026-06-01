# 视频识别

这个目录保存喉镜视频的弱监督推理脚手架。它不重新训练视频模型，而是抽帧后按唯一标准管线筛选有效帧，再复用 `图像识别/` 中的 8 分类 checkpoint 做疾病识别。

## 主要入口

| 文件 | 用途 |
|------|------|
| `video_inference.py` | 抽帧、质量 gate、声门二分类 gate、YOLO-Pose + DINOv3 ROI 裁剪、视频级全帧兜底、8 分类病种 top-fraction 投票聚合、证据秒段、Grad-CAM 和诊断叠加视频输出 |

## 当前标准管线

1. 基础质量检查：亮度、清晰度、黑边比例和饱和白色反光比例。
2. 声门/非声门 glottis gate：默认先用二分类模型筛出声门/有效声带帧，再进入 ROI。
3. YOLO-Pose + DINOv3 ROI 裁剪：对通过质量和 0/1 gate 的帧先裁掉已有黑边，再给裁后图统一加黑边跑三点声门 YOLO-Pose；YOLO 给出候选框和 A/L/R 三点后，用 DINOv3 点区域辅助头打分。只有 YOLO/DINO 分数达到接受阈值时才裁剪框内小 ROI 并拉伸成正方形；无框、空裁剪或分数不足时，单帧 ROI 仍判无效。
4. 视频级全帧兜底：如果整个视频没有任何 ROI 接受帧，但存在通过质量 + 0/1 gate 的上游有效帧，则把这些原帧送入 Swin 做一次兜底识别，避免严格 ROI 把整段视频清空；这个兜底只在视频级触发，不做逐帧 ROI 回退。
5. 8 分类疾病模型：优先只对被 ROI 接受的小框正方形做疾病推理；触发视频级兜底时才对上游有效原帧做推理。传入 `--no-roi-gate` 会显式跳过 ROI 并使用整帧输入。
6. 视频级聚合：对每个有效帧，在候选 VOC 病种中取概率位于 `--top-fraction` 的类别作为投票池；默认 `0.20` 且 7 个 VOC 病种时等价于每帧 top1 投票，最终选择票数最多的病种。证据秒段仍按预测病种概率 `>= --frame-threshold` 计算。

输出的 `variant` / `diagnosis_variant` 为兼容旧 CSV schema 保留，但现在固定为 `standard_roi_glottis_top_fraction`。`video_predictions.csv` 每个视频只输出一行，`summary.json` 也只统计这一套标准口径。

`frame_predictions.csv` 里的 `non_voc_prob` 和 `voc_sum_prob` 只用于审计 8 分类模型的帧级倾向，不再作为视频 gate 或二次筛选规则。

传入 `--render-video` 时会额外在项目根目录 `Output_Video/` 下输出 H.264/yuv420p 诊断叠加视频：8fps 抽样帧按 1/8 秒写回视频，右侧图像叠加 YOLO ROI 明暗遮罩和预测病种 Grad-CAM，左侧黑边使用中文字体显示 ROI/DINO 状态、当前/最长证据时间、最终标注和近 5 秒投票趋势折线图。若 `--candidate-labels` 仍为 `auto`，渲染模式会自动切到 `all-voc`，保证折线图展示 7 个 VOC 病种。

默认 checkpoint:

```text
8-class model = Results/main/roi_reflection/eight_class_roi_soft/best_model.pth, fallback Results/main/best_model.pth
glottis gate = Results/main/glottis_binary_benchmarks/20260505_183333_parallel/swin_base/best_model.pth
YOLO-Pose ROI = /home/or1ngelinux/CVProjects/Larynx/YOLOPoseVocalFold/Results/dinov3_aux_full_pipeline_v12_20260524/stage1_pose/yolo11m_stage1_manual_mixedneg60_blackpad_containment_l0p05_v12/weights/best.pt
DINOv3 ROI aux = /home/or1ngelinux/CVProjects/Larynx/YOLOPoseVocalFold/Results/dinov3_keypoint_aux/dinov3_vits16_oriented_point_region_hardneg_448_ldp200_v12_20260524/weights/best_aux_head.pt
DINOv3 ROI code root = /home/or1ngelinux/CVProjects/Larynx/YOLOPoseVocalFold
ROI accept threshold = 0.25
DINO aux accept threshold = 0.30
glottis gate = enabled by default
video-level full-frame fallback = enabled by default
```

`--max-segment-gap-sec` 默认 `1.0` 秒，会把闪光、瞬时模糊等短暂中断前后的有效片段合并。若要复现不使用 0/1 gate 的旧口径，可显式加 `--no-glottis-gate`；若要关闭整段视频无 ROI 时的全帧兜底，可加 `--no-roi-video-fallback`。

推荐从项目根目录运行，例如：

```bash
python 视频识别/video_inference.py \
  --video-root /mnt/data/LarynxData/videos/classified_videos \
  --output-dir ../../Results/main/video_inference_classified_standard \
  --sample-fps 8 \
  --batch-size 256 \
  --top-fraction 0.20 \
  --render-video \
  --min-evidence-duration-sec 0.5
```

默认 ROI checkpoint 已指向 2026-05-24 更新的一体化 ROI v12：YOLO-Pose 三点声门定位模型 + DINOv3 point-region auxiliary head；如需替换可显式传入 `--roi-localizer-model` 和 `--roi-dino-aux-model`。`--roi-valid-threshold` 表示 YOLO bbox 候选阈值，`--roi-accept-threshold` 和 `--roi-dino-aux-accept-threshold` 控制最终是否接受裁剪。逐帧表中 `roi_valid_prob` 是 YOLO 置信度经 DINO 正向证据修正后的 ROI 分数，`roi_yolo_prob` 和 `roi_dino_aux_score` 分别保留两段原始审计信息；分数不足的帧会写入 `roi_filter_reason=roi_reject_low_yolo_dino_score`。如果整段视频没有任何 ROI 接受帧，兜底帧会写入 `roi_filter_reason=roi_video_fallback_full_frame`。

未标注 `tbr` 批次可在同一命令基础上改 `--video-root /mnt/data/LarynxData/videos/tbr`，并加 `--allow-unlabeled --candidate-labels all-voc`。

默认每秒抽 8 帧，并只保存诊断表、逐帧概率、片段表和病人级 Grad-CAM。普通 `keyframes/` 原图默认关闭；需要调试标准管线最高分帧时再加 `--save-keyframes`。

反光控制：质量 gate 现在默认要求 `white_ratio <= 0.12`，会剔除大面积饱和白色反光帧，避免反光被 8 分类模型当作 `声带白斑` 证据。诊断表还会把阈值以上证据少于 `--min-evidence-duration-sec 1.0` 秒的病例标为 `low_confidence`，并把 `diagnosis_call_zh` 写成 `待人工复核`；这类输出只能用于复核，不应当直接当作稳定视频诊断。

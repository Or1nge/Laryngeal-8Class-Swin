# Q1 Visual-Model Reference Matrix

筛选范围：2025-07-01 至 2026-06-30；优先选择开放获取、可下载原文、与喉镜/医学视觉分类任务相近的 Q1 期刊论文。

## Selected Papers

| Paper | Journal / Q1 evidence | Published | Task similarity | Models | Reported metrics | Reported figures/tables | Local PDF |
|---|---|---:|---|---|---|---|---|
| Emre et al., "Vocal Fold Disorders Classification and Optimization of a Custom Video Laryngoscopy Dataset Through Structural Similarity Index and a Deep Learning-Based Approach" | Journal of Clinical Medicine. The journal page lists JCR Q1 and CiteScore Q1. | 2025-09-29 | Directly related: video laryngoscopy key-frame classification for healthy/nodule/polyp. | SSIM key-frame selection; MobileNetV2, NASNetMobile, DenseNet121/169/201, Xception; final hybrid MobileNetV2 + Xception with fine-tuning. | Precision, recall/sensitivity, F1-score, accuracy, MCC, cross-validation mean +/- std. | Dataset creation, frame extraction, method workflow, frozen-vs-fine-tuned confusion matrices, ROC curves, cross-validation table. | `jcm_2025_vocal_fold_disorders.pdf` |
| Kim et al., "Development and validation of a deep learning model for identifying high-quality laryngoscopic images" | Scientific Reports. SCImago lists Scientific Reports as Q1; Nature article is open access. | 2026-02-10 | Directly related: laryngoscopic image quality classification and filtering for downstream laryngeal AI datasets. | AlexNet, ResNet-50, MobileNetV2, ConvNeXt-tiny, ViT-B/16, DaViT, Swin Transformer V1-tiny, Swin Transformer V2-base. | Accuracy, precision, recall, F1-score, AUROC, AUPRC; binary high-vs-other metrics; inference speed; DeLong, McNemar, stratified paired bootstrap with Holm-Bonferroni correction. | Pipeline, confusion matrices, ROC curves, precision-recall curves, label-specific ROC, architecture-wise Grad-CAM, quality-specific Grad-CAM, rare-feature Grad-CAM, external validation. | `sci_rep_2026_laryngoscopic_quality.pdf` |
| Merabet et al., "Few-shot learning and explainable AI for colon cancer histopathology: A prototypical network with multi-technique interpretability" | International Journal of Medical Informatics. The journal is listed as Q1 in health care sciences/services by OOIR; the article is open access. | 2025-11-03 online / 2026 volume | Same visual-model direction on another body site: histopathology image classification with explainability and external validation. | Prototypical Network with ConvNeXt-Tiny backbone; compared with lightweight backbones; Grad-CAM and LIME for XAI. | Accuracy, balanced accuracy, precision, recall, F1-score, macro-F1, cross-validation mean +/- std, confidence intervals, external EBHI accuracy, ROC AUC, deletion/insertion AUC for XAI faithfulness. | Framework, preprocessing/training flow, convergence/training curves, cross-validation metrics, confusion matrix, ROC curve, Grad-CAM/LIME explanations, t-SNE/embedding visualization, XAI faithfulness tables. | `ijmi_2026_colon_fewshot_xai.pdf` |

## Project Outputs Generated

All project-side outputs are under `Results/literature_aligned_metrics/`.

| Literature pattern | Project output |
|---|---|
| Classification tables with accuracy, precision, recall/sensitivity, F1, AUROC, AUPRC | `summary_metrics.csv`, `per_class_metrics.csv` |
| Binary high-vs-other style analysis adapted to this project's hierarchy | `voc_binary_metrics.csv`, `voc_binary_roc_pr_confusion_test.png` |
| Confusion matrices | `confusion_matrix_test_counts.png`, `confusion_matrix_test_normalized.png` |
| ROC curves | `roc_curves_test.png` |
| Precision-recall curves | `precision_recall_curves_test.png` |
| Cross-validation / uncertainty-style reporting | `bootstrap_ci_test.csv` with stratified bootstrap 95% CIs |
| Training/convergence curves | `training_curves_literature_style.png` |
| Per-class comparison bars and class support | `per_class_metric_bars_test.png`, `support_distribution_test.png` |
| Explainability heatmaps | `gradcam_maps_existing.png` copied from the current model output |
| Embedding visualization inspired by the pathology few-shot paper | `tsne_test_features.png` |
| Image-level audit table | `predictions_train.csv`, `predictions_val.csv`, `predictions_test.csv` |

Notes:

- DeLong and McNemar tests require two or more competing models on the same cases. This project currently has one active final checkpoint, so the reproducible output uses stratified bootstrap confidence intervals instead.
- LIME and deletion/insertion XAI faithfulness AUC were not added to the default report because the current project pipeline standardizes on Grad-CAM and does not yet include a perturbation-XAI implementation or clinician/ROI validation protocol.

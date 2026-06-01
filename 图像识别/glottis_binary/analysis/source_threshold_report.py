#!/usr/bin/env python3
"""Build source-folder aware threshold diagnostics from saved predictions."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = Path(
    "/home/or1ngelinux/CVProjects/Larynx/laryngeal_multiclass/Results/main/"
    "glottis_binary_benchmarks/20260505_183333_parallel/swin_base"
)
FOCUS_SOURCE_FOLDERS = ["声带固定", "喉癌", "正常", "室带膨隆"]
PROFILE_RULES = {
    "high_specificity": {
        "constraint": "specificity_non_glottis >= 0.99 on val; choose highest glottis recall",
        "specificity_floor": 0.99,
    },
    "balanced": {
        "constraint": "choose highest val balanced accuracy",
    },
    "high_recall": {
        "constraint": "recall_glottis_sensitivity >= 0.98 on val; choose highest specificity",
        "recall_floor": 0.98,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    return parser.parse_args()


def label_to_int(values: pd.Series) -> np.ndarray:
    mapped = values.map({"non_glottis": 0, "glottis": 1, 0: 0, 1: 1})
    if mapped.isna().any():
        bad = sorted(values[mapped.isna()].astype(str).unique())
        raise ValueError(f"Unknown label values: {bad}")
    return mapped.astype(int).to_numpy()


def predictions_for_threshold(df: pd.DataFrame, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    y_true = label_to_int(df["true_label"] if "true_label" in df.columns else df["label"])
    y_pred = (df["prob_glottis"].astype(float).to_numpy() >= threshold).astype(int)
    return y_true, y_pred


def safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    return {
        "tn_non_glottis": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fp_non_glottis_as_glottis": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn_glottis_as_non_glottis": int(((y_true == 1) & (y_pred == 0)).sum()),
        "tp_glottis": int(((y_true == 1) & (y_pred == 1)).sum()),
    }


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float | int | None]:
    tp = counts["tp_glottis"]
    tn = counts["tn_non_glottis"]
    fp = counts["fp_non_glottis_as_glottis"]
    fn = counts["fn_glottis_as_non_glottis"]
    support = tp + tn + fp + fn
    positives = tp + fn
    negatives = tn + fp
    recall = safe_rate(tp, positives)
    specificity = safe_rate(tn, negatives)
    precision = safe_rate(tp, tp + fp)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = float(2 * precision * recall / (precision + recall))
    balanced_accuracy = None
    if recall is not None and specificity is not None:
        balanced_accuracy = float((recall + specificity) / 2)
    return {
        "support": support,
        "glottis_support": positives,
        "non_glottis_support": negatives,
        "accuracy": safe_rate(tp + tn, support),
        "balanced_accuracy": balanced_accuracy,
        "precision_glottis": precision,
        "recall_glottis_sensitivity": recall,
        "specificity_non_glottis": specificity,
        "f1_glottis": f1,
        "false_pass_non_glottis_rate": safe_rate(fp, negatives),
        "false_block_glottis_rate": safe_rate(fn, positives),
        **counts,
    }


def compute_overall_metrics(split_name: str, profile: str, threshold: float, df: pd.DataFrame) -> dict[str, Any]:
    y_true, y_pred = predictions_for_threshold(df, threshold)
    return {
        "split": split_name,
        "threshold_profile": profile,
        "threshold": threshold,
        **metrics_from_counts(binary_counts(y_true, y_pred)),
    }


def compute_source_metrics(split_name: str, profile: str, threshold: float, df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    y_true_all, y_pred_all = predictions_for_threshold(df, threshold)
    work = df.copy()
    work["_y_true"] = y_true_all
    work["_y_pred"] = y_pred_all
    for source_folder, group in work.groupby("source_folder", sort=True):
        y_true = group["_y_true"].to_numpy()
        y_pred = group["_y_pred"].to_numpy()
        counts = binary_counts(y_true, y_pred)
        label_values = sorted(group["true_label" if "true_label" in group.columns else "label"].astype(str).unique())
        rows.append(
            {
                "split": split_name,
                "threshold_profile": profile,
                "threshold": threshold,
                "source_folder": source_folder,
                "source_definition": ",".join(label_values),
                "is_focus_source": source_folder in FOCUS_SOURCE_FOLDERS,
                "prob_glottis_mean": float(group["prob_glottis"].mean()),
                "prob_glottis_median": float(group["prob_glottis"].median()),
                "prob_glottis_min": float(group["prob_glottis"].min()),
                "prob_glottis_max": float(group["prob_glottis"].max()),
                **metrics_from_counts(counts),
            }
        )
    return pd.DataFrame(rows)


def read_threshold_metrics(run_dir: Path, split_name: str, predictions: pd.DataFrame) -> pd.DataFrame:
    path = run_dir / f"threshold_metrics_{split_name}.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for threshold in np.round(np.arange(0.01, 1.0, 0.01), 2):
        rows.append(compute_overall_metrics(split_name, "grid", float(threshold), predictions))
    return pd.DataFrame(rows)


def select_profile_thresholds(val_metrics: pd.DataFrame) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}

    specificity_floor = PROFILE_RULES["high_specificity"]["specificity_floor"]
    candidates = val_metrics[val_metrics["specificity_non_glottis"] >= specificity_floor].copy()
    if candidates.empty:
        candidates = val_metrics.copy()
    row = candidates.sort_values(
        ["recall_glottis_sensitivity", "f1_glottis", "balanced_accuracy", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]
    selected["high_specificity"] = {
        "threshold": float(row["threshold"]),
        "selection": PROFILE_RULES["high_specificity"]["constraint"],
    }

    row = val_metrics.sort_values(
        ["balanced_accuracy", "f1_glottis", "accuracy", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]
    selected["balanced"] = {
        "threshold": float(row["threshold"]),
        "selection": PROFILE_RULES["balanced"]["constraint"],
    }

    recall_floor = PROFILE_RULES["high_recall"]["recall_floor"]
    candidates = val_metrics[val_metrics["recall_glottis_sensitivity"] >= recall_floor].copy()
    if candidates.empty:
        candidates = val_metrics.copy()
    row = candidates.sort_values(
        ["specificity_non_glottis", "balanced_accuracy", "f1_glottis", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]
    selected["high_recall"] = {
        "threshold": float(row["threshold"]),
        "selection": PROFILE_RULES["high_recall"]["constraint"],
    }
    return selected


def profile_metrics_block(
    profile: str,
    threshold: float,
    overall: pd.DataFrame,
    source_metrics: pd.DataFrame,
) -> dict[str, Any]:
    profile_overall = overall[overall["threshold_profile"] == profile].set_index("split")
    focus_rows = source_metrics[
        (source_metrics["threshold_profile"] == profile)
        & (source_metrics["source_folder"].isin(FOCUS_SOURCE_FOLDERS))
    ].copy()
    return {
        "threshold": threshold,
        "selection": PROFILE_RULES[profile]["constraint"],
        "overall": profile_overall.to_dict(orient="index"),
        "focus_source_folders": focus_rows.to_dict(orient="records"),
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def write_markdown_report(
    output_path: Path,
    run_dir: Path,
    profiles: dict[str, dict[str, Any]],
    overall: pd.DataFrame,
    source_metrics: pd.DataFrame,
) -> None:
    lines = [
        "# Source-aware glottis gate threshold report",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        "- Scope: offline analysis from saved val/test predictions; no retraining and no checkpoint changes.",
        "",
        "## Threshold recommendations",
        "",
        "| Profile | Threshold | Val recall | Val specificity | Test recall | Test specificity | Test FN | Test FP | Rule |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for profile, info in profiles.items():
        threshold = info["threshold"]
        val = overall[(overall["threshold_profile"] == profile) & (overall["split"] == "val")].iloc[0]
        test = overall[(overall["threshold_profile"] == profile) & (overall["split"] == "test")].iloc[0]
        lines.append(
            "| "
            f"{profile} | {threshold:.2f} | "
            f"{val['recall_glottis_sensitivity']:.4f} | {val['specificity_non_glottis']:.4f} | "
            f"{test['recall_glottis_sensitivity']:.4f} | {test['specificity_non_glottis']:.4f} | "
            f"{int(test['fn_glottis_as_non_glottis'])} | {int(test['fp_non_glottis_as_glottis'])} | "
            f"{info['selection']} |"
        )

    lines.extend(
        [
            "",
            "## Focus source folders on test split",
            "",
            "| Profile | Source folder | Definition | Support | Recall | Specificity | FN | FP | Mean prob_glottis |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    focus = source_metrics[
        (source_metrics["split"] == "test") & (source_metrics["source_folder"].isin(FOCUS_SOURCE_FOLDERS))
    ].copy()
    focus["source_folder"] = pd.Categorical(
        focus["source_folder"],
        categories=FOCUS_SOURCE_FOLDERS,
        ordered=True,
    )
    focus = focus.sort_values(["threshold_profile", "source_folder"])
    for _, row in focus.iterrows():
        recall = "NA" if pd.isna(row["recall_glottis_sensitivity"]) else f"{row['recall_glottis_sensitivity']:.4f}"
        specificity = "NA" if pd.isna(row["specificity_non_glottis"]) else f"{row['specificity_non_glottis']:.4f}"
        lines.append(
            "| "
            f"{row['threshold_profile']} | {row['source_folder']} | {row['source_definition']} | "
            f"{int(row['support'])} | {recall} | {specificity} | "
            f"{int(row['fn_glottis_as_non_glottis'])} | {int(row['fp_non_glottis_as_glottis'])} | "
            f"{row['prob_glottis_mean']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Reading notes",
            "",
            "- Positive source folders should be read mainly by recall and FN; negative source folders should be read mainly by specificity and FP.",
            "- Large source-folder drops under high_specificity indicate threshold/source-definition tension, not necessarily backbone overfitting.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir or (run_dir / "analysis" / "source_threshold_diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = {
        split_name: pd.read_csv(run_dir / f"predictions_{split_name}.csv")
        for split_name in (args.val_split, args.test_split)
    }
    val_metrics = read_threshold_metrics(run_dir, args.val_split, predictions[args.val_split])
    profiles = select_profile_thresholds(val_metrics)

    overall_rows: list[dict[str, Any]] = []
    source_frames: list[pd.DataFrame] = []
    error_frames: list[pd.DataFrame] = []
    for profile, info in profiles.items():
        threshold = info["threshold"]
        for split_name, df in predictions.items():
            overall_rows.append(compute_overall_metrics(split_name, profile, threshold, df))
            source_frames.append(compute_source_metrics(split_name, profile, threshold, df))
            y_true, y_pred = predictions_for_threshold(df, threshold)
            errors = df.loc[y_true != y_pred].copy()
            errors["threshold_profile"] = profile
            errors["threshold"] = threshold
            errors["pred_label_at_profile"] = np.where(y_pred[y_true != y_pred] == 1, "glottis", "non_glottis")
            errors["error_type_at_profile"] = np.where(
                y_true[y_true != y_pred] == 1,
                "false_negative_glottis_as_non_glottis",
                "false_positive_non_glottis_as_glottis",
            )
            error_frames.append(errors)

    overall = pd.DataFrame(overall_rows)
    source_metrics = pd.concat(source_frames, ignore_index=True)
    errors = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame()

    overall.to_csv(output_dir / "overall_threshold_profiles.csv", index=False)
    source_metrics.to_csv(output_dir / "source_folder_metrics_all.csv", index=False)
    for split_name in predictions:
        source_metrics[source_metrics["split"] == split_name].to_csv(
            output_dir / f"source_folder_metrics_{split_name}.csv",
            index=False,
        )
        errors[errors["split"] == split_name].to_csv(
            output_dir / f"error_samples_{split_name}_threshold_profiles.csv",
            index=False,
        )

    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir.resolve()),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_files": {
            split_name: str((run_dir / f"predictions_{split_name}.csv").resolve())
            for split_name in predictions
        },
        "profiles": {
            profile: profile_metrics_block(profile, info["threshold"], overall, source_metrics)
            for profile, info in profiles.items()
        },
    }
    (output_dir / "threshold_recommendations.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(
        output_dir / "source_threshold_report.md",
        run_dir,
        profiles,
        overall,
        source_metrics,
    )
    print(f"Report written to: {output_dir}")


if __name__ == "__main__":
    main()

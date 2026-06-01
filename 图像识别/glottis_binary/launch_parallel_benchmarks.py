#!/usr/bin/env python3
"""Launch glottis binary model benchmarks concurrently."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from common import (
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SPLIT_PATH,
    MODEL_REGISTRY,
    build_binary_split,
    default_train_config,
    write_json,
)


PROFILE_BATCHES = {
    "conservative": {
        "resnet50": (128, 384),
        "vit_base": (64, 192),
        "swin_base": (48, 160),
        "supcon_swin_base": (32, 128),
    },
    "balanced": {
        "resnet50": (192, 512),
        "vit_base": (96, 256),
        "swin_base": (80, 224),
        "supcon_swin_base": (48, 160),
    },
    "aggressive": {
        "resnet50": (256, 768),
        "vit_base": (128, 384),
        "swin_base": (112, 320),
        "supcon_swin_base": (64, 224),
    },
}


def parse_args() -> argparse.Namespace:
    cfg = default_train_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--build-split", action="store_true")
    parser.add_argument("--force-split", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY))
    parser.add_argument("--profile", choices=sorted(PROFILE_BATCHES), default="balanced")
    parser.add_argument("--epochs", type=int, default=cfg["epochs"])
    parser.add_argument("--patience", type=int, default=cfg["patience"])
    parser.add_argument("--supcon-epochs", type=int, default=cfg["supcon_epochs"])
    parser.add_argument("--supcon-patience", type=int, default=cfg["supcon_patience"])
    parser.add_argument("--learning-rate", type=float, default=cfg["learning_rate"])
    parser.add_argument("--supcon-learning-rate", type=float, default=cfg["supcon_learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=cfg["weight_decay"])
    parser.add_argument("--label-smoothing", type=float, default=cfg["label_smoothing"])
    parser.add_argument("--cache-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def collect_summary(run_root: Path, models: list[str]) -> pd.DataFrame:
    rows = []
    for model_key in models:
        model_dir = run_root / model_key
        metrics_path = model_dir / "metrics.csv"
        threshold_path = model_dir / "recommended_threshold.json"
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path)
        test = metrics[(metrics["split"] == "test") & (metrics["threshold_mode"] == "recommended_gate")]
        if test.empty:
            continue
        row = test.iloc[0].to_dict()
        row["model_key"] = model_key
        row["checkpoint"] = str((model_dir / "best_model.pth").resolve())
        if threshold_path.exists():
            threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
            row["recommended_threshold"] = threshold["threshold"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    for model_key in args.models:
        if model_key not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model '{model_key}'. Available: {sorted(MODEL_REGISTRY)}")

    if args.build_split or not args.split.exists():
        build_binary_split(
            dataset_root=args.dataset_root,
            output_path=args.split,
            manifest_path=args.manifest,
            force=args.force_split or not args.split.exists(),
        )

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S_parallel")
    run_root = args.output_root / run_name
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_root / "parallel_launch_config.json",
        {
            "models": args.models,
            "profile": args.profile,
            "profile_batches": PROFILE_BATCHES[args.profile],
            "command": " ".join(sys.argv),
        },
    )

    processes: list[tuple[str, subprocess.Popen, Any]] = []
    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    script = Path(__file__).resolve().parent / "train_benchmarks.py"
    try:
        for model_key in args.models:
            batch_size, eval_batch_size = PROFILE_BATCHES[args.profile][model_key]
            log_path = logs_dir / f"{model_key}.log"
            command = [
                sys.executable,
                "-u",
                str(script),
                "--models",
                model_key,
                "--split",
                str(args.split),
                "--manifest",
                str(args.manifest),
                "--dataset-root",
                str(args.dataset_root),
                "--output-root",
                str(args.output_root),
                "--run-name",
                run_name,
                "--cache-device",
                args.cache_device,
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--supcon-epochs",
                str(args.supcon_epochs),
                "--supcon-patience",
                str(args.supcon_patience),
                "--batch-size",
                str(batch_size),
                "--eval-batch-size",
                str(eval_batch_size),
                "--learning-rate",
                str(args.learning_rate),
                "--supcon-learning-rate",
                str(args.supcon_learning_rate),
                "--weight-decay",
                str(args.weight_decay),
                "--label-smoothing",
                str(args.label_smoothing),
            ]
            if args.no_pretrained:
                command.append("--no-pretrained")
            log_file = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[2],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            processes.append((model_key, process, log_file))
            print(f"Started {model_key}: pid={process.pid}, log={log_path}")

        statuses = {}
        while processes:
            still_running = []
            for model_key, process, log_file in processes:
                code = process.poll()
                if code is None:
                    still_running.append((model_key, process, log_file))
                    continue
                log_file.close()
                statuses[model_key] = code
                print(f"Finished {model_key}: exit_code={code}")
            processes = still_running
            write_json(run_root / "parallel_status.json", statuses)
            if processes:
                time.sleep(20)
    except KeyboardInterrupt:
        print("Interrupted; forwarding SIGINT to child process groups.")
        for _model_key, process, log_file in processes:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            log_file.close()
        raise

    summary = collect_summary(run_root, args.models)
    if not summary.empty:
        summary.to_csv(run_root / "benchmark_summary.csv", index=False)
        summary.to_csv(run_root / "parallel_benchmark_summary.csv", index=False)
        sort_cols = ["specificity_non_glottis", "accuracy", "recall_glottis_sensitivity"]
        print(summary.sort_values(sort_cols, ascending=False).to_string(index=False))
    failed = {model_key: code for model_key, code in statuses.items() if code != 0}
    if failed:
        raise SystemExit(f"Some model runs failed: {failed}")
    print(f"Parallel benchmark complete: {run_root}")


if __name__ == "__main__":
    main()

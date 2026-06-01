"""Run the training phases with an auto-named pipeline log.

Usage:
    python train_pipeline.py

The script redirects its own stdout/stderr, and the child training scripts'
stdout/stderr, to:
    <results_dir>/pipeline_<YYYYMMDD_HHMMSS>.log

It also updates:
    <results_dir>/pipeline_latest.log
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from shared import BASE_DIR, RESULTS_DIR, PHASE2_FINAL_METRICS_PATH, PHASE3_HISTORY_PATH


def update_latest_link(log_path):
    latest_path = Path(RESULTS_DIR) / "pipeline_latest.log"
    try:
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        latest_path.symlink_to(log_path.name)
    except OSError:
        latest_path.write_text(str(log_path) + "\n")


def redirect_output(log_path):
    log_file = open(log_path, "a", buffering=1)
    os.dup2(log_file.fileno(), sys.stdout.fileno())
    os.dup2(log_file.fileno(), sys.stderr.fileno())
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    return log_file


def run_step(name, command, env_overrides=None):
    print("=" * 80)
    print(f"[pipeline] starting {name}: {' '.join(command)}")
    print("=" * 80)
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    result = subprocess.run(command, cwd=BASE_DIR, env=env)
    print(f"[pipeline] {name} exit_status={result.returncode}")
    return result.returncode


def read_phase3_skip_status():
    history_path = Path(PHASE3_HISTORY_PATH)
    if not history_path.exists():
        return False, "missing phase3 history"
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"could not parse phase3 history: {exc}"
    if not history:
        return False, "empty phase3 history"
    first = history[0]
    return bool(first.get("skipped", False)), first.get("reason", "")


def read_phase2_final_metrics():
    metrics_path = Path(PHASE2_FINAL_METRICS_PATH)
    if not metrics_path.exists():
        return None
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[pipeline] could not parse Phase 2 final metrics: {exc}")
        return None


def print_phase2_result(metrics_payload):
    if not metrics_payload:
        print("[pipeline] Phase 2 final metrics file not found; keeping Phase 2 outputs.")
        return
    metrics = metrics_payload.get("metrics", {})
    print("[pipeline] stopping after Phase 2 because Phase 3 was skipped.")
    for split in ("train", "val", "test"):
        row = metrics.get(split, {})
        if row:
            print(
                f"[pipeline] Phase 2 {split}: "
                f"F1={row.get('f1', float('nan')):.4f} "
                f"Acc={row.get('acc', float('nan')):.4f} "
                f"AUC={row.get('auc', float('nan')):.4f} "
                f"Loss={row.get('loss', float('nan')):.4f}"
            )


def archive_phase2_result(metrics_payload):
    if not metrics_payload:
        return
    if str(os.environ.get("LARYNX_ARCHIVE_RUNS", "1")).lower() in {"0", "false", "no"}:
        print("[pipeline] final archive disabled by LARYNX_ARCHIVE_RUNS=0.")
        return
    from train_phase2 import archive_finished_run

    metrics = metrics_payload.get("metrics", {})
    archive_dir = archive_finished_run(
        metrics_payload.get("config", {}),
        metrics.get("train", {}),
        metrics.get("val", {}),
        metrics.get("test", {}),
        metrics.get("hierarchical_val", {}),
        metrics.get("hierarchical_test", {}),
        metrics_payload.get("best_epoch", 0),
    )
    if archive_dir is not None:
        print(f"[pipeline] archived Phase 2 final outputs: {archive_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1 through Phase 4.")
    parser.add_argument("--through-phase", type=int, default=4, choices=[1, 2, 3, 4])
    parser.add_argument("--phase1-config", default=None, help="Optional config path for train_phase1.py.")
    parser.add_argument("--phase2-config", default=None, help="Optional config path for train_phase2.py.")
    parser.add_argument("--phase3-config", default=None, help="Optional config path for train_phase3.py.")
    parser.add_argument("--phase4-config", default=None, help="Optional config path for train_phase4.py.")
    args = parser.parse_args()

    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = results_dir / f"pipeline_{started_at}.log"
    update_latest_link(log_path)

    log_file = redirect_output(log_path)
    try:
        print(f"[pipeline] log={log_path}")
        print(f"[pipeline] started_at={datetime.now().isoformat(timespec='seconds')}")
        print(f"[pipeline] cwd={BASE_DIR}")
        print(f"[pipeline] python={sys.executable}")

        phase1_cmd = [sys.executable, "-u", "train_phase1.py"]
        if args.phase1_config:
            phase1_cmd.extend(["--config", args.phase1_config])

        phase2_cmd = [sys.executable, "-u", "train_phase2.py"]
        if args.phase2_config:
            phase2_cmd.extend(["--config", args.phase2_config])

        phase3_cmd = [sys.executable, "-u", "train_phase3.py"]
        if args.phase3_config:
            phase3_cmd.extend(["--config", args.phase3_config])

        phase4_cmd = [sys.executable, "-u", "train_phase4.py"]
        if args.phase4_config:
            phase4_cmd.extend(["--config", args.phase4_config])

        steps = [("phase1", phase1_cmd), ("phase2", phase2_cmd)]
        if args.through_phase >= 3:
            steps.append(("phase3", phase3_cmd))
        if args.through_phase >= 4:
            steps.append(("phase4", phase4_cmd))

        status = 0
        for idx, (name, command) in enumerate(steps, start=1):
            env_overrides = None
            if name == "phase2" and args.through_phase >= 4:
                env_overrides = {"LARYNX_ARCHIVE_RUNS": "0"}
            status = run_step(name, command, env_overrides=env_overrides)
            if status != 0:
                remaining = ", ".join(step_name for step_name, _cmd in steps[idx:])
                if remaining:
                    print(f"[pipeline] skipping {remaining} because {name} failed")
                break
            if name == "phase3" and args.through_phase >= 4:
                skipped, reason = read_phase3_skip_status()
                if skipped:
                    if reason:
                        print(f"[pipeline] Phase 3 skipped: {reason}")
                    metrics_payload = read_phase2_final_metrics()
                    print_phase2_result(metrics_payload)
                    archive_phase2_result(metrics_payload)
                    break

        print(f"[pipeline] exit_status={status} finished_at={datetime.now().isoformat(timespec='seconds')}")
        return status
    finally:
        log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())

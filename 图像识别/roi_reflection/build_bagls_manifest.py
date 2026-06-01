#!/usr/bin/env python3
"""Build BAGLS ROI/reflection manifest and heuristic review queue."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from common import (
    DEFAULT_BAGLS_ROOT,
    ROI_RESULTS_DIR,
    build_manifest_rows,
    ensure_dir,
    write_json,
)


def maybe_extract(root: Path) -> None:
    for archive_name in ("training.zip", "test.zip"):
        archive = root / archive_name
        target = root / archive_name.replace(".zip", "")
        if archive.exists() and not target.exists():
            print(f"extract: {archive} -> {target}")
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bagls-root", type=Path, default=DEFAULT_BAGLS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROI_RESULTS_DIR / "manifest")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()

    bagls_root = args.bagls_root.expanduser().resolve()
    if not args.no_extract:
        maybe_extract(bagls_root)
    output_dir = ensure_dir(args.output_dir)
    manifest_path = output_dir / "bagls_roi_manifest.csv"
    review_path = output_dir / "review_queue.csv"
    summary_path = output_dir / "manifest_summary.json"
    if manifest_path.exists() and not args.force and not args.dry_run:
        raise FileExistsError(f"Refusing to overwrite {manifest_path}; pass --force.")

    df = build_manifest_rows(
        bagls_root,
        limit=args.limit,
        seed=args.seed,
        val_fraction=args.val_fraction,
    )
    review_df = df[df["review_needed"].astype(bool)].copy()
    summary = {
        "bagls_root": str(bagls_root),
        "rows": int(len(df)),
        "splits": df["split"].value_counts().to_dict(),
        "bagls_splits": df["bagls_split"].value_counts().to_dict(),
        "reflection_severity": df["reflection_severity"].value_counts(dropna=False).to_dict(),
        "review_needed": int(review_df.shape[0]),
        "recordings": int(df["recording_id"].nunique()),
        "dry_run": bool(args.dry_run),
        "limit": args.limit,
    }

    if args.dry_run:
        print(df.head(min(10, len(df))).to_string(index=False))
        print(summary)
        return

    df.to_csv(manifest_path, index=False)
    review_df.to_csv(review_path, index=False)
    write_json(summary_path, summary)
    print(f"Manifest: {manifest_path}")
    print(f"Review queue: {review_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

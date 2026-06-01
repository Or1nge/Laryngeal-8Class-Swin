#!/usr/bin/env python3
"""Build the read-only glottis/non-glottis binary manifest and patient split."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import DEFAULT_DATASET_ROOT, DEFAULT_MANIFEST_PATH, DEFAULT_SPLIT_PATH, build_binary_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--val-size-of-remaining", type=float, default=0.1111)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_binary_split(
        dataset_root=args.dataset_root,
        output_path=args.output,
        manifest_path=args.manifest,
        seed=args.seed,
        test_size=args.test_size,
        val_size_of_remaining=args.val_size_of_remaining,
        force=args.force,
    )
    print(f"Dataset root: {payload['dataset_root']}")
    print(f"Split written to: {args.output.resolve()}")
    print(f"Manifest written to: {args.manifest.resolve()}")
    print("Split summary:")
    for split_name in ("train", "val", "test"):
        stats = payload["stats"][split_name]
        print(
            f"  {split_name:>5s}: {stats['num_patient_groups']} patient groups, "
            f"{stats['num_images']} images, labels={stats['images_per_label']}"
        )
    print(f"Patient group overlap: {payload['audit']['patient_group_overlap']}")
    print(f"Patient alias overlap: {payload['audit']['patient_alias_overlap']}")
    print("Included source folders:")
    for row in payload["source_folders"]:
        if row["included"]:
            print(
                f"  {row['source_folder']}: {row['label']} "
                f"({row['num_images']} images; {row['mapping_reason']})"
            )


if __name__ == "__main__":
    main()

"""Build JSONL image manifests for ROI prediction from the frozen split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return parser.parse_args()


def get_patient_name(path: Path) -> str:
    basename = path.name
    if "_" in basename:
        return basename.split("_", 1)[0]
    stem = path.stem
    if len(stem) >= 8 and stem[:8].isdigit():
        return stem[:8]
    return stem


def main() -> None:
    args = parse_args()
    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    dataset_root = args.dataset_root or Path(split["dataset_root"])
    class_folders = split["class_folders"]
    patient_split = split.get("patients", split)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split = {name: [] for name in args.splits}
    patient_sets = {name: set(patient_split[name]) for name in args.splits}
    label_order = {name: idx for idx, name in enumerate(class_folders)}

    for class_name, folders in class_folders.items():
        for folder_name in folders:
            folder = dataset_root / folder_name
            if not folder.is_dir():
                continue
            for path in sorted(folder.rglob("*")):
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                patient_name = get_patient_name(path)
                for split_name, patient_names in patient_sets.items():
                    if patient_name in patient_names:
                        rows_by_split[split_name].append(
                            {
                                "original_source": str(path),
                                "source": str(path),
                                "class_name": class_name,
                                "source_folder": folder_name,
                                "label": label_order[class_name],
                                "patient_name": patient_name,
                            }
                        )
                        break

    summary = {
        "split_json": str(args.split_json),
        "dataset_root": str(dataset_root),
        "splits": {},
    }
    for split_name, rows in rows_by_split.items():
        out_path = args.out_dir / f"{split_name}_manifest.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["splits"][split_name] = {
            "manifest": str(out_path),
            "num_images": len(rows),
            "class_counts": dict(Counter(row["class_name"] for row in rows)),
        }

    summary_path = args.out_dir / "manifest_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

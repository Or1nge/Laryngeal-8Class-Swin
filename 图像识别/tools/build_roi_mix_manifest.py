"""Build original-image to ROI-crop mapping for ROI replacement training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--crop-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--action", default="auto_accept")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_to_original = {}
    source_to_action = {}
    for line in args.predictions.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        source = str(record.get("source", ""))
        original = str(record.get("original_source") or record.get("source", ""))
        if source and original:
            source_to_original[source] = original
            source_to_action[source] = str(record.get("action", ""))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.crop_manifest.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("cropped", "")).lower() != "true":
                continue
            if row.get("action") != args.action:
                continue
            source = str(row.get("source", ""))
            crop_path = str(row.get("crop_path") or row.get("output_path") or "")
            original = source_to_original.get(source)
            if not original or not crop_path:
                continue
            if not Path(original).exists() or not Path(crop_path).exists():
                continue
            rows.append(
                {
                    "image_path": original,
                    "roi_image_path": crop_path,
                    "action": source_to_action.get(source, row.get("action", "")),
                    "source": source,
                }
            )

    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "roi_image_path", "action", "source"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"out": str(args.out), "rows": len(rows), "action": args.action}, ensure_ascii=False))


if __name__ == "__main__":
    main()

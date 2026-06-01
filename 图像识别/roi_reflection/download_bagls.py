#!/usr/bin/env python3
"""Download BAGLS v1.1a training/test archives for local ROI gate work."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

from common import DEFAULT_BAGLS_ROOT, ensure_dir, write_json


ZENODO_FILES = {
    "training.zip": "https://zenodo.org/records/3762320/files/training.zip?download=1",
    "test.zip": "https://zenodo.org/records/3762320/files/test.zip?download=1",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, path: Path, skip_existing: bool = True) -> None:
    if skip_existing and path.exists() and path.stat().st_size > 0:
        print(f"exists: {path}")
        return
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"download: {url} -> {path}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bagls-root", type=Path, default=DEFAULT_BAGLS_ROOT)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-raw-videos", action="store_true", help="Reserved for future raw.zip/videos.zip downloads.")
    args = parser.parse_args()

    root = ensure_dir(args.bagls_root.expanduser())
    records = []
    for filename, url in ZENODO_FILES.items():
        path = root / filename
        download(url, path, skip_existing=args.skip_existing)
        records.append({"filename": filename, "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    write_json(
        root / "download_manifest.json",
        {
            "source": "https://zenodo.org/records/3762320",
            "license": "CC BY-NC-SA 4.0",
            "files": records,
            "raw_videos_downloaded": False,
        },
    )
    print(f"Wrote {root / 'download_manifest.json'}")


if __name__ == "__main__":
    main()

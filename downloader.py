from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"
ESC50_DIRNAME = "ESC-50-master"


def _download_file(url: str, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as resp, tmp_path.open("wb") as f:
            shutil.copyfileobj(resp, f)
        tmp_path.replace(out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def download_esc50(dest_dir: Path, force: bool = False) -> Path:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = dest_dir / ESC50_DIRNAME
    zip_path = dest_dir / "esc-50.zip"

    if dataset_dir.exists() and not force:
        print(f"ESC-50 already exists at {dataset_dir}")
        return dataset_dir

    if force and dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    if force or not zip_path.exists():
        print(f"Downloading ESC-50 from {ESC50_URL} ...")
        _download_file(ESC50_URL, zip_path)
        print(f"Saved to {zip_path}")

    print(f"Extracting to {dest_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    if not dataset_dir.exists():
        raise RuntimeError(f"Expected {dataset_dir} after extraction, but it was not found.")

    print(f"Done. Dataset is at {dataset_dir}")
    return dataset_dir

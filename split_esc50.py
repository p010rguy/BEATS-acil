from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, List

# 项目根目录（split_esc50.py 所在目录）
ROOT = Path(__file__).resolve().parent
ESC_ROOT = ROOT / "Dataset" / "ESC-50-master"

DEFAULT_SEED = 2026


def load_targets(meta_path: Path) -> List[int]:
    with meta_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        targets = {int(row["target"]) for row in reader}
    targets = sorted(targets)
    if len(targets) != 50:
        raise RuntimeError(f"Expected 50 targets, got {len(targets)}")
    return targets


def load_files_by_target_and_fold(meta_path: Path, test_fold: int):
    train_files: Dict[int, List[str]] = {}
    test_files: Dict[int, List[str]] = {}

    with meta_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target = int(row["target"])
            fold = int(row["fold"])
            fname = row["filename"]

            if fold == test_fold:
                test_files.setdefault(target, []).append(fname)
            else:
                train_files.setdefault(target, []).append(fname)

    return train_files, test_files


def make_splits(
    esc_root: Path,
    test_fold: int = 1,
    seed: int | None = None,
):
    meta_path = esc_root / "meta" / "esc50.csv"

    if not meta_path.exists():
        raise FileNotFoundError(f"esc50.csv not found at {meta_path}")

    if seed is None:
        seed = DEFAULT_SEED

    rng = random.Random(seed)

    # 1) 所有 50 类
    targets = load_targets(meta_path)
    print(f"Loaded {len(targets)} targets")
    rng.shuffle(targets)

    # 2) 25 + 5x5
    base_classes = targets[:25]
    inc_classes = [
        targets[25:30],
        targets[30:35],
        targets[35:40],
        targets[40:45],
        targets[45:50],
    ]

    # 3) 按 fold 收集文件
    train_files, test_files = load_files_by_target_and_fold(meta_path, test_fold)

    # sanity check
    for t in range(50):
        if t not in train_files:
            raise RuntimeError(f"Target {t} missing in train files")
        if t not in test_files:
            raise RuntimeError(f"Target {t} missing in test files")

    result = {
        "seed": seed,
        "test_fold": test_fold,
        "base_classes": base_classes,
        "incremental_classes": inc_classes,
        "train_files_by_target": {
            str(k): sorted(v) for k, v in train_files.items()
        },
        "test_files_by_target": {
            str(k): sorted(v) for k, v in test_files.items()
        },
    }

    return result


def resolve_audio_paths(
    files_by_target: Dict[str | int, List[str]],
    audio_dir: Path,
) -> Dict[str, List[str]]:
    audio_dir = audio_dir.resolve()
    resolved: Dict[str, List[str]] = {}
    missing: List[str] = []

    for target, files in files_by_target.items():
        key = str(target)
        paths: List[str] = []
        for fname in files:
            path = audio_dir / fname
            if not path.exists():
                missing.append(str(path))
            paths.append(str(path))
        resolved[key] = paths

    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} audio files under {audio_dir}. "
            f"Examples:\n{preview}"
        )

    return resolved


def add_audio_paths_to_splits(splits: dict, audio_dir: Path) -> dict:
    splits_with_audio = dict(splits)
    splits_with_audio["train_audio_by_target"] = resolve_audio_paths(
        splits["train_files_by_target"],
        audio_dir,
    )
    splits_with_audio["test_audio_by_target"] = resolve_audio_paths(
        splits["test_files_by_target"],
        audio_dir,
    )
    return splits_with_audio


__all__ = [
    "DEFAULT_SEED",
    "ESC_ROOT",
    "ROOT",
    "add_audio_paths_to_splits",
    "load_files_by_target_and_fold",
    "load_targets",
    "make_splits",
    "resolve_audio_paths",
]

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download the ESC-50 dataset.")
    parser.add_argument(
        "--dest",
        default="Dataset",
        help="Destination directory to place the dataset (default: Dataset).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract even if the dataset already exists.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducibility (default: 2026).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5).",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=5,
        help="Number of incremental phases (default: 5).",
    )
    parser.add_argument(
        "--baseclass",
        type=int,
        default=25,
        help="Number of base classes (default: 25).",
    )
    parser.add_argument(
        "--basetraining",
        action="store_true",
        help="Enable base training stage.",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Enable incremental learning stage.",
    )
    parser.add_argument(
        "--resume",
        default="",
        help="Path to a checkpoint to resume from (default: empty).",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="GPU index (default: -1, unused when using --device).",
    )
    parser.add_argument(
        "--device",
        default="mps",
        choices=["cuda", "mps", "cpu"],
        help="Training device preference (default: mps).",
    )
    parser.add_argument(
        "--fe-dim", 
        type=int, 
        default=None, 
        help="Feature expansion dim. None means disabled.")
    parser.add_argument(
        "--rg",
        type=float,
        default=1e-3,
        help="Ridge regularization coefficient for CLS alignment (default: 1e-3).",
    )


    return parser

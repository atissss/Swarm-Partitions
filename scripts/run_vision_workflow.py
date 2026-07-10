#!/usr/bin/env python3
"""Run the vision-model workflow end to end from one command.

Examples:
    python scripts/run_vision_workflow.py --mode train
    python scripts/run_vision_workflow.py --mode infer --dataset-root data/VisDrone2019-DET-val --model runs/exp002/best.pt
    python scripts/run_vision_workflow.py --mode evaluate --dataset-root data/VisDrone2019-DET-val --model runs/exp002/best.pt
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VISION_ROOT = ROOT / "vision_model"
PYTHON = sys.executable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the vision-model workflow")
    parser.add_argument(
        "--mode",
        choices=["train", "infer", "evaluate", "geotag"],
        default="infer",
        help="Which workflow stage to run",
    )
    parser.add_argument("--dataset-root", default=None, help="Dataset root for VisDrone-style splits")
    parser.add_argument("--train-root", default=None, help="Training split root")
    parser.add_argument("--val-root", default=None, help="Validation split root")
    parser.add_argument("--model", default=None, help="Path to a trained model checkpoint")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--run-name", default="exp003", help="Run name for training")
    parser.add_argument("--telemetry", default=None, help="Telemetry CSV path for geotagging")
    parser.add_argument("--camera", default="dji_mavic3", help="Camera config name")
    parser.add_argument("--python", default=str(PYTHON), help="Python interpreter to use")
    return parser


def run_command(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, cwd=VISION_ROOT, check=True)


def build_train_command(args: argparse.Namespace) -> list[str]:
    cmd = [args.python, "-m", "vision_model.cli.train", "dataset=visdrone"]
    if args.train_root:
        cmd.extend([f"dataset.splits.train.root={args.train_root}"])
    if args.val_root:
        cmd.extend([f"dataset.splits.val.root={args.val_root}"])
    else:
        cmd.extend(["dataset.splits.val.root=data/VisDrone2019-DET-val"])
    if args.epochs:
        cmd.extend([f"train.epochs={args.epochs}"])
    if args.run_name:
        cmd.extend([f"train.run_name={args.run_name}"])
    return cmd


def main() -> None:
    args = build_parser().parse_args()
    os.environ.setdefault("PYTHONPATH", str(VISION_ROOT / "src"))

    if args.mode == "train":
        cmd = build_train_command(args)
        run_command(cmd)

    elif args.mode == "infer":
        cmd = [args.python, "-m", "vision_model.cli.infer", "dataset=visdrone"]
        if args.dataset_root:
            cmd.extend([f"dataset.val.root={args.dataset_root}"])
        elif args.val_root:
            cmd.extend([f"dataset.val.root={args.val_root}"])
        else:
            cmd.extend(["dataset.val.root=data/VisDrone2019-DET-val"])
        if args.model:
            cmd.extend([f"model.model_path={args.model}"])
        else:
            cmd.extend(["model.model_path=data/models/exp003/best.pt"])
        run_command(cmd)

    elif args.mode == "evaluate":
        cmd = [args.python, "-m", "vision_model.cli.evaluate", "dataset=visdrone"]
        if args.dataset_root:
            cmd.extend([f"dataset.val.root={args.dataset_root}"])
        elif args.val_root:
            cmd.extend([f"dataset.val.root={args.val_root}"])
        else:
            cmd.extend(["dataset.val.root=data/VisDrone2019-DET-val"])
        if args.model:
            cmd.extend([f"model.model_path={args.model}"])
        else:
            cmd.extend(["model.model_path=data/models/exp003/best.pt"])
        run_command(cmd)

    elif args.mode == "geotag":
        cmd = [args.python, "-m", "vision_model.cli.geotag", "dataset=auair"]
        if args.dataset_root:
            cmd.extend([f"dataset.root={args.dataset_root}"])
        if args.telemetry:
            cmd.extend([f"geotag.telemetry_source={args.telemetry}"])
        if args.model:
            cmd.extend([f"model.model_path={args.model}"])
        else:
            cmd.extend(["model.model_path=data/models/exp003/best.pt"])
        if args.camera:
            cmd.extend([f"camera={args.camera}"])
        run_command(cmd)


if __name__ == "__main__":
    main()

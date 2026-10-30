#!/usr/bin/env python3
"""Run isolated S/P2 one-class detection + native-ROI Tiny U-Net experiments."""

import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def build_commands(args):
    python = sys.executable
    output = Path(args.output)
    det = Path(args.det_dataset)
    tracking = []
    if args.mlflow_uri:
        tracking.extend(["--mlflow-uri", args.mlflow_uri])
    if args.mlflow_experiment:
        tracking.extend(["--mlflow-experiment", args.mlflow_experiment])
    commands = []
    if args.stage in ("all", "det"):
        commands.append([python, "-m", "detrseg", "setup-detector", "--model-size", "s"])
        commands.append(
            [
                python,
                "-m",
                "detrseg",
                "train-detector",
                "--annotations",
                str(det / "annotations/instances_default.json"),
                "--images",
                str(det),
                "--split-report",
                str(det / "split_report.json"),
                "--output",
                str(output / "detector"),
                "--model-size",
                "s",
                "--p2",
                "--single-class",
                "--class-name",
                "ball",
                "--steps",
                str(args.det_steps),
                "--batch-size",
                str(args.det_batch_size),
                "--size",
                "640",
                "--lr",
                "0.0002",
                "--val-every",
                "250",
                "--budget-seconds",
                "7200",
                "--device",
                args.device,
                *tracking,
            ]
        )
    if args.stage in ("all", "seg"):
        commands.append(
            [
                python,
                "scripts/train_seg_coco.py",
                "--dataset",
                args.seg_dataset,
                "--output",
                str(output / "seg"),
                "--architecture",
                "tiny_unet",
                "--width",
                str(args.width),
                "--size",
                str(args.roi_size),
                "--roi-resize-mode",
                "native",
                "--no-pretrained-encoder",
                "--steps",
                str(args.seg_steps),
                "--batch-size",
                str(args.seg_batch_size),
                "--lr",
                "0.001",
                "--val-every",
                "50",
                "--budget-seconds",
                "3600",
                "--device",
                args.device,
                *tracking,
            ]
        )
    return commands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--det-dataset", default="dataset/sub_dataset/det")
    parser.add_argument("--seg-dataset", default="dataset/sub_dataset/seg")
    parser.add_argument(
        "--output", default=f"runs/experiment_s_p2_tiny_{datetime.now():%Y%m%d_%H%M%S}"
    )
    parser.add_argument("--stage", choices=("all", "det", "seg"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--det-steps", type=int, default=3000)
    parser.add_argument("--seg-steps", type=int, default=1000)
    parser.add_argument("--det-batch-size", type=int, default=4)
    parser.add_argument("--seg-batch-size", type=int, default=32)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument(
        "--roi-size", type=int, default=64, help="Native padding canvas; never resizes ROI pixels"
    )
    parser.add_argument("--mlflow-uri", help="MLflow URL; MLFLOW_TRACKING_URI also works")
    parser.add_argument("--mlflow-experiment")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without reading data, creating files or training",
    )
    args = parser.parse_args(argv)
    if min(args.det_steps, args.seg_steps, args.det_batch_size, args.seg_batch_size) < 1:
        parser.error("steps and batch sizes must be positive")
    if args.width < 4 or args.width % 4 or args.roi_size < 16:
        parser.error("width must be a multiple of 4 >=4, ROI size >=16")
    for command in build_commands(args):
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            try:
                subprocess.run(command, cwd=PROJECT, check=True)
            except (OSError, subprocess.CalledProcessError) as exc:
                parser.exit(1, f"Experiment stopped: {exc}\n")


if __name__ == "__main__":
    main()

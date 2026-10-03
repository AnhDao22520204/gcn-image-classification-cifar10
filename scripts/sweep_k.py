from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["patch_gcn", "cnn_gcn"], default="cnn_gcn")
    p.add_argument("--values", nargs="+", type=int, default=[2, 4, 6])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-subset", type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    for k in args.values:
        out = Path("results") / "runs" / f"{args.model}_k{k}"
        cmd = [
            sys.executable, "train.py",
            "--model", args.model,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--train-subset", str(args.train_subset),
            "--k", str(k),
            "--output-dir", str(out),
        ]
        print("\nRunning:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

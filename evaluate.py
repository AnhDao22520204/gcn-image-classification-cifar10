from __future__ import annotations

import argparse

import torch
from torch import nn

from src.data import CIFAR10_CLASSES, build_cifar10_loaders
from src.engine import evaluate
from src.metrics import classification_metrics
from src.models import PatchGCN, CNNGCN
from src.plotting import plot_confusion_matrix
from src.utils import get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["patch_gcn", "cnn_gcn"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()

    _, test_loader = build_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_subset=1,
        num_workers=args.num_workers,
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    saved = ckpt.get("args", {})

    if args.model == "patch_gcn":
        model = PatchGCN(
            patch_size=saved.get("patch_size", 8),
            embedding_dim=saved.get("embedding_dim", 128),
            k=saved.get("k", 4),
            dropout=saved.get("dropout", 0.4),
        )
    else:
        model = CNNGCN(
            k=saved.get("k", 4),
            dropout=saved.get("dropout", 0.4),
            pretrained_backbone=False,
        )

    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    result = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    metrics = classification_metrics(result.y_true, result.y_pred)

    print(f"Loss:      {result.loss:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision_macro']:.4f}")
    print(f"Recall:    {metrics['recall_macro']:.4f}")
    print(f"F1:        {metrics['f1_macro']:.4f}")

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        CIFAR10_CLASSES,
        "confusion_matrix_eval.png",
    )


if __name__ == "__main__":
    main()

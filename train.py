from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import nn

from src.data import CIFAR10_CLASSES, build_cifar10_loaders
from src.engine import train_one_epoch, evaluate
from src.metrics import classification_metrics
from src.models import PatchGCN, CNNGCN
from src.plotting import plot_history, plot_confusion_matrix
from src.utils import ensure_dir, get_device, save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["patch_gcn", "cnn_gcn"], required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--train-subset", type=int, default=1000)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--pretrained-backbone", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_model(args):
    if args.model == "patch_gcn":
        return PatchGCN(
            patch_size=args.patch_size,
            embedding_dim=args.embedding_dim,
            k=args.k,
            dropout=args.dropout,
        )
    return CNNGCN(
        k=args.k,
        dropout=args.dropout,
        pretrained_backbone=args.pretrained_backbone,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    out_dir = ensure_dir(args.output_dir or Path("results") / "runs" / args.model)

    train_loader, test_loader = build_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_subset=args.train_subset,
        num_workers=args.num_workers,
    )

    model = make_model(args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    best_accuracy = -1.0
    best_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device)
        te = evaluate(model, test_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": tr.loss,
            "train_accuracy": tr.accuracy,
            "test_loss": te.loss,
            "test_accuracy": te.accuracy,
            "test_precision_macro": te.precision_macro,
            "test_recall_macro": te.recall_macro,
            "test_f1_macro": te.f1_macro,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={tr.loss:.4f} acc={tr.accuracy:.4f} | "
            f"test loss={te.loss:.4f} acc={te.accuracy:.4f}"
        )

        if te.accuracy > best_accuracy:
            best_accuracy = te.accuracy
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "test_accuracy": te.accuracy,
                },
                best_path,
            )

    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    plot_history(history, out_dir)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    final = evaluate(model, test_loader, criterion, device)
    metrics = classification_metrics(final.y_true, final.y_pred)
    metrics["test_loss"] = final.loss
    metrics["best_epoch"] = checkpoint["epoch"]
    metrics["model"] = args.model
    metrics["k"] = args.k
    save_json(metrics, out_dir / "metrics.json")

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        CIFAR10_CLASSES,
        out_dir / "confusion_matrix.png",
    )

    print(f"Best checkpoint: {best_path}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision_macro']:.4f}")
    print(f"Recall:    {metrics['recall_macro']:.4f}")
    print(f"F1:        {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()

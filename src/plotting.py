from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_history(history: list[dict], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = [r["epoch"] for r in history]

    fig = plt.figure()
    plt.plot(epochs, [r["train_loss"] for r in history], label="Training Loss")
    plt.plot(epochs, [r["test_loss"] for r in history], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Test Loss")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=180)
    plt.close(fig)

    fig = plt.figure()
    plt.plot(epochs, [r["train_accuracy"] for r in history], label="Training Accuracy")
    plt.plot(epochs, [r["test_accuracy"] for r in history], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy vs Test Accuracy")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "accuracy_curve.png", dpi=180)
    plt.close(fig)


def plot_confusion_matrix(matrix, class_names, out_path) -> None:
    matrix = np.asarray(matrix)
    fig = plt.figure(figsize=(9, 8))
    ax = plt.gca()
    im = ax.imshow(matrix)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix on CIFAR-10 Test Set",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

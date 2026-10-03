from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from tqdm import tqdm

from src.metrics import classification_metrics


@dataclass
class EpochResult:
    loss: float
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    y_true: list[int]
    y_pred: list[int]


def _run_epoch(model, loader, criterion, device, optimizer=None) -> EpochResult:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0
    all_true, all_pred = [], []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = logits.argmax(dim=1)
            all_true.extend(labels.detach().cpu().tolist())
            all_pred.extend(preds.detach().cpu().tolist())

    m = classification_metrics(all_true, all_pred)
    return EpochResult(
        loss=total_loss / max(total_samples, 1),
        accuracy=m["accuracy"],
        precision_macro=m["precision_macro"],
        recall_macro=m["recall_macro"],
        f1_macro=m["f1_macro"],
        y_true=all_true,
        y_pred=all_pred,
    )


def train_one_epoch(model, loader, criterion, optimizer, device):
    return _run_epoch(model, loader, criterion, device, optimizer)


def evaluate(model, loader, criterion, device):
    return _run_epoch(model, loader, criterion, device, optimizer=None)

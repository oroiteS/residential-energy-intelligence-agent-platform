"""TCN 训练与评估公共逻辑。"""

from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from classification.TCN.config import DataConfig, ExperimentConfig, ModelConfig
from classification.TCN.constants import LABELS
from classification.TCN.dataset import ClassificationDataset, load_classification_samples, split_samples
from classification.TCN.model import TCNClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_macro_f1(predictions: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    f1_scores: list[float] = []
    for class_index in range(num_classes):
        true_positive = np.logical_and(predictions == class_index, targets == class_index).sum()
        false_positive = np.logical_and(predictions == class_index, targets != class_index).sum()
        false_negative = np.logical_and(predictions != class_index, targets == class_index).sum()

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return float(sum(f1_scores) / num_classes)


def build_model(model_config: ModelConfig) -> TCNClassifier:
    return TCNClassifier(
        input_channels=model_config.input_channels,
        num_classes=model_config.num_classes,
        channel_sizes=model_config.channel_sizes,
        kernel_size=model_config.kernel_size,
        dropout=model_config.dropout,
    )


def create_split_datasets(
    data_config: DataConfig,
    seed: int,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[ClassificationDataset, ClassificationDataset, ClassificationDataset]:
    samples = load_classification_samples(data_config.data_path)
    train_samples, val_samples, test_samples = split_samples(
        samples=samples,
        train_ratio=data_config.train_ratio,
        val_ratio=data_config.val_ratio,
        seed=seed,
    )

    train_dataset = ClassificationDataset(train_samples, mean=mean, std=std)
    reference_mean = train_dataset.mean if mean is None else mean
    reference_std = train_dataset.std if std is None else std
    val_dataset = ClassificationDataset(val_samples, mean=reference_mean, std=reference_std)
    test_dataset = ClassificationDataset(test_samples, mean=reference_mean, std=reference_std)
    return train_dataset, val_dataset, test_dataset


def create_data_loaders(
    train_dataset: ClassificationDataset,
    val_dataset: ClassificationDataset,
    test_dataset: ClassificationDataset,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def create_eval_loader(dataset: ClassificationDataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            total_loss += float(loss.item()) * len(labels)
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(labels.cpu().numpy())

    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(targets)
    return {
        "loss": total_loss / len(y_true),
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": compute_macro_f1(y_pred, y_true, num_classes=len(LABELS)),
    }


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for features, labels in data_loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * len(labels)
        predictions.append(logits.detach().argmax(dim=1).cpu().numpy())
        targets.append(labels.cpu().numpy())

    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(targets)
    return {
        "loss": total_loss / len(y_true),
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": compute_macro_f1(y_pred, y_true, num_classes=len(LABELS)),
    }


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    train_dataset: ClassificationDataset,
    experiment_config: ExperimentConfig,
    metrics: dict[str, float],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": train_dataset.mean.tolist(),
            "feature_std": train_dataset.std.tolist(),
            "labels": LABELS,
            "config": experiment_config.to_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )


def save_json_summary(output_path: Path, payload: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, object]:
    return torch.load(checkpoint_path, map_location=device)


def checkpoint_to_normalization(checkpoint: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    return mean, std

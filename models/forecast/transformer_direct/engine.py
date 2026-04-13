"""Patch-based direct Transformer 训练与评估公共逻辑。"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    current_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(current_dir))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from config import DataConfig, ModelConfig, TrainConfig
    from model import PatchDirectTransformerForecaster
else:
    from .config import DataConfig, ModelConfig, TrainConfig
    from .model import PatchDirectTransformerForecaster
from forecast.LSTM.dataset import ForecastDataset, ForecastNormalizationStats
from forecast.LSTM.engine import (
    checkpoint_to_normalization,
    count_parameters,
    create_data_loaders,
    create_eval_loader,
    create_split_datasets,
    load_checkpoint,
    normalization_to_checkpoint_payload,
    save_checkpoint,
    save_json_summary,
    set_seed,
)


def compute_regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    errors = predictions - targets
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    smape_denominator = np.clip(
        np.abs(predictions) + np.abs(targets),
        1e-6,
        None,
    )
    smape = float(np.mean((2.0 * np.abs(errors)) / smape_denominator) * 100.0)
    wape_denominator = float(np.clip(np.sum(np.abs(targets)), 1e-6, None))
    wape = float(np.sum(np.abs(errors)) / wape_denominator * 100.0)
    return {"mae": mae, "rmse": rmse, "smape": smape, "wape": wape}


def build_model(model_config: ModelConfig) -> PatchDirectTransformerForecaster:
    return PatchDirectTransformerForecaster(
        input_size=model_config.input_size,
        d_model=model_config.d_model,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        ffn_dim=model_config.ffn_dim,
        dropout=model_config.dropout,
        target_length=model_config.target_length,
        patch_length=model_config.patch_length,
        patch_stride=model_config.patch_stride,
    )


def build_criterion(train_config: TrainConfig) -> nn.Module:
    loss_name = train_config.loss_name.lower()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "huber":
        return nn.HuberLoss(delta=train_config.huber_delta)
    raise ValueError(f"不支持的损失函数: {train_config.loss_name}")


def describe_loss(train_config: TrainConfig) -> str:
    loss_name = train_config.loss_name.lower()
    if loss_name == "huber":
        return f"HuberLoss(delta={train_config.huber_delta})"
    if loss_name == "mse":
        return "MSELoss"
    raise ValueError(f"不支持的损失函数: {train_config.loss_name}")


def _denormalize(
    tensor: np.ndarray,
    denorm_mean: np.ndarray,
    denorm_std: np.ndarray,
) -> np.ndarray:
    return tensor * denorm_std + denorm_mean


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    stage_label: str = "验证",
    log_interval: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    denorm_means: list[np.ndarray] = []
    denorm_stds: list[np.ndarray] = []
    total_batches = len(data_loader)

    with torch.no_grad():
        progress = tqdm(
            data_loader,
            total=total_batches,
            desc=stage_label,
            leave=False,
            dynamic_ncols=True,
            mininterval=1.0,
        )
        for features, labels, batch_denorm_mean, batch_denorm_std in progress:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += float(loss.item()) * len(labels)
            total_samples += len(labels)
            predictions.append(outputs.cpu().numpy())
            targets.append(labels.cpu().numpy())
            denorm_means.append(batch_denorm_mean.cpu().numpy())
            denorm_stds.append(batch_denorm_std.cpu().numpy())
            average_loss = total_loss / max(1, total_samples)
            progress.set_postfix(avg_loss=f"{average_loss:.4f}")

    y_pred = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(targets, axis=0)
    mean_array = np.concatenate(denorm_means, axis=0)
    std_array = np.concatenate(denorm_stds, axis=0)
    pred_denorm = _denormalize(y_pred, mean_array, std_array)
    true_denorm = _denormalize(y_true, mean_array, std_array)
    metrics = compute_regression_metrics(pred_denorm, true_denorm)
    metrics["loss"] = total_loss / len(y_true)
    return metrics


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float | None = None,
    stage_label: str = "训练",
    log_interval: int | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    denorm_means: list[np.ndarray] = []
    denorm_stds: list[np.ndarray] = []
    total_batches = len(data_loader)
    progress = tqdm(
        data_loader,
        total=total_batches,
        desc=stage_label,
        leave=False,
        dynamic_ncols=True,
        mininterval=1.0,
    )

    for features, labels, batch_denorm_mean, batch_denorm_std in progress:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm,
            )
        optimizer.step()

        total_loss += float(loss.item()) * len(labels)
        total_samples += len(labels)
        predictions.append(outputs.detach().cpu().numpy())
        targets.append(labels.cpu().numpy())
        denorm_means.append(batch_denorm_mean.cpu().numpy())
        denorm_stds.append(batch_denorm_std.cpu().numpy())
        average_loss = total_loss / max(1, total_samples)
        progress.set_postfix(avg_loss=f"{average_loss:.4f}")

    y_pred = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(targets, axis=0)
    mean_array = np.concatenate(denorm_means, axis=0)
    std_array = np.concatenate(denorm_stds, axis=0)
    pred_denorm = _denormalize(y_pred, mean_array, std_array)
    true_denorm = _denormalize(y_true, mean_array, std_array)
    metrics = compute_regression_metrics(pred_denorm, true_denorm)
    metrics["loss"] = total_loss / len(y_true)
    return metrics


__all__ = [
    "ForecastDataset",
    "ForecastNormalizationStats",
    "DataConfig",
    "build_model",
    "build_criterion",
    "describe_loss",
    "evaluate",
    "train_one_epoch",
    "set_seed",
    "count_parameters",
    "create_split_datasets",
    "create_data_loaders",
    "create_eval_loader",
    "save_checkpoint",
    "save_json_summary",
    "load_checkpoint",
    "checkpoint_to_normalization",
    "normalization_to_checkpoint_payload",
]

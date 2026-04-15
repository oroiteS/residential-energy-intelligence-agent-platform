"""Patch-based direct Transformer 训练与评估公共逻辑。"""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
from typing import Any

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
    from dataset import (
        DEFAULT_NORMALIZATION_MODE,
        LEGACY_NORMALIZATION_MODE,
        ForecastDataset,
        ForecastNormalizationStats,
        load_forecast_samples,
        split_samples,
    )
    from model import PatchDirectTransformerForecaster
else:
    from .config import DataConfig, ModelConfig, TrainConfig
    from .dataset import (
        DEFAULT_NORMALIZATION_MODE,
        LEGACY_NORMALIZATION_MODE,
        ForecastDataset,
        ForecastNormalizationStats,
        load_forecast_samples,
        split_samples,
    )
    from .model import PatchDirectTransformerForecaster


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def create_split_datasets(
    data_config: DataConfig,
    seed: int,
    normalization: ForecastNormalizationStats | None = None,
) -> tuple[ForecastDataset, ForecastDataset, ForecastDataset]:
    samples = load_forecast_samples(
        data_config.data_path,
        feature_names=data_config.feature_names,
    )
    train_samples, val_samples, test_samples = split_samples(
        samples=samples,
        split_mode=data_config.split_mode,
        train_ratio=data_config.train_ratio,
        val_ratio=data_config.val_ratio,
        seed=seed,
    )

    train_dataset = ForecastDataset(
        train_samples,
        data_path=data_config.data_path,
        feature_names=data_config.feature_names,
        normalization=normalization,
        aggregate_mode=data_config.aggregate_normalization,
        aggregate_eps=data_config.aggregate_norm_eps,
    )
    reference_normalization = (
        train_dataset.normalization if normalization is None else normalization
    )
    val_dataset = ForecastDataset(
        val_samples,
        data_path=data_config.data_path,
        feature_names=data_config.feature_names,
        normalization=reference_normalization,
        aggregate_mode=data_config.aggregate_normalization,
        aggregate_eps=data_config.aggregate_norm_eps,
    )
    test_dataset = ForecastDataset(
        test_samples,
        data_path=data_config.data_path,
        feature_names=data_config.feature_names,
        normalization=reference_normalization,
        aggregate_mode=data_config.aggregate_normalization,
        aggregate_eps=data_config.aggregate_norm_eps,
    )
    return train_dataset, val_dataset, test_dataset


def create_data_loaders(
    train_dataset: ForecastDataset,
    val_dataset: ForecastDataset,
    test_dataset: ForecastDataset,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    loader_kwargs = {"batch_size": batch_size, "num_workers": 0}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def create_eval_loader(dataset: ForecastDataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    train_dataset: ForecastDataset,
    experiment_config,
    metrics: dict[str, float],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "normalization": normalization_to_checkpoint_payload(
                train_dataset.normalization
            ),
            "config": experiment_config.to_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )


def save_json_summary(output_path: Path, payload: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, object]:
    return torch.load(checkpoint_path, map_location=device)


def checkpoint_to_normalization(
    checkpoint: dict[str, Any],
) -> ForecastNormalizationStats:
    raw_normalization = checkpoint.get("normalization")
    if isinstance(raw_normalization, dict):
        aggregate_mode = str(
            raw_normalization.get("aggregate_mode", DEFAULT_NORMALIZATION_MODE)
        )
        aggregate_eps = float(raw_normalization.get("aggregate_eps", 1e-6))
        if aggregate_mode == DEFAULT_NORMALIZATION_MODE:
            return ForecastNormalizationStats(
                aggregate_mode=aggregate_mode,
                aggregate_eps=aggregate_eps,
                auxiliary_mean=np.asarray(
                    raw_normalization["auxiliary_mean"], dtype=np.float32
                ),
                auxiliary_std=np.asarray(
                    raw_normalization["auxiliary_std"], dtype=np.float32
                ),
            )
        if aggregate_mode == LEGACY_NORMALIZATION_MODE:
            return ForecastNormalizationStats(
                aggregate_mode=aggregate_mode,
                aggregate_eps=aggregate_eps,
                feature_mean=np.asarray(
                    raw_normalization["feature_mean"], dtype=np.float32
                ),
                feature_std=np.asarray(
                    raw_normalization["feature_std"], dtype=np.float32
                ),
                target_mean=float(raw_normalization["target_mean"]),
                target_std=float(raw_normalization["target_std"]),
            )
        raise ValueError(f"checkpoint 中存在未知的 aggregate_mode: {aggregate_mode}")

    return ForecastNormalizationStats(
        aggregate_mode=LEGACY_NORMALIZATION_MODE,
        aggregate_eps=1e-6,
        feature_mean=np.asarray(checkpoint["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(checkpoint["feature_std"], dtype=np.float32),
        target_mean=float(checkpoint["target_mean"]),
        target_std=float(checkpoint["target_std"]),
    )


def normalization_to_checkpoint_payload(
    normalization: ForecastNormalizationStats,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "aggregate_mode": normalization.aggregate_mode,
        "aggregate_eps": normalization.aggregate_eps,
    }
    if normalization.aggregate_mode == DEFAULT_NORMALIZATION_MODE:
        if normalization.auxiliary_mean is None or normalization.auxiliary_std is None:
            raise ValueError("input_window 模式缺少 auxiliary_mean / auxiliary_std")
        payload["auxiliary_mean"] = normalization.auxiliary_mean.tolist()
        payload["auxiliary_std"] = normalization.auxiliary_std.tolist()
        return payload

    if normalization.aggregate_mode == LEGACY_NORMALIZATION_MODE:
        if (
            normalization.feature_mean is None
            or normalization.feature_std is None
            or normalization.target_mean is None
            or normalization.target_std is None
        ):
            raise ValueError(
                "global 模式缺少 feature_mean / feature_std / target_mean / target_std"
            )
        payload["feature_mean"] = normalization.feature_mean.tolist()
        payload["feature_std"] = normalization.feature_std.tolist()
        payload["target_mean"] = normalization.target_mean
        payload["target_std"] = normalization.target_std
        return payload

    raise ValueError(f"不支持的 aggregate_mode: {normalization.aggregate_mode}")


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    stage_label: str = "验证",
    log_interval: int | None = None,
) -> dict[str, float]:
    metrics, _, _ = evaluate_with_predictions(
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        device=device,
        stage_label=stage_label,
        log_interval=log_interval,
    )
    return metrics


def evaluate_with_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    stage_label: str = "验证",
    log_interval: int | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
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
    return metrics, pred_denorm, true_denorm


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
    "compute_regression_metrics",
    "describe_loss",
    "evaluate",
    "evaluate_with_predictions",
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

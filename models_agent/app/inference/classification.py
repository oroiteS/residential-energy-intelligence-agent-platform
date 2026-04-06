"""TCN 分类推理实现。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


LABELS = [
    "day_high_night_low",
    "day_low_night_high",
    "all_day_high",
    "all_day_low",
]
SEQUENCE_LENGTH = 96
FEATURE_NAMES = (
    "aggregate",
    "slot_sin",
    "slot_cos",
    "weekday_sin",
    "weekday_cos",
)
INPUT_CHANNELS = len(FEATURE_NAMES)
INDEX_TO_LABEL = {index: label for index, label in enumerate(LABELS)}


@dataclass(slots=True)
class ClassificationModelConfig:
    input_channels: int
    num_classes: int
    channel_sizes: list[int]
    kernel_size: int
    dropout: float


@dataclass(slots=True)
class ClassificationPredictConfig:
    checkpoint_path: Path | None
    batch_size: int


@dataclass(slots=True)
class ClassificationExperimentConfig:
    model: ClassificationModelConfig
    predict: ClassificationPredictConfig
    train_output_dir: Path


def detect_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_path(path_value: str | None, base_dir: Path) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_config(config_path: Path) -> ClassificationExperimentConfig:
    import yaml

    config_path = config_path.resolve()
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base_dir = config_path.parent.parent

    model_raw = raw_config.get("model", {})
    predict_raw = raw_config.get("predict", {})
    train_raw = raw_config.get("train", {})

    model = ClassificationModelConfig(
        input_channels=int(model_raw.get("input_channels", INPUT_CHANNELS)),
        num_classes=int(model_raw.get("num_classes", len(LABELS))),
        channel_sizes=[
            int(value) for value in model_raw.get("channel_sizes", [32, 64, 128])
        ],
        kernel_size=int(model_raw.get("kernel_size", 3)),
        dropout=float(model_raw.get("dropout", 0.2)),
    )
    train_output_dir = _resolve_path(
        str(
            train_raw.get(
                "output_dir",
                "../models_agent/checkpoints/classification/tcn",
            )
        ),
        base_dir,
    )
    checkpoint_raw = predict_raw.get("checkpoint_path")
    predict = ClassificationPredictConfig(
        checkpoint_path=(
            _resolve_path(str(checkpoint_raw), base_dir)
            if checkpoint_raw
            else None
        ),
        batch_size=int(predict_raw.get("batch_size", 128)),
    )
    if model.input_channels != INPUT_CHANNELS:
        raise ValueError(f"model.input_channels 必须等于 {INPUT_CHANNELS}")
    return ClassificationExperimentConfig(
        model=model,
        predict=predict,
        train_output_dir=train_output_dir,
    )


class Chomp1d(nn.Module):
    """裁掉因因果卷积补齐带来的尾部长度。"""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return tensor
        return tensor[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """带残差连接的 TCN 基础模块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.final_relu = nn.ReLU()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        out = self.conv1(tensor)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        residual = tensor if self.downsample is None else self.downsample(tensor)
        return self.final_relu(out + residual)


class TCNClassifier(nn.Module):
    """面向 96x5 日级样本的 TCN 四分类器。"""

    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
        num_classes: int = 4,
        channel_sizes: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if channel_sizes is None:
            channel_sizes = [32, 64, 128]

        layers: list[nn.Module] = []
        in_channels = input_channels
        for block_index, out_channels in enumerate(channel_sizes):
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=2**block_index,
                    dropout=dropout,
                )
            )
            in_channels = out_channels

        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channel_sizes[-1], channel_sizes[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channel_sizes[-1] // 2, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                f"输入维度应为 [batch, seq_len, channels]，实际为 {tuple(features.shape)}"
            )
        temporal_features = features.transpose(1, 2)
        encoded = self.network(temporal_features)
        pooled = self.pool(encoded)
        return self.classifier(pooled)


def build_model(model_config: ClassificationModelConfig) -> TCNClassifier:
    return TCNClassifier(
        input_channels=model_config.input_channels,
        num_classes=model_config.num_classes,
        channel_sizes=model_config.channel_sizes,
        kernel_size=model_config.kernel_size,
        dropout=model_config.dropout,
    )


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location=device)


def _resolve_model_config(
    fallback_model_config: ClassificationModelConfig,
    checkpoint: dict[str, Any],
) -> ClassificationModelConfig:
    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, dict):
        return fallback_model_config

    model_raw = raw_config.get("model")
    if not isinstance(model_raw, dict):
        return fallback_model_config

    resolved_model_config = ClassificationModelConfig(
        input_channels=int(
            model_raw.get("input_channels", fallback_model_config.input_channels)
        ),
        num_classes=int(
            model_raw.get("num_classes", fallback_model_config.num_classes)
        ),
        channel_sizes=[
            int(value)
            for value in model_raw.get(
                "channel_sizes",
                fallback_model_config.channel_sizes,
            )
        ],
        kernel_size=int(model_raw.get("kernel_size", fallback_model_config.kernel_size)),
        dropout=float(model_raw.get("dropout", fallback_model_config.dropout)),
    )
    if resolved_model_config.input_channels != INPUT_CHANNELS:
        raise ValueError(f"checkpoint 中的 input_channels 必须等于 {INPUT_CHANNELS}")
    return resolved_model_config


def checkpoint_to_normalization(
    checkpoint: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    return mean, std


def _normalize_features(
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    safe_std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return (features.astype(np.float32) - mean.astype(np.float32)) / safe_std


def _build_temporal_feature_sequences(date_value: str) -> dict[str, list[float]]:
    base_date = pd.to_datetime(date_value)
    if pd.isna(base_date):
        raise ValueError("无法根据 timestamp 自动生成时间特征，请提供合法时间")

    timestamps = pd.date_range(
        start=base_date.normalize(),
        periods=SEQUENCE_LENGTH,
        freq="15min",
    )
    slot_index = timestamps.hour * 4 + timestamps.minute // 15
    weekday_index = timestamps.dayofweek
    slot_angle = 2.0 * np.pi * slot_index.to_numpy(dtype=np.float32) / 96.0
    weekday_angle = 2.0 * np.pi * weekday_index.to_numpy(dtype=np.float32) / 7.0
    return {
        "slot_sin": np.sin(slot_angle).astype(np.float32).tolist(),
        "slot_cos": np.cos(slot_angle).astype(np.float32).tolist(),
        "weekday_sin": np.sin(weekday_angle).astype(np.float32).tolist(),
        "weekday_cos": np.cos(weekday_angle).astype(np.float32).tolist(),
    }


def _build_feature_values(sample: dict[str, Any]) -> dict[str, list[float]]:
    aggregate = sample.get("aggregate")
    if not isinstance(aggregate, list) or len(aggregate) != SEQUENCE_LENGTH:
        raise ValueError("aggregate 输入序列长度必须为 96")

    date_value = str(sample.get("date", "")).strip()
    if not date_value:
        date_value = str(sample.get("timestamp", "")).strip()
    temporal_sequences = _build_temporal_feature_sequences(date_value)

    return {
        "aggregate": [float(value) for value in aggregate],
        "slot_sin": temporal_sequences["slot_sin"],
        "slot_cos": temporal_sequences["slot_cos"],
        "weekday_sin": temporal_sequences["weekday_sin"],
        "weekday_cos": temporal_sequences["weekday_cos"],
    }


def predict_single_sample(sample: dict[str, Any], config_path: Path) -> dict[str, Any]:
    experiment_config = load_config(config_path)
    checkpoint_path = experiment_config.predict.checkpoint_path or (
        experiment_config.train_output_dir / "best_model.pt"
    )
    device_name = detect_device()
    device = torch.device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model_config = _resolve_model_config(experiment_config.model, checkpoint)
    feature_mean, feature_std = checkpoint_to_normalization(checkpoint)

    feature_values = _build_feature_values(sample)
    features = np.stack(
        [
            np.asarray(feature_values["aggregate"], dtype=np.float32),
            np.asarray(feature_values["slot_sin"], dtype=np.float32),
            np.asarray(feature_values["slot_cos"], dtype=np.float32),
            np.asarray(feature_values["weekday_sin"], dtype=np.float32),
            np.asarray(feature_values["weekday_cos"], dtype=np.float32),
        ],
        axis=-1,
    )
    normalized = _normalize_features(
        features[np.newaxis, ...],
        feature_mean,
        feature_std,
    )

    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(normalized).to(device))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

    label_index = int(probabilities.argmax())
    return {
        "predicted_label": INDEX_TO_LABEL[label_index],
        "confidence": float(probabilities[label_index]),
        "prob_day_high_night_low": float(probabilities[0]),
        "prob_day_low_night_high": float(probabilities[1]),
        "prob_all_day_high": float(probabilities[2]),
        "prob_all_day_low": float(probabilities[3]),
        "runtime_device": device_name,
        "runtime_loss": "CrossEntropyLoss",
    }

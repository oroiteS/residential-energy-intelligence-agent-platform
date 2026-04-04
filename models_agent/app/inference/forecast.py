"""LSTM 预测最小推理实现。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


INPUT_LENGTH = 288
TARGET_LENGTH = 96
DEFAULT_NORMALIZATION_MODE = "input_window"
LEGACY_NORMALIZATION_MODE = "global"
ALL_FEATURE_NAMES = (
    "aggregate",
    "active_appliance_count",
    "burst_event_count",
)


@dataclass(slots=True)
class ForecastModelConfig:
    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    target_length: int


@dataclass(slots=True)
class ForecastPredictConfig:
    checkpoint_path: Path | None
    batch_size: int


@dataclass(slots=True)
class ForecastExperimentConfig:
    model: ForecastModelConfig
    predict: ForecastPredictConfig
    train_output_dir: Path
    feature_names: list[str]


@dataclass(slots=True)
class ForecastNormalizationStats:
    aggregate_mode: str
    aggregate_eps: float
    auxiliary_mean: np.ndarray | None = None
    auxiliary_std: np.ndarray | None = None
    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None
    target_mean: float | None = None
    target_std: float | None = None


def detect_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_path(path_value: str | None, base_dir: Path) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_config(config_path: Path) -> ForecastExperimentConfig:
    import yaml

    config_path = config_path.resolve()
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.parent.parent

    model = ForecastModelConfig(
        input_size=int(
            raw_config["model"].get(
                "input_size",
                len(raw_config["data"].get("feature_names", ["aggregate"])),
            )
        ),
        hidden_size=int(raw_config["model"].get("hidden_size", 256)),
        num_layers=int(raw_config["model"].get("num_layers", 3)),
        dropout=float(raw_config["model"].get("dropout", 0.2)),
        target_length=int(raw_config["model"].get("target_length", 96)),
    )
    feature_names = [
        str(item)
        for item in raw_config.get("data", {}).get("feature_names", ["aggregate"])
    ]
    train_output_dir = _resolve_path(raw_config["train"]["output_dir"], base_dir)
    checkpoint_raw = raw_config.get("predict", {}).get("checkpoint_path")
    predict = ForecastPredictConfig(
        checkpoint_path=_resolve_path(checkpoint_raw, base_dir) if checkpoint_raw else None,
        batch_size=int(raw_config.get("predict", {}).get("batch_size", 128)),
    )
    if not feature_names:
        raise ValueError("feature_names 不能为空")
    if feature_names[0] != "aggregate":
        raise ValueError("feature_names 第一个特征必须是 aggregate")
    invalid_feature_names = set(feature_names).difference(ALL_FEATURE_NAMES)
    if invalid_feature_names:
        raise ValueError(f"存在不支持的 feature_names: {sorted(invalid_feature_names)}")
    if model.input_size != len(feature_names):
        raise ValueError("model.input_size 必须等于 feature_names 长度")

    return ForecastExperimentConfig(
        model=model,
        predict=predict,
        train_output_dir=train_output_dir,
        feature_names=feature_names,
    )


class Seq2SeqLSTMForecaster(nn.Module):
    """基于编码器-解码器结构的多步预测模型。"""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        target_length: int = TARGET_LENGTH,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.target_length = target_length
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        decoder_targets: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"输入维度应为 [batch, seq_len, channels]，实际为 {tuple(features.shape)}")
        if features.size(-1) != self.input_size:
            raise ValueError(f"输入通道数应为 {self.input_size}，实际为 {features.size(-1)}")

        batch_size = features.size(0)
        _, encoder_state = self.encoder(features)

        decoder_input = torch.zeros(
            batch_size,
            1,
            1,
            device=features.device,
            dtype=features.dtype,
        )
        hidden, cell = encoder_state

        predictions: list[torch.Tensor] = []
        for step_index in range(self.target_length):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            current_prediction = self.output_layer(decoder_output[:, -1, :])
            predictions.append(current_prediction)
            decoder_input = current_prediction.unsqueeze(1)

        return torch.cat(predictions, dim=1)


def build_model(model_config: ForecastModelConfig) -> Seq2SeqLSTMForecaster:
    return Seq2SeqLSTMForecaster(
        input_size=model_config.input_size,
        hidden_size=model_config.hidden_size,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
        target_length=model_config.target_length,
    )


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location=device)


def checkpoint_to_normalization(checkpoint: dict[str, Any]) -> ForecastNormalizationStats:
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


def _normalize_features(
    features: np.ndarray,
    normalization: ForecastNormalizationStats,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_features = features.astype(np.float32)

    if normalization.aggregate_mode == DEFAULT_NORMALIZATION_MODE:
        if normalization.auxiliary_mean is None or normalization.auxiliary_std is None:
            raise ValueError("input_window 模式缺少 auxiliary_mean / auxiliary_std")

        aggregate = raw_features[:, :, 0]
        aggregate_mean = aggregate.mean(axis=1, keepdims=True).astype(np.float32)
        aggregate_std = aggregate.std(axis=1, keepdims=True).astype(np.float32)
        safe_aggregate_std = np.where(
            aggregate_std < normalization.aggregate_eps,
            1.0,
            aggregate_std,
        ).astype(np.float32)
        normalized_aggregate = (
            (aggregate - aggregate_mean) / safe_aggregate_std
        )[:, :, np.newaxis].astype(np.float32)
        raw_auxiliary = raw_features[:, :, 1:]
        if raw_auxiliary.shape[-1] == 0:
            normalized = normalized_aggregate
        else:
            normalized_auxiliary = (
                raw_auxiliary - normalization.auxiliary_mean.astype(np.float32)
            ) / normalization.auxiliary_std.astype(np.float32)
            normalized = np.concatenate(
                [normalized_aggregate, normalized_auxiliary.astype(np.float32)],
                axis=-1,
            )
        return normalized, aggregate_mean, safe_aggregate_std

    if normalization.aggregate_mode == LEGACY_NORMALIZATION_MODE:
        if normalization.feature_mean is None or normalization.feature_std is None:
            raise ValueError("global 模式缺少 feature_mean / feature_std")
        if normalization.target_mean is None or normalization.target_std is None:
            raise ValueError("global 模式缺少 target_mean / target_std")
        normalized = (
            raw_features - normalization.feature_mean.astype(np.float32)
        ) / normalization.feature_std.astype(np.float32)
        denorm_mean = np.full(
            (len(raw_features), 1),
            float(normalization.target_mean),
            dtype=np.float32,
        )
        denorm_std = np.full(
            (len(raw_features), 1),
            float(normalization.target_std),
            dtype=np.float32,
        )
        return normalized.astype(np.float32), denorm_mean, denorm_std

    raise ValueError(f"不支持的 aggregate_mode: {normalization.aggregate_mode}")


def predict_single_sample(
    aggregate: list[float],
    active_appliance_count: list[int],
    burst_event_count: list[int],
    config_path: Path,
) -> list[float]:
    if not (
        len(aggregate) == len(active_appliance_count) == len(burst_event_count) == INPUT_LENGTH
    ):
        raise ValueError("预测输入三条序列长度都必须为 288")

    experiment_config = load_config(config_path)
    checkpoint_path = experiment_config.predict.checkpoint_path or (
        experiment_config.train_output_dir / "best_model.pt"
    )
    device_name = detect_device()
    device = torch.device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, device)
    normalization = checkpoint_to_normalization(checkpoint)

    feature_arrays: list[np.ndarray] = []
    for feature_name in experiment_config.feature_names:
        if feature_name == "aggregate":
            feature_arrays.append(np.asarray(aggregate, dtype=np.float32))
        elif feature_name == "active_appliance_count":
            feature_arrays.append(
                np.asarray(active_appliance_count, dtype=np.float32)
            )
        elif feature_name == "burst_event_count":
            feature_arrays.append(
                np.asarray(burst_event_count, dtype=np.float32)
            )
        else:
            raise ValueError(f"不支持的 feature_name: {feature_name}")
    features = np.stack(feature_arrays, axis=-1)
    normalized, denorm_mean, denorm_std = _normalize_features(
        features[np.newaxis, ...],
        normalization,
    )

    model = build_model(experiment_config.model).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        outputs = model(torch.from_numpy(normalized).to(device), teacher_forcing_ratio=0.0)
        predictions = outputs.cpu().numpy()[0]

    return [
        float(value * denorm_std[0, 0] + denorm_mean[0, 0])
        for value in predictions
    ]

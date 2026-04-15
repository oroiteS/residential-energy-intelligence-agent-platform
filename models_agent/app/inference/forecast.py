"""预测推理实现，支持 LSTM 与两个独立 Transformer。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


STEPS_PER_DAY = 96
DEFAULT_LSTM_INPUT_LENGTH = 7 * STEPS_PER_DAY
DEFAULT_TRANSFORMER_INPUT_LENGTH = 7 * STEPS_PER_DAY
TARGET_LENGTH = STEPS_PER_DAY
DEFAULT_NORMALIZATION_MODE = "input_window"
LEGACY_NORMALIZATION_MODE = "global"
SUPPORTED_FORECAST_MODEL_TYPES = (
    "lstm",
    "transformer_encoder_direct",
    "transformer_encdec_direct",
)
MODEL_TYPE_ALIASES = {
    "transformer": "transformer_encoder_direct",
}
ALL_FEATURE_NAMES = (
    "aggregate",
    "active_appliance_count",
    "burst_event_count",
    "slot_sin",
    "slot_cos",
    "weekday_sin",
    "weekday_cos",
)


@dataclass(slots=True)
class ForecastDataConfig:
    feature_names: list[str]
    aggregate_normalization: str
    aggregate_norm_eps: float


@dataclass(slots=True)
class ForecastPredictConfig:
    checkpoint_path: Path | None
    batch_size: int


@dataclass(slots=True)
class LSTMModelConfig:
    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    target_length: int
    input_length: int


@dataclass(slots=True)
class TransformerModelConfig:
    input_size: int
    d_model: int
    num_layers: int
    num_heads: int
    ffn_dim: int
    dropout: float
    target_length: int
    patch_length: int
    patch_stride: int
    input_length: int


@dataclass(slots=True)
class LSTMRuntimeConfig:
    model: LSTMModelConfig
    predict: ForecastPredictConfig
    train_output_dir: Path


@dataclass(slots=True)
class TransformerRuntimeConfig:
    model: TransformerModelConfig
    predict: ForecastPredictConfig
    train_output_dir: Path


@dataclass(slots=True)
class ForecastExperimentConfig:
    default_model_type: str
    data: ForecastDataConfig
    lstm: LSTMRuntimeConfig
    transformer_encoder_direct: TransformerRuntimeConfig
    transformer_encdec_direct: TransformerRuntimeConfig


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
    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def _normalize_model_type(model_type: str) -> str:
    normalized_model_type = model_type.strip().lower()
    return MODEL_TYPE_ALIASES.get(normalized_model_type, normalized_model_type)


def _resolve_path(path_value: str | None, base_dir: Path) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_predict_config(
    raw_section: dict[str, Any],
    base_dir: Path,
) -> ForecastPredictConfig:
    checkpoint_raw = raw_section.get("checkpoint_path")
    return ForecastPredictConfig(
        checkpoint_path=(
            _resolve_path(str(checkpoint_raw), base_dir) if checkpoint_raw else None
        ),
        batch_size=int(raw_section.get("batch_size", 128)),
    )


def _build_lstm_runtime(
    raw_section: dict[str, Any],
    base_dir: Path,
    feature_count: int,
) -> LSTMRuntimeConfig:
    model_raw = raw_section.get("model", {})
    predict_raw = raw_section.get("predict", {})
    train_output_dir = _resolve_path(
        str(
            raw_section.get(
                "train_output_dir",
                "../models_agent/checkpoints/forecast/lstm",
            )
        ),
        base_dir,
    )
    return LSTMRuntimeConfig(
        model=LSTMModelConfig(
            input_size=int(model_raw.get("input_size", feature_count)),
            hidden_size=int(model_raw.get("hidden_size", 256)),
            num_layers=int(model_raw.get("num_layers", 3)),
            dropout=float(model_raw.get("dropout", 0.2)),
            target_length=int(model_raw.get("target_length", TARGET_LENGTH)),
            input_length=int(
                model_raw.get("input_length", DEFAULT_LSTM_INPUT_LENGTH)
            ),
        ),
        predict=_load_predict_config(predict_raw, base_dir),
        train_output_dir=train_output_dir,
    )


def _build_transformer_runtime(
    raw_section: dict[str, Any],
    base_dir: Path,
    feature_count: int,
    default_output_dir: str,
) -> TransformerRuntimeConfig:
    model_raw = raw_section.get("model", {})
    predict_raw = raw_section.get("predict", {})
    train_output_dir = _resolve_path(
        str(raw_section.get("train_output_dir", default_output_dir)),
        base_dir,
    )
    return TransformerRuntimeConfig(
        model=TransformerModelConfig(
            input_size=int(model_raw.get("input_size", feature_count)),
            d_model=int(model_raw.get("d_model", 256)),
            num_layers=int(model_raw.get("num_layers", 3)),
            num_heads=int(model_raw.get("num_heads", 8)),
            ffn_dim=int(model_raw.get("ffn_dim", 512)),
            dropout=float(model_raw.get("dropout", 0.2)),
            target_length=int(model_raw.get("target_length", TARGET_LENGTH)),
            patch_length=int(model_raw.get("patch_length", 16)),
            patch_stride=int(model_raw.get("patch_stride", 8)),
            input_length=int(
                model_raw.get("input_length", DEFAULT_TRANSFORMER_INPUT_LENGTH)
            ),
        ),
        predict=_load_predict_config(predict_raw, base_dir),
        train_output_dir=train_output_dir,
    )


def load_config(config_path: Path) -> ForecastExperimentConfig:
    import yaml

    config_path = config_path.resolve()
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base_dir = config_path.parent.parent

    default_model_type = _normalize_model_type(
        str(raw_config.get("default_model_type", "lstm"))
    )
    data_raw = raw_config.get("data", {})
    data_config = ForecastDataConfig(
        feature_names=[
            str(item)
            for item in data_raw.get(
                "feature_names",
                [
                    "aggregate",
                    "slot_sin",
                    "slot_cos",
                    "weekday_sin",
                    "weekday_cos",
                    "active_appliance_count",
                    "burst_event_count",
                ],
            )
        ],
        aggregate_normalization=str(
            data_raw.get("aggregate_normalization", DEFAULT_NORMALIZATION_MODE)
        ),
        aggregate_norm_eps=float(data_raw.get("aggregate_norm_eps", 1e-6)),
    )

    lstm_runtime = _build_lstm_runtime(
        raw_config.get("lstm", {}),
        base_dir,
        len(data_config.feature_names),
    )
    transformer_encoder_direct_runtime = _build_transformer_runtime(
        raw_config.get("transformer_encoder_direct", {}),
        base_dir,
        len(data_config.feature_names),
        "../models_agent/checkpoints/forecast/transformer_encoder_direct",
    )
    transformer_encdec_direct_runtime = _build_transformer_runtime(
        raw_config.get("transformer_encdec_direct", {}),
        base_dir,
        len(data_config.feature_names),
        "../models_agent/checkpoints/forecast/transformer_encdec_direct",
    )

    if default_model_type not in SUPPORTED_FORECAST_MODEL_TYPES:
        raise ValueError(
            "default_model_type 只支持 lstm / transformer_encoder_direct / "
            "transformer_encdec_direct"
        )
    if not data_config.feature_names:
        raise ValueError("feature_names 不能为空")
    if data_config.feature_names[0] != "aggregate":
        raise ValueError("feature_names 第一个特征必须是 aggregate")
    if len(set(data_config.feature_names)) != len(data_config.feature_names):
        raise ValueError("feature_names 不允许重复")
    invalid_feature_names = set(data_config.feature_names).difference(ALL_FEATURE_NAMES)
    if invalid_feature_names:
        raise ValueError(f"存在不支持的 feature_names: {sorted(invalid_feature_names)}")
    if data_config.aggregate_normalization not in {
        DEFAULT_NORMALIZATION_MODE,
        LEGACY_NORMALIZATION_MODE,
    }:
        raise ValueError("aggregate_normalization 只支持 input_window 或 global")
    if lstm_runtime.model.input_size != len(data_config.feature_names):
        raise ValueError("lstm.model.input_size 必须等于 feature_names 长度")
    if lstm_runtime.model.target_length != TARGET_LENGTH:
        raise ValueError(f"lstm.target_length 必须等于 {TARGET_LENGTH}")
    if lstm_runtime.model.input_length <= 0:
        raise ValueError("lstm.input_length 必须大于 0")

    for model_name, runtime in (
        ("transformer_encoder_direct", transformer_encoder_direct_runtime),
        ("transformer_encdec_direct", transformer_encdec_direct_runtime),
    ):
        if runtime.model.input_size != len(data_config.feature_names):
            raise ValueError(f"{model_name}.input_size 必须等于 feature_names 长度")
        if runtime.model.d_model % runtime.model.num_heads != 0:
            raise ValueError(f"{model_name}.d_model 必须能被 num_heads 整除")
        if runtime.model.target_length != TARGET_LENGTH:
            raise ValueError(f"{model_name}.target_length 必须等于 {TARGET_LENGTH}")
        if runtime.model.input_length <= 0:
            raise ValueError(f"{model_name}.input_length 必须大于 0")

    return ForecastExperimentConfig(
        default_model_type=default_model_type,
        data=data_config,
        lstm=lstm_runtime,
        transformer_encoder_direct=transformer_encoder_direct_runtime,
        transformer_encdec_direct=transformer_encdec_direct_runtime,
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
        input_length: int = DEFAULT_LSTM_INPUT_LENGTH,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.target_length = target_length
        self.input_length = input_length
        lstm_dropout = dropout if num_layers > 1 else 0.0

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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                f"输入维度应为 [batch, seq_len, channels]，实际为 {tuple(features.shape)}"
            )
        if features.size(1) != self.input_length:
            raise ValueError(
                f"输入序列长度应为 {self.input_length}，实际为 {features.size(1)}"
            )
        if features.size(-1) != self.input_size:
            raise ValueError(
                f"输入通道数应为 {self.input_size}，实际为 {features.size(-1)}"
            )

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
        for _ in range(self.target_length):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            current_prediction = self.output_layer(decoder_output[:, -1, :])
            predictions.append(current_prediction)
            decoder_input = current_prediction.unsqueeze(1)
        return torch.cat(predictions, dim=1)


class PatchDirectTransformerForecaster(nn.Module):
    """面向 7 天输入的 patch-based direct Transformer。"""

    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.2,
        target_length: int = TARGET_LENGTH,
        patch_length: int = 16,
        patch_stride: int = 8,
        input_length: int = DEFAULT_TRANSFORMER_INPUT_LENGTH,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.history_length = input_length
        self.target_length = target_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.num_patches = self._compute_num_patches()

        self.patch_projection = nn.Linear(input_size * patch_length, d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.output_layer = nn.Sequential(
            nn.LayerNorm(self.num_patches * d_model),
            nn.Linear(self.num_patches * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, target_length),
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def _compute_num_patches(self) -> int:
        if self.patch_length <= 0 or self.patch_stride <= 0:
            raise ValueError("patch_length 和 patch_stride 必须大于 0")
        if self.patch_length > self.history_length:
            raise ValueError("patch_length 不能大于输入历史长度")
        return 1 + (self.history_length - self.patch_length) // self.patch_stride

    def _patchify(self, features: torch.Tensor) -> torch.Tensor:
        patches = features.transpose(1, 2).unfold(
            dimension=2,
            size=self.patch_length,
            step=self.patch_stride,
        )
        patches = patches.permute(0, 2, 1, 3).contiguous()
        batch_size, patch_count, channel_count, patch_length = patches.shape
        if patch_count != self.num_patches:
            raise RuntimeError(
                f"patch 数量异常，期望 {self.num_patches}，实际 {patch_count}"
            )
        return patches.view(batch_size, patch_count, channel_count * patch_length)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                f"输入维度应为 [batch, seq_len, channels]，实际为 {tuple(features.shape)}"
            )
        if features.size(1) != self.history_length:
            raise ValueError(
                f"输入序列长度应为 {self.history_length}，实际为 {features.size(1)}"
            )
        if features.size(-1) != self.input_size:
            raise ValueError(
                f"输入通道数应为 {self.input_size}，实际为 {features.size(-1)}"
            )

        patch_tokens = self.patch_projection(self._patchify(features))
        encoded = self.transformer(
            patch_tokens + self.position_embedding[:, : patch_tokens.size(1), :]
        )
        flattened = encoded.reshape(encoded.size(0), -1)
        return self.output_layer(flattened)


class PatchEncDecDirectTransformerForecaster(nn.Module):
    """面向 7 天输入的 patch encoder-decoder direct Transformer。"""

    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.2,
        target_length: int = TARGET_LENGTH,
        patch_length: int = 16,
        patch_stride: int = 8,
        input_length: int = DEFAULT_TRANSFORMER_INPUT_LENGTH,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.history_length = input_length
        self.target_length = target_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.num_patches = self._compute_num_patches()

        self.patch_projection = nn.Linear(input_size * patch_length, d_model)
        self.encoder_position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, d_model)
        )
        self.decoder_query_embedding = nn.Parameter(
            torch.zeros(1, target_length, d_model)
        )
        self.decoder_position_embedding = nn.Parameter(
            torch.zeros(1, target_length, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.output_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.normal_(self.encoder_position_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.decoder_query_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.decoder_position_embedding, mean=0.0, std=0.02)

    def _compute_num_patches(self) -> int:
        if self.patch_length <= 0 or self.patch_stride <= 0:
            raise ValueError("patch_length 和 patch_stride 必须大于 0")
        if self.patch_length > self.history_length:
            raise ValueError("patch_length 不能大于输入历史长度")
        return 1 + (self.history_length - self.patch_length) // self.patch_stride

    def _patchify(self, features: torch.Tensor) -> torch.Tensor:
        patches = features.transpose(1, 2).unfold(
            dimension=2,
            size=self.patch_length,
            step=self.patch_stride,
        )
        patches = patches.permute(0, 2, 1, 3).contiguous()
        batch_size, patch_count, channel_count, patch_length = patches.shape
        if patch_count != self.num_patches:
            raise RuntimeError(
                f"patch 数量异常，期望 {self.num_patches}，实际 {patch_count}"
            )
        return patches.view(batch_size, patch_count, channel_count * patch_length)

    def _build_decoder_queries(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        queries = self.decoder_query_embedding + self.decoder_position_embedding
        return queries.to(device=device, dtype=dtype).expand(batch_size, -1, -1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                f"输入维度应为 [batch, seq_len, channels]，实际为 {tuple(features.shape)}"
            )
        if features.size(1) != self.history_length:
            raise ValueError(
                f"输入序列长度应为 {self.history_length}，实际为 {features.size(1)}"
            )
        if features.size(-1) != self.input_size:
            raise ValueError(
                f"输入通道数应为 {self.input_size}，实际为 {features.size(-1)}"
            )

        patch_tokens = self.patch_projection(self._patchify(features))
        memory = self.encoder(
            patch_tokens
            + self.encoder_position_embedding[:, : patch_tokens.size(1), :]
        )
        decoded = self.decoder(
            tgt=self._build_decoder_queries(
                batch_size=memory.size(0),
                device=memory.device,
                dtype=memory.dtype,
            ),
            memory=memory,
        )
        return self.output_layer(decoded).squeeze(-1)


def build_lstm_model(model_config: LSTMModelConfig) -> Seq2SeqLSTMForecaster:
    return Seq2SeqLSTMForecaster(
        input_size=model_config.input_size,
        hidden_size=model_config.hidden_size,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
        target_length=model_config.target_length,
        input_length=model_config.input_length,
    )


def build_transformer_encoder_direct_model(
    model_config: TransformerModelConfig,
) -> PatchDirectTransformerForecaster:
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
        input_length=model_config.input_length,
    )


def build_transformer_encdec_direct_model(
    model_config: TransformerModelConfig,
) -> PatchEncDecDirectTransformerForecaster:
    return PatchEncDecDirectTransformerForecaster(
        input_size=model_config.input_size,
        d_model=model_config.d_model,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        ffn_dim=model_config.ffn_dim,
        dropout=model_config.dropout,
        target_length=model_config.target_length,
        patch_length=model_config.patch_length,
        patch_stride=model_config.patch_stride,
        input_length=model_config.input_length,
    )


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location=device)


def _resolve_feature_names(
    fallback_feature_names: list[str],
    checkpoint: dict[str, Any],
) -> list[str]:
    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, dict):
        return fallback_feature_names

    data_raw = raw_config.get("data")
    if not isinstance(data_raw, dict):
        return fallback_feature_names

    raw_feature_names = data_raw.get("feature_names")
    if not isinstance(raw_feature_names, list) or not raw_feature_names:
        return fallback_feature_names

    resolved_feature_names = [str(item) for item in raw_feature_names]
    if resolved_feature_names[0] != "aggregate":
        raise ValueError("checkpoint 中的 feature_names 第一个特征必须是 aggregate")
    invalid_feature_names = set(resolved_feature_names).difference(ALL_FEATURE_NAMES)
    if invalid_feature_names:
        raise ValueError(
            f"checkpoint 中存在不支持的 feature_names: {sorted(invalid_feature_names)}"
        )
    return resolved_feature_names


def _resolve_lstm_model_config(
    fallback_model_config: LSTMModelConfig,
    checkpoint: dict[str, Any],
) -> LSTMModelConfig:
    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, dict):
        return fallback_model_config

    model_raw = raw_config.get("model")
    if not isinstance(model_raw, dict):
        return fallback_model_config

    return LSTMModelConfig(
        input_size=int(model_raw.get("input_size", fallback_model_config.input_size)),
        hidden_size=int(model_raw.get("hidden_size", fallback_model_config.hidden_size)),
        num_layers=int(model_raw.get("num_layers", fallback_model_config.num_layers)),
        dropout=float(model_raw.get("dropout", fallback_model_config.dropout)),
        target_length=int(
            model_raw.get("target_length", fallback_model_config.target_length)
        ),
        input_length=int(
            model_raw.get("input_length", fallback_model_config.input_length)
        ),
    )


def _infer_transformer_input_length(
    checkpoint: dict[str, Any],
    patch_length: int,
    patch_stride: int,
    fallback_input_length: int,
) -> int:
    if patch_length <= 0 or patch_stride <= 0:
        return fallback_input_length

    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, dict):
        return fallback_input_length

    for position_key in ("position_embedding", "encoder_position_embedding"):
        position_embedding = model_state.get(position_key)
        if isinstance(position_embedding, torch.Tensor) and position_embedding.ndim == 3:
            num_patches = int(position_embedding.size(1))
            return patch_length + patch_stride * max(0, num_patches - 1)

    return fallback_input_length


def _resolve_transformer_model_config(
    fallback_model_config: TransformerModelConfig,
    checkpoint: dict[str, Any],
) -> TransformerModelConfig:
    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, dict):
        return fallback_model_config

    model_raw = raw_config.get("model")
    if not isinstance(model_raw, dict):
        return fallback_model_config

    patch_length = int(model_raw.get("patch_length", fallback_model_config.patch_length))
    patch_stride = int(model_raw.get("patch_stride", fallback_model_config.patch_stride))
    input_length = int(
        model_raw.get(
            "input_length",
            _infer_transformer_input_length(
                checkpoint,
                patch_length,
                patch_stride,
                fallback_model_config.input_length,
            ),
        )
    )

    return TransformerModelConfig(
        input_size=int(model_raw.get("input_size", fallback_model_config.input_size)),
        d_model=int(model_raw.get("d_model", fallback_model_config.d_model)),
        num_layers=int(model_raw.get("num_layers", fallback_model_config.num_layers)),
        num_heads=int(model_raw.get("num_heads", fallback_model_config.num_heads)),
        ffn_dim=int(model_raw.get("ffn_dim", fallback_model_config.ffn_dim)),
        dropout=float(model_raw.get("dropout", fallback_model_config.dropout)),
        target_length=int(
            model_raw.get("target_length", fallback_model_config.target_length)
        ),
        patch_length=patch_length,
        patch_stride=patch_stride,
        input_length=input_length,
    )


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


def _build_temporal_feature_sequences(
    start_value: str,
    input_length: int,
) -> dict[str, list[float]]:
    base_time = pd.to_datetime(start_value)
    if pd.isna(base_time):
        raise ValueError("无法根据 timestamp 自动生成时间特征，请提供合法时间")

    timestamps = pd.date_range(start=base_time, periods=input_length, freq="15min")
    slot_index = timestamps.hour * 4 + timestamps.minute // 15
    weekday_index = timestamps.dayofweek
    slot_angle = 2.0 * np.pi * slot_index.to_numpy(dtype=np.float32) / STEPS_PER_DAY
    weekday_angle = 2.0 * np.pi * weekday_index.to_numpy(dtype=np.float32) / 7.0
    return {
        "slot_sin": np.sin(slot_angle).astype(np.float32).tolist(),
        "slot_cos": np.cos(slot_angle).astype(np.float32).tolist(),
        "weekday_sin": np.sin(weekday_angle).astype(np.float32).tolist(),
        "weekday_cos": np.cos(weekday_angle).astype(np.float32).tolist(),
    }


def _build_feature_values(
    sample: dict[str, Any],
    feature_names: list[str],
    input_length: int,
) -> dict[str, list[float]]:
    temporal_feature_names = {
        "slot_sin",
        "slot_cos",
        "weekday_sin",
        "weekday_cos",
    }
    feature_values: dict[str, list[float] | None] = {}

    series = sample.get("series")
    if isinstance(series, list):
        for feature_name in feature_names:
            raw_values = [item.get(feature_name) for item in series]
            if any(value is None for value in raw_values):
                feature_values[feature_name] = None
            else:
                feature_values[feature_name] = [float(value) for value in raw_values]
    else:
        for feature_name in feature_names:
            raw_value = sample.get(feature_name)
            feature_values[feature_name] = raw_value if raw_value is not None else None

    aggregate = feature_values.get("aggregate")
    if aggregate is None or len(aggregate) != input_length:
        raise ValueError(f"aggregate 输入序列长度必须为 {input_length}")

    missing_temporal_features = [
        feature_name
        for feature_name in feature_names
        if feature_name in temporal_feature_names and feature_values.get(feature_name) is None
    ]
    if missing_temporal_features:
        input_start = str(sample.get("input_start", "")).strip()
        if not input_start and isinstance(series, list) and series:
            input_start = str(series[0].get("timestamp", "")).strip()
        temporal_sequences = _build_temporal_feature_sequences(input_start, input_length)
        for feature_name in missing_temporal_features:
            feature_values[feature_name] = temporal_sequences[feature_name]

    for feature_name in feature_names:
        values = feature_values.get(feature_name)
        if values is None or len(values) != input_length:
            raise ValueError(f"{feature_name} 输入序列长度必须为 {input_length}")

    return {
        feature_name: [float(value) for value in feature_values[feature_name] or []]
        for feature_name in feature_names
    }


def _select_runtime_config(
    experiment_config: ForecastExperimentConfig,
    model_type: str,
) -> tuple[str, LSTMRuntimeConfig | TransformerRuntimeConfig]:
    normalized_model_type = _normalize_model_type(model_type)
    if normalized_model_type == "lstm":
        return normalized_model_type, experiment_config.lstm
    if normalized_model_type == "transformer_encoder_direct":
        return normalized_model_type, experiment_config.transformer_encoder_direct
    if normalized_model_type == "transformer_encdec_direct":
        return normalized_model_type, experiment_config.transformer_encdec_direct
    raise ValueError(
        "预测模型只支持 lstm / transformer_encoder_direct / "
        "transformer_encdec_direct"
    )


def get_required_input_length(
    config_path: Path,
    model_type: str | None = None,
) -> int:
    experiment_config = load_config(config_path)
    selected_model_type, runtime_config = _select_runtime_config(
        experiment_config,
        model_type or experiment_config.default_model_type,
    )
    checkpoint_path = runtime_config.predict.checkpoint_path or (
        runtime_config.train_output_dir / "best_model.pt"
    )
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
        if selected_model_type == "lstm":
            return _resolve_lstm_model_config(runtime_config.model, checkpoint).input_length
        return _resolve_transformer_model_config(
            runtime_config.model,
            checkpoint,
        ).input_length

    return runtime_config.model.input_length


def predict_single_sample(
    sample: dict[str, Any],
    config_path: Path,
    model_type: str | None = None,
) -> list[float]:
    experiment_config = load_config(config_path)
    selected_model_type, runtime_config = _select_runtime_config(
        experiment_config,
        model_type or experiment_config.default_model_type,
    )
    checkpoint_path = runtime_config.predict.checkpoint_path or (
        runtime_config.train_output_dir / "best_model.pt"
    )
    device_name = detect_device()
    device = torch.device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, device)
    feature_names = _resolve_feature_names(
        experiment_config.data.feature_names,
        checkpoint,
    )

    if selected_model_type == "lstm":
        model_config = _resolve_lstm_model_config(runtime_config.model, checkpoint)
    else:
        model_config = _resolve_transformer_model_config(
            runtime_config.model,
            checkpoint,
        )

    feature_values = _build_feature_values(
        sample,
        feature_names,
        model_config.input_length,
    )
    features = np.stack(
        [
            np.asarray(feature_values[feature_name], dtype=np.float32)
            for feature_name in feature_names
        ],
        axis=-1,
    )

    normalization = checkpoint_to_normalization(checkpoint)
    normalized, denorm_mean, denorm_std = _normalize_features(
        features[np.newaxis, ...],
        normalization,
    )

    if selected_model_type == "lstm":
        if model_config.input_size != len(feature_names):
            raise ValueError("LSTM checkpoint 的 input_size 与 feature_names 长度不一致")
        model = build_lstm_model(model_config).to(device)
    elif selected_model_type == "transformer_encoder_direct":
        if model_config.input_size != len(feature_names):
            raise ValueError(
                "Transformer Encoder Direct checkpoint 的 input_size 与 "
                "feature_names 长度不一致"
            )
        model = build_transformer_encoder_direct_model(model_config).to(device)
    else:
        if model_config.input_size != len(feature_names):
            raise ValueError(
                "Transformer EncDec Direct checkpoint 的 input_size 与 "
                "feature_names 长度不一致"
            )
        model = build_transformer_encdec_direct_model(model_config).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        outputs = model(torch.from_numpy(normalized).to(device))
        predictions = outputs.cpu().numpy()[0]

    return [
        max(0.0, float(value * denorm_std[0, 0] + denorm_mean[0, 0]))
        for value in predictions
    ]

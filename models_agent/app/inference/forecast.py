"""TFT 预测推理实现。"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


STEPS_PER_DAY = 96
DEFAULT_INPUT_LENGTH = 7 * STEPS_PER_DAY
TARGET_LENGTH = STEPS_PER_DAY
DEFAULT_PROFILE_CLASSIFIER_CONFIG_PATH = (
    "../models_agent/configs/forecast_profile_classification.yaml"
)
SUPPORTED_FORECAST_MODEL_TYPES = ("tft",)
BASE_FEATURE_NAMES = (
    "aggregate",
    "slot_sin",
    "slot_cos",
    "weekday_sin",
    "weekday_cos",
)
DECODER_FEATURE_DIM = 5
BASELINE_STAT_DIM = 9


@dataclass(slots=True)
class ForecastPredictConfig:
    checkpoint_path: Path
    profile_classifier_config_path: Path
    profile_labels: tuple[str, ...]


@dataclass(slots=True)
class TftRuntimeConfig:
    input_length: int
    target_length: int
    baseline_week_weight: float
    baseline_recent_weight: float
    normalization_quantile: float
    predict: ForecastPredictConfig


@dataclass(slots=True)
class ForecastExperimentConfig:
    default_model_type: str
    tft: TftRuntimeConfig


@dataclass(slots=True)
class TftInferenceBundle:
    model: "TemporalFusionTransformer"
    device: torch.device
    input_length: int
    target_length: int
    baseline_week_weight: float
    baseline_recent_weight: float
    normalization_quantile: float
    profile_labels: tuple[str, ...]
    profile_classifier_config_path: Path


class GatedLinearUnit(nn.Module):
    """门控线性单元。"""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value, gate = self.linear(inputs).chunk(2, dim=-1)
        return value * torch.sigmoid(gate)


class GateAddNorm(nn.Module):
    """门控残差与层归一化。"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        residual_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.glu = GatedLinearUnit(input_dim, output_dim)
        self.residual_projection = (
            nn.Identity()
            if residual_dim is None or residual_dim == output_dim
            else nn.Linear(residual_dim, output_dim)
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, inputs: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.norm(self.glu(inputs) + self.residual_projection(residual))


class GatedResidualNetwork(nn.Module):
    """TFT 使用的 GRN 模块。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        resolved_output_dim = output_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, resolved_output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = GateAddNorm(
            input_dim=resolved_output_dim,
            output_dim=resolved_output_dim,
            residual_dim=input_dim,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.elu(self.fc1(inputs))
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        return self.gate(hidden, inputs)


class VariableSelectionNetwork(nn.Module):
    """连续变量选择网络。"""

    def __init__(self, num_variables: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.num_variables = num_variables
        self.variable_projections = nn.ModuleList(
            [nn.Linear(1, hidden_size) for _ in range(num_variables)]
        )
        self.variable_grns = nn.ModuleList(
            [
                GatedResidualNetwork(
                    input_dim=hidden_size,
                    hidden_dim=hidden_size,
                    output_dim=hidden_size,
                    dropout=dropout,
                )
                for _ in range(num_variables)
            ]
        )
        self.weight_grn = GatedResidualNetwork(
            input_dim=num_variables,
            hidden_dim=hidden_size,
            output_dim=num_variables,
            dropout=dropout,
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sparse_weights = torch.softmax(self.weight_grn(inputs), dim=-1)
        transformed_variables: list[torch.Tensor] = []
        for variable_index in range(self.num_variables):
            single_variable = inputs[..., variable_index : variable_index + 1]
            projected = self.variable_projections[variable_index](single_variable)
            transformed = self.variable_grns[variable_index](projected)
            transformed_variables.append(transformed)

        stacked = torch.stack(transformed_variables, dim=-2)
        combined = torch.sum(sparse_weights.unsqueeze(-1) * stacked, dim=-2)
        return combined, sparse_weights


class ExpertResidualHead(nn.Module):
    """单个 residual expert。"""

    def __init__(self, input_dim: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.grn = GatedResidualNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_size,
            output_dim=hidden_size,
            dropout=dropout,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.grn(inputs)).squeeze(-1)


class TemporalFusionTransformer(nn.Module):
    """与训练侧一致的 baseline-aware TFT v2 主体。"""

    def __init__(
        self,
        encoder_input_dim: int,
        decoder_input_dim: int,
        hidden_size: int,
        lstm_layers: int,
        attention_heads: int,
        dropout: float,
        prediction_length: int,
        profile_dim: int,
        baseline_stat_dim: int,
        router_prior_weight: float = 1.0,
        global_gate_bias_init: float = -0.2,
        local_gate_bias_init: float = 0.0,
        return_attention_weights: bool = False,
    ) -> None:
        super().__init__()
        self.prediction_length = prediction_length
        self.profile_dim = profile_dim
        self.return_attention_weights = return_attention_weights
        self.router_prior_weight = router_prior_weight

        self.encoder_vsn = VariableSelectionNetwork(
            num_variables=encoder_input_dim,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.decoder_vsn = VariableSelectionNetwork(
            num_variables=decoder_input_dim,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        lstm_dropout = dropout if lstm_layers > 1 else 0.0
        self.encoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.post_lstm_gate = GateAddNorm(
            input_dim=hidden_size,
            output_dim=hidden_size,
            residual_dim=hidden_size,
        )
        self.enrichment_grn = GatedResidualNetwork(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=hidden_size,
            dropout=dropout,
        )
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.post_attention_gate = GateAddNorm(
            input_dim=hidden_size,
            output_dim=hidden_size,
            residual_dim=hidden_size,
        )
        self.positionwise_grn = GatedResidualNetwork(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=hidden_size,
            dropout=dropout,
        )
        self.pre_output_gate = GateAddNorm(
            input_dim=hidden_size,
            output_dim=hidden_size,
            residual_dim=hidden_size,
        )

        context_input_dim = hidden_size * 2 + profile_dim + baseline_stat_dim
        self.context_grn = GatedResidualNetwork(
            input_dim=context_input_dim,
            hidden_dim=hidden_size,
            output_dim=hidden_size,
            dropout=dropout,
        )
        self.router_grn = GatedResidualNetwork(
            input_dim=hidden_size + profile_dim,
            hidden_dim=hidden_size,
            output_dim=hidden_size,
            dropout=dropout,
        )
        self.router_head = nn.Linear(hidden_size, profile_dim)

        expert_input_dim = hidden_size * 2
        self.expert_heads = nn.ModuleList(
            [
                ExpertResidualHead(
                    input_dim=expert_input_dim,
                    hidden_size=hidden_size,
                    dropout=dropout,
                )
                for _ in range(profile_dim)
            ]
        )
        self.local_gate_grn = GatedResidualNetwork(
            input_dim=expert_input_dim,
            hidden_dim=hidden_size,
            output_dim=hidden_size,
            dropout=dropout,
        )
        self.local_gate_head = nn.Linear(hidden_size, 1)
        self.global_gate_grn = GatedResidualNetwork(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=hidden_size,
            dropout=dropout,
        )
        self.global_gate_head = nn.Linear(hidden_size, 1)

        nn.init.constant_(self.local_gate_head.bias, local_gate_bias_init)
        nn.init.constant_(self.global_gate_head.bias, global_gate_bias_init)

    def forward(
        self,
        encoder_cont: torch.Tensor,
        decoder_known: torch.Tensor,
        profile_prior: torch.Tensor,
        baseline_stats: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoder_selected, encoder_weights = self.encoder_vsn(encoder_cont)
        decoder_selected, decoder_weights = self.decoder_vsn(decoder_known)

        encoder_output, encoder_state = self.encoder_lstm(encoder_selected)
        decoder_output, _ = self.decoder_lstm(decoder_selected, encoder_state)
        sequence = torch.cat([encoder_output, decoder_output], dim=1)
        residual_sequence = torch.cat([encoder_selected, decoder_selected], dim=1)
        sequence = self.post_lstm_gate(sequence, residual_sequence)
        sequence = self.enrichment_grn(sequence)

        query = sequence[:, -self.prediction_length :, :]
        attention_mask = self._build_attention_mask(
            encoder_length=encoder_selected.size(1),
            prediction_length=self.prediction_length,
            device=sequence.device,
        )
        attention_output, attention_weights = self.self_attention(
            query=query,
            key=sequence,
            value=sequence,
            attn_mask=attention_mask,
            need_weights=self.return_attention_weights,
            average_attn_weights=False,
        )
        attention_output = self.post_attention_gate(attention_output, query)
        decoder_features = self.positionwise_grn(attention_output)
        decoder_features = self.pre_output_gate(decoder_features, attention_output)

        context_features = torch.cat(
            [
                encoder_output.mean(dim=1),
                decoder_features.mean(dim=1),
                profile_prior,
                baseline_stats,
            ],
            dim=-1,
        )
        context_vector = self.context_grn(context_features)

        router_input = torch.cat([context_vector, profile_prior], dim=-1)
        router_logits = self.router_head(self.router_grn(router_input))
        router_logits = router_logits + self.router_prior_weight * torch.log(
            profile_prior.clamp_min(1e-6)
        )
        expert_weights = torch.softmax(router_logits, dim=-1)

        expanded_context = context_vector.unsqueeze(1).expand(-1, self.prediction_length, -1)
        expert_input = torch.cat([decoder_features, expanded_context], dim=-1)
        expert_predictions = torch.stack(
            [expert_head(expert_input) for expert_head in self.expert_heads],
            dim=-1,
        )
        mixed_residual = torch.sum(expert_predictions * expert_weights.unsqueeze(1), dim=-1)

        global_gate = torch.sigmoid(
            self.global_gate_head(self.global_gate_grn(context_vector))
        ).squeeze(-1)
        local_gate = torch.sigmoid(
            self.local_gate_head(self.local_gate_grn(expert_input))
        ).squeeze(-1)
        residual_prediction = mixed_residual * global_gate.unsqueeze(-1) * local_gate

        return {
            "residual_prediction": residual_prediction,
            "mixed_residual": mixed_residual,
            "expert_predictions": expert_predictions,
            "expert_weights": expert_weights,
            "profile_prior": profile_prior,
            "global_gate": global_gate,
            "local_gate": local_gate,
            "context_vector": context_vector,
            "encoder_variable_weights": encoder_weights,
            "decoder_variable_weights": decoder_weights,
            "attention_weights": attention_weights if self.return_attention_weights else None,
        }

    @staticmethod
    def _build_attention_mask(
        encoder_length: int,
        prediction_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_length = encoder_length + prediction_length
        key_positions = torch.arange(total_length, device=device)
        query_positions = encoder_length + torch.arange(prediction_length, device=device)
        return key_positions.unsqueeze(0) > query_positions.unsqueeze(1)


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def _normalize_model_type(model_type: str) -> str:
    normalized = model_type.strip().lower()
    if normalized not in SUPPORTED_FORECAST_MODEL_TYPES:
        raise ValueError("预测模型只支持 tft")
    return normalized


def _resolve_path(path_value: str | None, base_dir: Path) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


@lru_cache(maxsize=8)
def load_config(config_path: Path) -> ForecastExperimentConfig:
    import yaml

    resolved_path = config_path.resolve()
    raw_config = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    base_dir = resolved_path.parent.parent

    default_model_type = _normalize_model_type(
        str(raw_config.get("default_model_type", "tft"))
    )
    tft_raw = raw_config.get("tft", {})
    checkpoint_path = _resolve_path(
        str(
            tft_raw.get(
                "checkpoint_path",
                "../models_agent/checkpoints/forecast/tft/best.ckpt",
            )
        ),
        base_dir,
    )
    if checkpoint_path is None:
        raise ValueError("tft.checkpoint_path 不能为空")

    profile_classifier_config_path = _resolve_path(
        str(
            tft_raw.get(
                "profile_classifier_config_path",
                DEFAULT_PROFILE_CLASSIFIER_CONFIG_PATH,
            )
        ),
        base_dir,
    )
    if profile_classifier_config_path is None:
        raise ValueError("tft.profile_classifier_config_path 不能为空")

    profile_labels = tuple(
        str(label)
        for label in tft_raw.get(
            "profile_labels",
            [
                "afternoon_peak",
                "all_day_low",
                "day_low_night_high",
                "morning_peak",
            ],
        )
    )
    if not profile_labels:
        raise ValueError("tft.profile_labels 不能为空")

    return ForecastExperimentConfig(
        default_model_type=default_model_type,
        tft=TftRuntimeConfig(
            input_length=int(tft_raw.get("input_length", DEFAULT_INPUT_LENGTH)),
            target_length=int(tft_raw.get("target_length", TARGET_LENGTH)),
            baseline_week_weight=float(tft_raw.get("baseline_week_weight", 0.8)),
            baseline_recent_weight=float(tft_raw.get("baseline_recent_weight", 0.2)),
            normalization_quantile=float(tft_raw.get("normalization_quantile", 0.95)),
            predict=ForecastPredictConfig(
                checkpoint_path=checkpoint_path,
                profile_classifier_config_path=profile_classifier_config_path,
                profile_labels=profile_labels,
            ),
        ),
    )


def _load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location=device)


def _build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> TemporalFusionTransformer:
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    feature_spec = hyper_parameters.get("feature_spec", {})
    model_config = hyper_parameters.get("model_config", {})
    model = TemporalFusionTransformer(
        encoder_input_dim=int(feature_spec["feature_dim"]),
        decoder_input_dim=DECODER_FEATURE_DIM,
        hidden_size=int(model_config["hidden_size"]),
        lstm_layers=int(model_config["lstm_layers"]),
        attention_heads=int(model_config["attention_heads"]),
        dropout=float(model_config["dropout"]),
        prediction_length=int(feature_spec["target_length"]),
        profile_dim=int(feature_spec["profile_dim"]),
        baseline_stat_dim=int(feature_spec["baseline_stat_dim"]),
        router_prior_weight=float(model_config.get("router_prior_weight", 1.0)),
        global_gate_bias_init=float(model_config.get("global_gate_bias_init", -0.2)),
        local_gate_bias_init=float(model_config.get("local_gate_bias_init", 0.0)),
        return_attention_weights=bool(model_config.get("return_attention_weights", False)),
    ).to(device)

    state_dict = {
        key.removeprefix("model."): value
        for key, value in checkpoint.get("state_dict", {}).items()
        if key.startswith("model.")
    }
    if not state_dict:
        raise ValueError("TFT checkpoint 缺少 model.* 权重")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@lru_cache(maxsize=8)
def _load_runtime_bundle(
    config_path_str: str,
    model_type: str,
    device_name: str,
) -> TftInferenceBundle:
    experiment_config = load_config(Path(config_path_str))
    selected_model_type = _normalize_model_type(model_type)
    if selected_model_type != "tft":
        raise ValueError("预测模型只支持 tft")

    runtime_config = experiment_config.tft
    device = torch.device(device_name)
    checkpoint = _load_checkpoint(runtime_config.predict.checkpoint_path, device)
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    data_config = hyper_parameters.get("data_config", {})
    feature_spec = hyper_parameters.get("feature_spec", {})

    target_length = int(feature_spec.get("target_length", runtime_config.target_length))
    profile_dim = int(feature_spec.get("profile_dim", len(runtime_config.predict.profile_labels)))
    if target_length != runtime_config.target_length:
        raise ValueError(
            f"TFT checkpoint 的 target_length={target_length} 与配置不一致"
        )
    if profile_dim != len(runtime_config.predict.profile_labels):
        raise ValueError(
            "TFT checkpoint 的 profile_dim 与 tft.profile_labels 数量不一致"
        )

    return TftInferenceBundle(
        model=_build_model_from_checkpoint(checkpoint, device),
        device=device,
        input_length=int(runtime_config.input_length),
        target_length=target_length,
        baseline_week_weight=float(
            data_config.get(
                "baseline_week_weight",
                runtime_config.baseline_week_weight,
            )
        ),
        baseline_recent_weight=float(
            data_config.get(
                "baseline_recent_weight",
                runtime_config.baseline_recent_weight,
            )
        ),
        normalization_quantile=float(
            data_config.get(
                "normalization_quantile",
                runtime_config.normalization_quantile,
            )
        ),
        profile_labels=runtime_config.predict.profile_labels,
        profile_classifier_config_path=runtime_config.predict.profile_classifier_config_path,
    )


def _classification_worker_path() -> Path:
    return Path(__file__).resolve().with_name("classification_worker.py")


def _predict_day_profile_probabilities(
    day_frame: pd.DataFrame,
    profile_classifier_config_path: Path,
    profile_labels: tuple[str, ...],
) -> np.ndarray:
    payload = {
        "sample": {
            "sample_id": f"profile_{day_frame.iloc[0]['timestamp'].date()}",
            "date": str(day_frame.iloc[0]["timestamp"].date()),
            "aggregate": day_frame["aggregate"].astype(float).tolist(),
        },
        "config_path": str(profile_classifier_config_path),
    }
    completed = subprocess.run(
        [sys.executable, str(_classification_worker_path())],
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(
            f"profile 分类器子进程执行失败: {stderr or '分类子进程异常退出'}"
        )

    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError("profile 分类器子进程未返回结果")

    result = json.loads(stdout)
    probability_map = result.get("probabilities", {}) or {}
    if not isinstance(probability_map, dict):
        raise RuntimeError("profile 分类器返回的 probabilities 格式错误")
    return np.asarray(
        [float(probability_map.get(label, 0.0)) for label in profile_labels],
        dtype=np.float32,
    )


def _extract_series(sample: dict[str, Any], input_length: int) -> pd.DataFrame:
    series = sample.get("series")
    if not isinstance(series, list) or len(series) != input_length:
        raise ValueError(f"预测输入历史序列长度必须为 {input_length}")

    frame = pd.DataFrame(series)
    required_columns = {"timestamp", "aggregate"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"预测输入缺少字段: {sorted(missing_columns)}")

    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    if frame["timestamp"].isna().any():
        raise ValueError("预测输入中的 timestamp 必须为合法时间")
    frame["aggregate"] = frame["aggregate"].astype(np.float32)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def _build_temporal_features(timestamps: pd.Series) -> dict[str, np.ndarray]:
    slot_index = timestamps.dt.hour * 4 + timestamps.dt.minute // 15
    weekday_index = timestamps.dt.dayofweek
    slot_angle = 2.0 * np.pi * slot_index.to_numpy(dtype=np.float32) / STEPS_PER_DAY
    weekday_angle = 2.0 * np.pi * weekday_index.to_numpy(dtype=np.float32) / 7.0
    return {
        "slot_sin": np.sin(slot_angle).astype(np.float32),
        "slot_cos": np.cos(slot_angle).astype(np.float32),
        "weekday_sin": np.sin(weekday_angle).astype(np.float32),
        "weekday_cos": np.cos(weekday_angle).astype(np.float32),
    }


def _build_profile_feature_matrix(
    frame: pd.DataFrame,
    target_length: int,
    profile_labels: tuple[str, ...],
    profile_classifier_config_path: Path,
    provided_profile_probability_days: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], str]:
    if len(frame) % target_length != 0:
        raise ValueError("历史序列长度必须能按整天切分")

    day_dates = [
        str(frame.iloc[start_index]["timestamp"].date())
        for start_index in range(0, len(frame), target_length)
    ]
    resolved_probability_days: list[dict[str, Any]] = []
    day_probabilities: list[np.ndarray] = []
    if provided_profile_probability_days:
        probability_map_by_date: dict[str, dict[str, float]] = {}
        for item in provided_profile_probability_days:
            if not isinstance(item, dict):
                raise ValueError("profile_probability_days 的元素必须是对象")
            date_value = str(item.get("date", "")).strip()
            if not date_value:
                raise ValueError("profile_probability_days.date 不能为空")
            raw_probabilities = item.get("probabilities", {})
            if not isinstance(raw_probabilities, dict):
                raise ValueError("profile_probability_days.probabilities 必须是对象")
            probability_map_by_date[date_value] = {
                str(label): float(raw_probabilities.get(label, 0.0))
                for label in profile_labels
            }

        missing_dates = [
            date_value for date_value in day_dates if date_value not in probability_map_by_date
        ]
        if missing_dates:
            raise ValueError(
                f"profile_probability_days 缺少以下日期: {missing_dates}"
            )

        for date_value in day_dates:
            probability_map = probability_map_by_date[date_value]
            day_probability = np.asarray(
                [float(probability_map[label]) for label in profile_labels],
                dtype=np.float32,
            )
            day_probabilities.append(day_probability)
            resolved_probability_days.append(
                {
                    "date": date_value,
                    "probabilities": {
                        label: float(probability_map[label]) for label in profile_labels
                    },
                }
            )
        probability_source = "request"
    else:
        for start_index in range(0, len(frame), target_length):
            day_frame = frame.iloc[start_index : start_index + target_length]
            day_probability = _predict_day_profile_probabilities(
                day_frame=day_frame,
                profile_classifier_config_path=profile_classifier_config_path,
                profile_labels=profile_labels,
            )
            day_probabilities.append(day_probability)
            resolved_probability_days.append(
                {
                    "date": str(day_frame.iloc[0]["timestamp"].date()),
                    "probabilities": {
                        label: float(day_probability[index])
                        for index, label in enumerate(profile_labels)
                    },
                }
            )
        probability_source = "internal_classifier"
    day_probability_matrix = np.stack(day_probabilities, axis=0)

    repeated_probabilities = np.repeat(day_probability_matrix, target_length, axis=0)
    return (
        repeated_probabilities.astype(np.float32),
        resolved_probability_days,
        probability_source,
    )


def _build_encoder_features(
    frame: pd.DataFrame,
    bundle: TftInferenceBundle,
    provided_profile_probability_days: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], str]:
    temporal_features = _build_temporal_features(frame["timestamp"])
    profile_matrix, resolved_probability_days, probability_source = _build_profile_feature_matrix(
        frame=frame,
        target_length=bundle.target_length,
        profile_labels=bundle.profile_labels,
        profile_classifier_config_path=bundle.profile_classifier_config_path,
        provided_profile_probability_days=provided_profile_probability_days,
    )
    feature_columns = [
        frame["aggregate"].to_numpy(dtype=np.float32),
        temporal_features["slot_sin"],
        temporal_features["slot_cos"],
        temporal_features["weekday_sin"],
        temporal_features["weekday_cos"],
    ]
    encoder_cont = np.concatenate(
        [
            np.stack(feature_columns, axis=-1),
            profile_matrix,
        ],
        axis=-1,
    ).astype(np.float32)
    return (
        encoder_cont,
        profile_matrix,
        resolved_probability_days,
        probability_source,
    )


def _build_baseline(
    aggregate: np.ndarray,
    bundle: TftInferenceBundle,
) -> np.ndarray:
    first_day = aggregate[: bundle.target_length]
    last_day = aggregate[-bundle.target_length :]
    baseline = (
        bundle.baseline_week_weight * first_day
        + bundle.baseline_recent_weight * last_day
    )
    return baseline.astype(np.float32)


def _build_profile_prior(
    profile_matrix: np.ndarray,
    bundle: TftInferenceBundle,
) -> np.ndarray:
    first_day_prior = profile_matrix[: bundle.target_length].mean(axis=0)
    last_day_prior = profile_matrix[-bundle.target_length :].mean(axis=0)
    profile_prior = (
        bundle.baseline_week_weight * first_day_prior
        + bundle.baseline_recent_weight * last_day_prior
    )
    profile_prior = np.clip(profile_prior, a_min=1e-6, a_max=None)
    profile_prior = profile_prior / np.sum(profile_prior)
    return profile_prior.astype(np.float32)


def _safe_corr(first_day: np.ndarray, last_day: np.ndarray) -> np.float32:
    first_std = float(np.std(first_day))
    last_std = float(np.std(last_day))
    if first_std < 1e-6 or last_std < 1e-6:
        return np.float32(0.0)
    return np.float32(np.corrcoef(first_day, last_day)[0, 1])


def _build_baseline_stats(
    aggregate_norm: np.ndarray,
    bundle: TftInferenceBundle,
) -> np.ndarray:
    first_day = aggregate_norm[: bundle.target_length]
    last_day = aggregate_norm[-bundle.target_length :]
    week_matrix = aggregate_norm.reshape(-1, bundle.target_length)
    difference = last_day - first_day
    recent_ramp = np.diff(last_day, prepend=last_day[:1])
    day_totals = week_matrix.sum(axis=1)
    return np.asarray(
        [
            np.mean(np.abs(difference)),
            np.sqrt(np.mean(np.square(difference))),
            np.max(np.abs(difference)),
            _safe_corr(first_day, last_day),
            np.std(last_day),
            np.mean(last_day),
            np.mean(np.abs(recent_ramp)),
            np.max(np.abs(recent_ramp)),
            np.std(day_totals) / max(np.mean(day_totals), 1e-6),
        ],
        dtype=np.float32,
    )


def _build_decoder_known_features(
    target_start: pd.Timestamp,
    baseline_norm: np.ndarray,
    target_length: int,
) -> np.ndarray:
    slots = np.arange(target_length, dtype=np.float32)
    slot_angle = 2.0 * np.pi * slots / float(target_length)
    weekday_index = float(pd.Timestamp(target_start).dayofweek)
    weekday_angle = 2.0 * np.pi * weekday_index / 7.0
    return np.stack(
        [
            np.sin(slot_angle).astype(np.float32),
            np.cos(slot_angle).astype(np.float32),
            np.full(target_length, np.sin(weekday_angle), dtype=np.float32),
            np.full(target_length, np.cos(weekday_angle), dtype=np.float32),
            baseline_norm.astype(np.float32),
        ],
        axis=-1,
    )


def get_required_input_length(
    config_path: Path,
    model_type: str | None = None,
) -> int:
    experiment_config = load_config(config_path)
    selected_model_type = _normalize_model_type(
        model_type or experiment_config.default_model_type
    )
    if selected_model_type != "tft":
        raise ValueError("预测模型只支持 tft")
    return experiment_config.tft.input_length


def predict_single_sample_detailed(
    sample: dict[str, Any],
    config_path: Path,
    model_type: str | None = None,
) -> dict[str, Any]:
    experiment_config = load_config(config_path)
    selected_model_type = _normalize_model_type(
        model_type or experiment_config.default_model_type
    )
    device_name = detect_device()
    bundle = _load_runtime_bundle(
        str(config_path.resolve()),
        selected_model_type,
        device_name,
    )
    frame = _extract_series(sample, bundle.input_length)
    provided_profile_probability_days = sample.get("profile_probability_days")
    encoder_cont, profile_matrix, resolved_probability_days, probability_source = (
        _build_encoder_features(
            frame,
            bundle,
            provided_profile_probability_days=(
                list(provided_profile_probability_days)
                if isinstance(provided_profile_probability_days, list)
                else None
            ),
        )
    )
    aggregate = encoder_cont[:, 0].copy()
    baseline = _build_baseline(aggregate, bundle)
    scale = max(float(np.quantile(aggregate, bundle.normalization_quantile)), 1.0)

    encoder_cont[:, 0] /= scale
    aggregate_norm = encoder_cont[:, 0].copy()
    baseline_norm = baseline / scale
    profile_prior = _build_profile_prior(profile_matrix, bundle)
    baseline_stats = _build_baseline_stats(aggregate_norm, bundle)

    raw_target_start = str(
        sample.get("target_start")
        or sample.get("forecast_start")
        or frame.iloc[-1]["timestamp"] + pd.Timedelta(minutes=15)
    )
    target_start = pd.to_datetime(raw_target_start)
    if pd.isna(target_start):
        raise ValueError("预测样本缺少合法的 target_start / forecast_start")
    decoder_known = _build_decoder_known_features(
        target_start=target_start,
        baseline_norm=baseline_norm,
        target_length=bundle.target_length,
    )

    encoder_tensor = torch.from_numpy(encoder_cont[np.newaxis, ...]).to(
        bundle.device,
        dtype=torch.float32,
    )
    decoder_tensor = torch.from_numpy(decoder_known[np.newaxis, ...]).to(
        bundle.device,
        dtype=torch.float32,
    )
    profile_prior_tensor = torch.from_numpy(profile_prior[np.newaxis, ...]).to(
        bundle.device,
        dtype=torch.float32,
    )
    baseline_stats_tensor = torch.from_numpy(baseline_stats[np.newaxis, ...]).to(
        bundle.device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        outputs = bundle.model(
            encoder_cont=encoder_tensor,
            decoder_known=decoder_tensor,
            profile_prior=profile_prior_tensor,
            baseline_stats=baseline_stats_tensor,
        )
        residual_prediction = outputs["residual_prediction"].detach().float().cpu().numpy()[0]

    prediction = (baseline_norm + residual_prediction) * scale
    prediction = np.clip(prediction, a_min=0.0, a_max=None).astype(np.float32)
    return {
        "predictions": [float(value) for value in prediction.tolist()],
        "profile_probability_days": resolved_probability_days,
        "profile_probability_source": probability_source,
        "profile_prior": {
            label: float(profile_prior[index])
            for index, label in enumerate(bundle.profile_labels)
        },
    }


def predict_single_sample(
    sample: dict[str, Any],
    config_path: Path,
    model_type: str | None = None,
) -> list[float]:
    return predict_single_sample_detailed(
        sample=sample,
        config_path=config_path,
        model_type=model_type,
    )["predictions"]

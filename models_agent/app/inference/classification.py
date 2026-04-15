"""XGBoost 分类推理实现。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LABELS = [
    "daytime_active",
    "daytime_peak_strong",
    "flat_stable",
    "night_dominant",
]
LABEL_DISPLAY_NAMES = {
    "daytime_active": "白天活跃型",
    "daytime_peak_strong": "白天尖峰明显型",
    "flat_stable": "平稳基线型",
    "night_dominant": "夜间主导型",
}
SEQUENCE_LENGTH = 96
FEATURE_NAMES = ("aggregate",)
BLOCK_SIZE = 12
NUM_BLOCKS = SEQUENCE_LENGTH // BLOCK_SIZE
AGGREGATE_COLUMNS = tuple(f"aggregate_{index:03d}" for index in range(SEQUENCE_LENGTH))
TABULAR_FEATURE_NAMES = (
    "full_mean",
    "full_std",
    "full_min",
    "full_max",
    "full_range",
    "load_factor",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "day_mean",
    "night_mean",
    "day_std",
    "night_std",
    "day_night_diff",
    "day_night_ratio",
    "night_day_ratio",
    "morning_mean",
    "daytime_mean",
    "evening_mean",
    "overnight_mean",
    "peak_value",
    "peak_slot_index_norm",
    "valley_value",
    "valley_slot_index_norm",
    "peak_to_mean_ratio",
    "ramp_abs_mean",
    "ramp_abs_std",
    "ramp_abs_max",
    "ramp_up_mean",
    "ramp_down_mean",
    "high_load_ratio",
    "low_load_ratio",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
    "block_mean_00",
    "block_mean_01",
    "block_mean_02",
    "block_mean_03",
    "block_mean_04",
    "block_mean_05",
    "block_mean_06",
    "block_mean_07",
)
DAY_START_SLOT = 32
DAY_END_SLOT = 72


@dataclass(slots=True)
class ClassificationDataConfig:
    labels: list[str]


@dataclass(slots=True)
class ClassificationModelConfig:
    num_boost_round: int


@dataclass(slots=True)
class ClassificationPredictConfig:
    checkpoint_path: Path | None
    batch_size: int


@dataclass(slots=True)
class ClassificationExperimentConfig:
    data: ClassificationDataConfig
    model: ClassificationModelConfig
    predict: ClassificationPredictConfig
    train_output_dir: Path


def _import_xgboost():
    try:
        import xgboost as xgb
    except Exception as exc:  # pragma: no cover - 运行期环境相关
        raise RuntimeError(
            "当前环境缺少 xgboost 依赖，分类推理无法运行。"
            "请在 models_agent 环境中安装 xgboost。"
        ) from exc
    return xgb


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

    data_raw = raw_config.get("data", {})
    model_raw = raw_config.get("model", {})
    train_raw = raw_config.get("train", {})
    predict_raw = raw_config.get("predict", {})

    data_config = ClassificationDataConfig(
        labels=[str(label) for label in data_raw.get("labels", LABELS)],
    )
    model_config = ClassificationModelConfig(
        num_boost_round=int(model_raw.get("num_boost_round", 2000)),
    )
    train_output_dir = _resolve_path(
        str(
            train_raw.get(
                "output_dir",
                "../models_agent/checkpoints/classification/xgboost",
            )
        ),
        base_dir,
    )
    checkpoint_raw = predict_raw.get("checkpoint_path")
    predict_config = ClassificationPredictConfig(
        checkpoint_path=(
            _resolve_path(str(checkpoint_raw), base_dir) if checkpoint_raw else None
        ),
        batch_size=int(predict_raw.get("batch_size", 128)),
    )

    if not data_config.labels:
        raise ValueError("classification.data.labels 不能为空")
    if len(set(data_config.labels)) != len(data_config.labels):
        raise ValueError("classification.data.labels 不允许重复")
    if model_config.num_boost_round <= 0:
        raise ValueError("classification.model.num_boost_round 必须大于 0")

    return ClassificationExperimentConfig(
        data=data_config,
        model=model_config,
        predict=predict_config,
        train_output_dir=train_output_dir,
    )


def get_checkpoint_path(config_path: Path) -> Path:
    experiment_config = load_config(config_path)
    return experiment_config.predict.checkpoint_path or (
        experiment_config.train_output_dir / "best_model.json"
    )


def load_model(model_path: Path):
    xgb = _import_xgboost()
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster


def _safe_ratio(
    numerator: float,
    denominator: float,
    epsilon: float = 1e-6,
) -> float:
    return float(numerator / max(denominator, epsilon))


def _compute_weekday_features(date_value: str) -> tuple[float, float, float]:
    parsed = pd.to_datetime(date_value)
    if pd.isna(parsed):
        raise ValueError("分类样本需要合法的 date 字段")
    weekday_index = float(parsed.dayofweek)
    weekday_angle = 2.0 * np.pi * weekday_index / 7.0
    return (
        float(np.sin(weekday_angle)),
        float(np.cos(weekday_angle)),
        float(1.0 if int(weekday_index) >= 5 else 0.0),
    )


def build_tabular_feature_vector(
    aggregate_values: np.ndarray,
    date_value: str,
) -> np.ndarray:
    if aggregate_values.shape != (SEQUENCE_LENGTH,):
        raise ValueError(f"aggregate 序列长度必须为 {SEQUENCE_LENGTH}")

    aggregate = aggregate_values.astype(np.float32)
    day_values = aggregate[DAY_START_SLOT:DAY_END_SLOT]
    night_values = np.concatenate([aggregate[:DAY_START_SLOT], aggregate[DAY_END_SLOT:]])
    morning_values = aggregate[24:40]
    daytime_values = aggregate[40:64]
    evening_values = aggregate[64:88]
    overnight_values = np.concatenate([aggregate[:24], aggregate[88:]])
    quantiles = np.quantile(aggregate, [0.10, 0.25, 0.50, 0.75, 0.90]).astype(np.float32)
    ramp = np.diff(aggregate, prepend=aggregate[0])
    ramp_abs = np.abs(ramp)
    ramp_up = ramp[ramp > 0]
    ramp_down = np.abs(ramp[ramp < 0])

    peak_index = int(np.argmax(aggregate))
    valley_index = int(np.argmin(aggregate))
    peak_value = float(aggregate[peak_index])
    valley_value = float(aggregate[valley_index])
    full_mean = float(aggregate.mean())
    full_max = float(aggregate.max())
    full_min = float(aggregate.min())
    full_std = float(aggregate.std())
    p25 = float(quantiles[1])
    p75 = float(quantiles[3])
    weekday_sin, weekday_cos, is_weekend = _compute_weekday_features(date_value)

    block_means = aggregate.reshape(NUM_BLOCKS, BLOCK_SIZE).mean(axis=1).astype(np.float32)
    feature_values = {
        "full_mean": full_mean,
        "full_std": full_std,
        "full_min": full_min,
        "full_max": full_max,
        "full_range": full_max - full_min,
        "load_factor": _safe_ratio(full_mean, full_max),
        "p10": float(quantiles[0]),
        "p25": p25,
        "p50": float(quantiles[2]),
        "p75": p75,
        "p90": float(quantiles[4]),
        "day_mean": float(day_values.mean()),
        "night_mean": float(night_values.mean()),
        "day_std": float(day_values.std()),
        "night_std": float(night_values.std()),
        "day_night_diff": float(day_values.mean() - night_values.mean()),
        "day_night_ratio": _safe_ratio(float(day_values.mean()), float(night_values.mean())),
        "night_day_ratio": _safe_ratio(float(night_values.mean()), float(day_values.mean())),
        "morning_mean": float(morning_values.mean()),
        "daytime_mean": float(daytime_values.mean()),
        "evening_mean": float(evening_values.mean()),
        "overnight_mean": float(overnight_values.mean()),
        "peak_value": peak_value,
        "peak_slot_index_norm": float(peak_index / (len(aggregate) - 1)),
        "valley_value": valley_value,
        "valley_slot_index_norm": float(valley_index / (len(aggregate) - 1)),
        "peak_to_mean_ratio": _safe_ratio(peak_value, full_mean),
        "ramp_abs_mean": float(ramp_abs.mean()),
        "ramp_abs_std": float(ramp_abs.std()),
        "ramp_abs_max": float(ramp_abs.max()),
        "ramp_up_mean": float(ramp_up.mean()) if len(ramp_up) else 0.0,
        "ramp_down_mean": float(ramp_down.mean()) if len(ramp_down) else 0.0,
        "high_load_ratio": float((aggregate > p75).mean()),
        "low_load_ratio": float((aggregate < p25).mean()),
        "weekday_sin": weekday_sin,
        "weekday_cos": weekday_cos,
        "is_weekend": is_weekend,
    }
    for block_index, block_mean in enumerate(block_means):
        feature_values[f"block_mean_{block_index:02d}"] = float(block_mean)

    return np.asarray(
        [feature_values[feature_name] for feature_name in TABULAR_FEATURE_NAMES],
        dtype=np.float32,
    )


def _predict_probabilities(
    booster,
    features: np.ndarray,
    fallback_rounds: int,
) -> np.ndarray:
    xgb = _import_xgboost()
    dmatrix = xgb.DMatrix(features, feature_names=list(TABULAR_FEATURE_NAMES))

    best_iteration = getattr(booster, "best_iteration", None)
    if best_iteration is not None and int(best_iteration) >= 0:
        probabilities = booster.predict(
            dmatrix,
            iteration_range=(0, int(best_iteration) + 1),
        )
    elif fallback_rounds > 0:
        probabilities = booster.predict(
            dmatrix,
            iteration_range=(0, fallback_rounds),
        )
    else:
        probabilities = booster.predict(dmatrix)

    probability_array = np.asarray(probabilities, dtype=np.float32)
    if probability_array.ndim == 1:
        probability_array = probability_array[:, np.newaxis]
    return probability_array


def _probability_field_name(label: str) -> str:
    return f"prob_{label.replace('-', '_')}"


def predict_single_sample(
    sample: dict[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    aggregate = sample.get("aggregate")
    if aggregate is None or len(aggregate) != SEQUENCE_LENGTH:
        raise ValueError(f"aggregate 输入序列长度必须为 {SEQUENCE_LENGTH}")

    date_value = str(sample.get("date", "")).strip()
    if not date_value:
        raise ValueError("分类推理需要提供 date")

    experiment_config = load_config(config_path)
    checkpoint_path = get_checkpoint_path(config_path)
    booster = load_model(checkpoint_path)

    features = build_tabular_feature_vector(
        np.asarray(aggregate, dtype=np.float32),
        date_value=date_value,
    )[np.newaxis, :]
    probabilities = _predict_probabilities(
        booster=booster,
        features=features,
        fallback_rounds=experiment_config.model.num_boost_round,
    )[0]

    label_names = experiment_config.data.labels
    if len(probabilities) != len(label_names):
        raise ValueError(
            f"模型输出类别数与配置 labels 不一致："
            f"output={len(probabilities)} labels={len(label_names)}"
        )

    label_index = int(np.argmax(probabilities))
    probability_map = {
        label_names[index]: float(probabilities[index])
        for index in range(len(label_names))
    }
    result: dict[str, Any] = {
        "predicted_label": label_names[label_index],
        "confidence": float(probabilities[label_index]),
        "probabilities": probability_map,
        "runtime_library": "xgboost",
        "best_iteration": (
            int(getattr(booster, "best_iteration"))
            if getattr(booster, "best_iteration", None) is not None
            else None
        ),
    }
    for label_name, probability in probability_map.items():
        result[_probability_field_name(label_name)] = probability
    return result

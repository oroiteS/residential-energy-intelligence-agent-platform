"""XGBoost 分类数据集与统计特征构造。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import pandas as pd

from classification.xgboost.constants import (
    AGGREGATE_COLUMNS,
    BLOCK_SIZE,
    NUM_BLOCKS,
    TABULAR_FEATURE_NAMES,
)
from data.process.common.constants import DAY_END_SLOT, DAY_START_SLOT


@dataclass(slots=True)
class TabularClassificationSample:
    sample_id: str
    house_id: str
    date: str
    features: np.ndarray
    label_index: int | None = None


@dataclass(slots=True)
class LabelVocabulary:
    label_names: list[str]

    @property
    def label_to_index(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.label_names)}

    @property
    def index_to_label(self) -> dict[int, str]:
        return {index: label for index, label in enumerate(self.label_names)}


def infer_label_vocabulary(data_frame: pd.DataFrame) -> LabelVocabulary:
    if "label_name" not in data_frame.columns:
        raise ValueError("训练数据缺少 label_name 字段，无法推断标签体系")
    label_names = sorted({str(value).strip() for value in data_frame["label_name"].tolist()})
    if not label_names:
        raise ValueError("训练数据中的标签集合为空")
    return LabelVocabulary(label_names=label_names)


def _safe_ratio(numerator: float, denominator: float, epsilon: float = 1e-6) -> float:
    return float(numerator / max(denominator, epsilon))


def _compute_weekday_features(date_value: str) -> tuple[float, float, float]:
    parsed = pd.to_datetime(date_value)
    if pd.isna(parsed):
        raise ValueError("XGBoost 分类样本需要合法的 date 字段")
    weekday_index = float(parsed.dayofweek)
    weekday_angle = 2.0 * np.pi * weekday_index / 7.0
    return (
        float(np.sin(weekday_angle)),
        float(np.cos(weekday_angle)),
        float(1.0 if int(weekday_index) >= 5 else 0.0),
    )


def build_tabular_feature_vector(aggregate_values: np.ndarray, date_value: str) -> np.ndarray:
    if aggregate_values.shape != (len(AGGREGATE_COLUMNS),):
        raise ValueError(f"aggregate 序列长度必须为 {len(AGGREGATE_COLUMNS)}")

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


def _validate_columns(data_frame: pd.DataFrame, require_label: bool) -> None:
    required_columns = {"sample_id", "house_id", "date", *AGGREGATE_COLUMNS}
    if require_label:
        required_columns.add("label_name")
    missing_columns = required_columns.difference(data_frame.columns)
    if missing_columns:
        raise ValueError(f"分类数据缺少必要字段: {sorted(missing_columns)}")


def _build_samples_from_frame(
    data_frame: pd.DataFrame,
    require_label: bool,
    label_vocabulary: LabelVocabulary | None = None,
) -> list[TabularClassificationSample]:
    _validate_columns(data_frame, require_label=require_label)
    label_to_index = label_vocabulary.label_to_index if label_vocabulary is not None else {}
    samples: list[TabularClassificationSample] = []
    for row in data_frame.itertuples(index=False):
        aggregate = np.asarray(
            [getattr(row, column) for column in AGGREGATE_COLUMNS],
            dtype=np.float32,
        )
        date_value = str(row.date)
        label_index: int | None = None
        if require_label:
            label_name = str(row.label_name)
            if label_name not in label_to_index:
                raise ValueError(f"未知分类标签: {label_name}")
            label_index = label_to_index[label_name]
        samples.append(
            TabularClassificationSample(
                sample_id=str(row.sample_id),
                house_id=str(row.house_id),
                date=date_value,
                features=build_tabular_feature_vector(aggregate, date_value=date_value),
                label_index=label_index,
            )
        )
    return samples


def load_training_samples(data_path: Path) -> list[TabularClassificationSample]:
    data_frame = pd.read_csv(data_path)
    label_vocabulary = infer_label_vocabulary(data_frame)
    return _build_samples_from_frame(
        data_frame,
        require_label=True,
        label_vocabulary=label_vocabulary,
    )


def load_prediction_samples(data_frame: pd.DataFrame) -> list[TabularClassificationSample]:
    return _build_samples_from_frame(data_frame, require_label=False)


def split_samples(
    samples: list[TabularClassificationSample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[TabularClassificationSample], list[TabularClassificationSample], list[TabularClassificationSample]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio 必须在 0 和 1 之间")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio 必须在 0 和 1 之间")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio 必须小于 1")
    if len(samples) < 3:
        raise ValueError("样本数过少，无法切分训练/验证/测试集")

    shuffled_samples = samples.copy()
    random.Random(seed).shuffle(shuffled_samples)

    total = len(shuffled_samples)
    train_end = max(1, int(total * train_ratio))
    val_end = max(train_end + 1, int(total * (train_ratio + val_ratio)))
    val_end = min(val_end, total - 1)

    train_samples = shuffled_samples[:train_end]
    val_samples = shuffled_samples[train_end:val_end]
    test_samples = shuffled_samples[val_end:]

    if not val_samples:
        val_samples = [train_samples.pop()]
    if not test_samples:
        test_samples = [val_samples.pop()]

    return train_samples, val_samples, test_samples


def samples_to_xy(
    samples: list[TabularClassificationSample],
) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame]:
    if not samples:
        raise ValueError("样本列表不能为空")
    features = np.stack([sample.features for sample in samples], axis=0).astype(np.float32)
    labels = (
        np.asarray([sample.label_index for sample in samples], dtype=np.int32)
        if samples[0].label_index is not None
        else None
    )
    metadata = pd.DataFrame(
        {
            "sample_id": [sample.sample_id for sample in samples],
            "house_id": [sample.house_id for sample in samples],
            "date": [sample.date for sample in samples],
        }
    )
    return features, labels, metadata

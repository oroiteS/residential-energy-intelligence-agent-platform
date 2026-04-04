"""LSTM 预测数据集加载、归一化与切分。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from forecast.LSTM.constants import ALL_FEATURE_NAMES, INPUT_LENGTH, TARGET_LENGTH


T = TypeVar("T")
AGGREGATE_CHANNEL_INDEX = 0
DEFAULT_NORMALIZATION_MODE = "input_window"
LEGACY_NORMALIZATION_MODE = "global"


def _build_feature_columns(prefix: str, length: int) -> list[str]:
    return [f"{prefix}_{index:03d}" for index in range(length)]


X_AGGREGATE_COLUMNS = _build_feature_columns("x_aggregate", INPUT_LENGTH)
X_ACTIVE_COLUMNS = _build_feature_columns("x_active_count", INPUT_LENGTH)
X_BURST_COLUMNS = _build_feature_columns("x_burst_count", INPUT_LENGTH)
X_SLOT_SIN_COLUMNS = _build_feature_columns("x_slot_sin", INPUT_LENGTH)
X_SLOT_COS_COLUMNS = _build_feature_columns("x_slot_cos", INPUT_LENGTH)
X_WEEKDAY_SIN_COLUMNS = _build_feature_columns("x_weekday_sin", INPUT_LENGTH)
X_WEEKDAY_COS_COLUMNS = _build_feature_columns("x_weekday_cos", INPUT_LENGTH)
Y_AGGREGATE_COLUMNS = _build_feature_columns("y_aggregate", TARGET_LENGTH)
FEATURE_COLUMN_MAP = {
    "aggregate": X_AGGREGATE_COLUMNS,
    "active_appliance_count": X_ACTIVE_COLUMNS,
    "burst_event_count": X_BURST_COLUMNS,
    "slot_sin": X_SLOT_SIN_COLUMNS,
    "slot_cos": X_SLOT_COS_COLUMNS,
    "weekday_sin": X_WEEKDAY_SIN_COLUMNS,
    "weekday_cos": X_WEEKDAY_COS_COLUMNS,
}
FEATURE_PAYLOAD_FIELD_MAP = {
    "aggregate": "aggregate",
    "active_appliance_count": "active_appliance_count",
    "burst_event_count": "burst_event_count",
    "slot_sin": "slot_sin",
    "slot_cos": "slot_cos",
    "weekday_sin": "weekday_sin",
    "weekday_cos": "weekday_cos",
}


def build_temporal_feature_sequences(input_start: str) -> dict[str, list[float]]:
    start_timestamp = pd.to_datetime(input_start)
    if pd.isna(start_timestamp):
        raise ValueError("无法从 input_start 解析时间特征，请提供合法的日期或时间戳")

    timestamps = pd.date_range(
        start=start_timestamp,
        periods=INPUT_LENGTH,
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


@dataclass(slots=True)
class ForecastSample:
    sample_id: str
    house_id: str
    input_start: str
    input_end: str
    target_start: str
    target_end: str
    features: np.ndarray
    target: np.ndarray


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


class ForecastDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """将预测样本转换为可训练的数据集。"""

    def __init__(
        self,
        samples: list[ForecastSample],
        feature_names: list[str],
        normalization: ForecastNormalizationStats | None = None,
        aggregate_mode: str = DEFAULT_NORMALIZATION_MODE,
        aggregate_eps: float = 1e-6,
    ) -> None:
        if not samples:
            raise ValueError("预测数据集不能为空")
        if not feature_names:
            raise ValueError("feature_names 不能为空")
        if feature_names[0] != "aggregate":
            raise ValueError("feature_names 第一个特征必须是 aggregate")
        invalid_feature_names = set(feature_names).difference(ALL_FEATURE_NAMES)
        if invalid_feature_names:
            raise ValueError(f"存在不支持的 feature_names: {sorted(invalid_feature_names)}")

        self.samples = samples
        self.feature_names = feature_names
        raw_features = np.stack(
            [sample.features for sample in samples], axis=0
        ).astype(np.float32)
        raw_targets = np.stack(
            [sample.target for sample in samples], axis=0
        ).astype(np.float32)

        if normalization is None:
            normalization = self._build_training_normalization(
                raw_features=raw_features,
                raw_targets=raw_targets,
                aggregate_mode=aggregate_mode,
                aggregate_eps=aggregate_eps,
            )

        self.normalization = normalization
        (
            self.features,
            self.targets,
            self.denorm_mean,
            self.denorm_std,
        ) = self._apply_normalization(
            raw_features=raw_features,
            raw_targets=raw_targets,
            normalization=normalization,
        )

    @staticmethod
    def _build_training_normalization(
        raw_features: np.ndarray,
        raw_targets: np.ndarray,
        aggregate_mode: str,
        aggregate_eps: float,
    ) -> ForecastNormalizationStats:
        if aggregate_mode == DEFAULT_NORMALIZATION_MODE:
            raw_auxiliary = raw_features[:, :, 1:]
            if raw_auxiliary.shape[-1] == 0:
                auxiliary_mean = np.zeros((1, 1, 0), dtype=np.float32)
                safe_auxiliary_std = np.ones((1, 1, 0), dtype=np.float32)
            else:
                auxiliary_mean = raw_auxiliary.mean(axis=(0, 1), keepdims=True)
                auxiliary_std = raw_auxiliary.std(axis=(0, 1), keepdims=True)
                safe_auxiliary_std = np.where(
                    auxiliary_std < aggregate_eps,
                    1.0,
                    auxiliary_std,
                ).astype(np.float32)
            return ForecastNormalizationStats(
                aggregate_mode=aggregate_mode,
                aggregate_eps=aggregate_eps,
                auxiliary_mean=auxiliary_mean.astype(np.float32),
                auxiliary_std=safe_auxiliary_std,
            )

        if aggregate_mode == LEGACY_NORMALIZATION_MODE:
            feature_mean = raw_features.mean(axis=(0, 1), keepdims=True)
            feature_std = raw_features.std(axis=(0, 1), keepdims=True)
            target_mean = float(raw_targets.mean())
            target_std = float(raw_targets.std())
            safe_feature_std = np.where(
                feature_std < aggregate_eps,
                1.0,
                feature_std,
            ).astype(np.float32)
            safe_target_std = 1.0 if target_std < aggregate_eps else float(target_std)
            return ForecastNormalizationStats(
                aggregate_mode=aggregate_mode,
                aggregate_eps=aggregate_eps,
                feature_mean=feature_mean.astype(np.float32),
                feature_std=safe_feature_std,
                target_mean=target_mean,
                target_std=safe_target_std,
            )

        raise ValueError(f"不支持的 aggregate_mode: {aggregate_mode}")

    @staticmethod
    def _apply_normalization(
        raw_features: np.ndarray,
        raw_targets: np.ndarray,
        normalization: ForecastNormalizationStats,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if normalization.aggregate_mode == DEFAULT_NORMALIZATION_MODE:
            if normalization.auxiliary_mean is None or normalization.auxiliary_std is None:
                raise ValueError("input_window 模式缺少 auxiliary_mean / auxiliary_std")

            aggregate = raw_features[:, :, AGGREGATE_CHANNEL_INDEX]
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
                normalized_features = normalized_aggregate
            else:
                normalized_auxiliary = (
                    raw_auxiliary - normalization.auxiliary_mean
                ) / normalization.auxiliary_std
                normalized_features = np.concatenate(
                    [normalized_aggregate, normalized_auxiliary.astype(np.float32)],
                    axis=-1,
                )
            normalized_targets = (
                (raw_targets - aggregate_mean) / safe_aggregate_std
            ).astype(np.float32)
            return (
                normalized_features,
                normalized_targets,
                aggregate_mean.astype(np.float32),
                safe_aggregate_std.astype(np.float32),
            )

        if normalization.aggregate_mode == LEGACY_NORMALIZATION_MODE:
            if (
                normalization.feature_mean is None
                or normalization.feature_std is None
                or normalization.target_mean is None
                or normalization.target_std is None
            ):
                raise ValueError("global 模式缺少完整的全局归一化参数")

            normalized_features = (
                raw_features - normalization.feature_mean
            ) / normalization.feature_std
            normalized_targets = (
                raw_targets - normalization.target_mean
            ) / normalization.target_std
            denorm_mean = np.full(
                (len(raw_targets), 1),
                float(normalization.target_mean),
                dtype=np.float32,
            )
            denorm_std = np.full(
                (len(raw_targets), 1),
                float(normalization.target_std),
                dtype=np.float32,
            )
            return (
                normalized_features.astype(np.float32),
                normalized_targets.astype(np.float32),
                denorm_mean,
                denorm_std,
            )

        raise ValueError(f"不支持的 aggregate_mode: {normalization.aggregate_mode}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_tensor = torch.from_numpy(self.features[index])
        target_tensor = torch.from_numpy(self.targets[index])
        denorm_mean_tensor = torch.from_numpy(self.denorm_mean[index])
        denorm_std_tensor = torch.from_numpy(self.denorm_std[index])
        return (
            feature_tensor,
            target_tensor,
            denorm_mean_tensor,
            denorm_std_tensor,
        )


def load_forecast_samples(
    data_path: Path,
    feature_names: list[str],
) -> list[ForecastSample]:
    data_frame = pd.read_csv(data_path)
    required_feature_columns = {
        column
        for feature_name in feature_names
        for column in FEATURE_COLUMN_MAP[feature_name]
    }
    required_columns = {
        "sample_id",
        "house_id",
        "input_start",
        "input_end",
        "target_start",
        "target_end",
        *required_feature_columns,
        *Y_AGGREGATE_COLUMNS,
    }
    missing_columns = required_columns.difference(data_frame.columns)
    if missing_columns:
        raise ValueError(f"预测数据缺少必要字段: {sorted(missing_columns)}")

    samples: list[ForecastSample] = []
    for row in data_frame.itertuples(index=False):
        feature_values = [
            np.asarray(
                [getattr(row, column) for column in FEATURE_COLUMN_MAP[feature_name]],
                dtype=np.float32,
            )
            for feature_name in feature_names
        ]
        target_values = np.asarray(
            [getattr(row, column) for column in Y_AGGREGATE_COLUMNS],
            dtype=np.float32,
        )

        features = np.stack(feature_values, axis=-1)
        if features.shape != (INPUT_LENGTH, len(feature_names)):
            raise ValueError(f"预测输入形状不正确: {features.shape}")
        if target_values.shape != (TARGET_LENGTH,):
            raise ValueError(f"预测目标形状不正确: {target_values.shape}")

        samples.append(
            ForecastSample(
                sample_id=str(row.sample_id),
                house_id=str(row.house_id),
                input_start=str(row.input_start),
                input_end=str(row.input_end),
                target_start=str(row.target_start),
                target_end=str(row.target_end),
                features=features,
                target=target_values,
            )
        )
    return samples


def _split_by_ratio(
    values: list[T],
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[T], list[T], list[T]]:
    total = len(values)
    train_end = max(1, int(total * train_ratio))
    val_end = max(train_end + 1, int(total * (train_ratio + val_ratio)))
    val_end = min(val_end, total - 1)

    train_values = values[:train_end]
    val_values = values[train_end:val_end]
    test_values = values[val_end:]

    if not val_values:
        val_values = [train_values.pop()]
    if not test_values:
        test_values = [val_values.pop()]
    return train_values, val_values, test_values


def split_samples(
    samples: list[ForecastSample],
    split_mode: str = "by_house",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[ForecastSample], list[ForecastSample], list[ForecastSample]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio 必须在 0 和 1 之间")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio 必须在 0 和 1 之间")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio 必须小于 1")
    if len(samples) < 3:
        raise ValueError("样本数过少，无法切分训练/验证/测试集")

    rng = random.Random(seed)

    if split_mode == "by_house":
        unique_houses = sorted({sample.house_id for sample in samples})
        if len(unique_houses) < 3:
            raise ValueError("按家庭切分至少需要 3 个不同家庭")

        shuffled_houses = unique_houses.copy()
        rng.shuffle(shuffled_houses)
        train_houses, val_houses, test_houses = _split_by_ratio(
            shuffled_houses, train_ratio=train_ratio, val_ratio=val_ratio
        )
        train_house_set = set(train_houses)
        val_house_set = set(val_houses)
        test_house_set = set(test_houses)
        train_samples = [
            sample for sample in samples if sample.house_id in train_house_set
        ]
        val_samples = [
            sample for sample in samples if sample.house_id in val_house_set
        ]
        test_samples = [
            sample for sample in samples if sample.house_id in test_house_set
        ]
        if not train_samples or not val_samples or not test_samples:
            raise ValueError("按家庭切分失败：训练/验证/测试集中至少有一个为空")
        return train_samples, val_samples, test_samples

    shuffled_samples = samples.copy()
    rng.shuffle(shuffled_samples)
    return _split_by_ratio(
        shuffled_samples, train_ratio=train_ratio, val_ratio=val_ratio
    )

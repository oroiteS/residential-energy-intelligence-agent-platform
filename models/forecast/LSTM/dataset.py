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

from forecast.LSTM.constants import (
    ALL_FEATURE_NAMES,
    INPUT_LENGTH,
    STEPS_PER_DAY,
    TARGET_LENGTH,
)


T = TypeVar("T")
AGGREGATE_CHANNEL_INDEX = 0
DEFAULT_NORMALIZATION_MODE = "input_window"
LEGACY_NORMALIZATION_MODE = "global"
STATS_BATCH_SIZE = 1024


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
FEATURE_STORAGE_ORDER = tuple(ALL_FEATURE_NAMES)
FEATURE_STORAGE_INDEX = {
    feature_name: index for index, feature_name in enumerate(FEATURE_STORAGE_ORDER)
}


def resolve_forecast_array_paths(data_path: Path) -> tuple[Path, Path]:
    stem = data_path.stem
    return (
        data_path.with_name(f"{stem}_features.npy"),
        data_path.with_name(f"{stem}_targets.npy"),
    )


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

    slot_angle = (
        2.0 * np.pi * slot_index.to_numpy(dtype=np.float32) / float(STEPS_PER_DAY)
    )
    weekday_angle = 2.0 * np.pi * weekday_index.to_numpy(dtype=np.float32) / 7.0
    return {
        "slot_sin": np.sin(slot_angle).astype(np.float32).tolist(),
        "slot_cos": np.cos(slot_angle).astype(np.float32).tolist(),
        "weekday_sin": np.sin(weekday_angle).astype(np.float32).tolist(),
        "weekday_cos": np.cos(weekday_angle).astype(np.float32).tolist(),
    }


@dataclass(slots=True)
class ForecastSample:
    row_index: int
    house_id: str


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
    """基于 memmap 懒加载预测样本，避免整表常驻内存。"""

    def __init__(
        self,
        samples: list[ForecastSample],
        data_path: Path,
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
            raise ValueError(
                f"存在不支持的 feature_names: {sorted(invalid_feature_names)}"
            )

        self.samples = samples
        self.data_path = data_path
        self.feature_names = feature_names
        self.aggregate_mode = aggregate_mode
        self.aggregate_eps = aggregate_eps
        self.row_indices = np.asarray(
            [sample.row_index for sample in samples], dtype=np.int64
        )
        self.feature_channel_indices = np.asarray(
            [FEATURE_STORAGE_INDEX[feature_name] for feature_name in feature_names],
            dtype=np.int64,
        )
        self.auxiliary_channel_indices = self.feature_channel_indices[1:]
        self.feature_store, self.target_store = self._load_array_stores(data_path)

        if normalization is None:
            normalization = self._build_training_normalization(
                aggregate_mode=aggregate_mode,
                aggregate_eps=aggregate_eps,
            )
        self.normalization = normalization

    def _load_array_stores(
        self,
        data_path: Path,
    ) -> tuple[np.memmap, np.memmap]:
        feature_path, target_path = resolve_forecast_array_paths(data_path)
        if not feature_path.exists() or not target_path.exists():
            raise FileNotFoundError(
                "未找到预测数组文件，请先重新执行 build-forecast 生成 .npy 数据集"
            )

        feature_store = np.load(feature_path, mmap_mode="r")
        target_store = np.load(target_path, mmap_mode="r")
        if feature_store.ndim != 3:
            raise ValueError(f"预测特征数组维度错误: {feature_store.shape}")
        if target_store.ndim != 2:
            raise ValueError(f"预测目标数组维度错误: {target_store.shape}")
        if feature_store.shape[1:] != (INPUT_LENGTH, len(FEATURE_STORAGE_ORDER)):
            raise ValueError(f"预测特征数组形状错误: {feature_store.shape}")
        if target_store.shape[1] != TARGET_LENGTH:
            raise ValueError(f"预测目标数组形状错误: {target_store.shape}")
        if feature_store.shape[0] != target_store.shape[0]:
            raise ValueError(
                "预测特征数组与目标数组样本数不一致: "
                f"{feature_store.shape[0]} != {target_store.shape[0]}"
            )
        max_row_index = int(self.row_indices.max())
        if max_row_index >= feature_store.shape[0]:
            raise ValueError(
                f"样本索引越界: 最大 row_index={max_row_index}, "
                f"数组样本数={feature_store.shape[0]}"
            )
        return feature_store, target_store

    @staticmethod
    def _iter_batches(
        row_indices: np.ndarray,
        batch_size: int = STATS_BATCH_SIZE,
    ) -> list[np.ndarray]:
        return [
            row_indices[start : start + batch_size]
            for start in range(0, len(row_indices), batch_size)
        ]

    def _load_feature_batch(
        self,
        row_indices: np.ndarray,
        channel_indices: np.ndarray,
        dtype: np.dtype = np.float32,
    ) -> np.ndarray:
        if channel_indices.size == 0:
            return np.empty(
                (len(row_indices), INPUT_LENGTH, 0),
                dtype=dtype,
            )
        raw_batch = np.asarray(self.feature_store[row_indices], dtype=dtype)
        return np.take(raw_batch, channel_indices, axis=-1)

    def _build_training_normalization(
        self,
        aggregate_mode: str,
        aggregate_eps: float,
    ) -> ForecastNormalizationStats:
        if aggregate_mode == DEFAULT_NORMALIZATION_MODE:
            if self.auxiliary_channel_indices.size == 0:
                auxiliary_mean = np.zeros((1, 1, 0), dtype=np.float32)
                auxiliary_std = np.ones((1, 1, 0), dtype=np.float32)
                return ForecastNormalizationStats(
                    aggregate_mode=aggregate_mode,
                    aggregate_eps=aggregate_eps,
                    auxiliary_mean=auxiliary_mean,
                    auxiliary_std=auxiliary_std,
                )

            total_points = 0
            auxiliary_sum = np.zeros(
                (len(self.feature_names) - 1,),
                dtype=np.float64,
            )
            auxiliary_square_sum = np.zeros_like(auxiliary_sum)
            for batch_indices in self._iter_batches(self.row_indices):
                raw_auxiliary = self._load_feature_batch(
                    batch_indices,
                    self.auxiliary_channel_indices,
                    dtype=np.float64,
                )
                auxiliary_sum += raw_auxiliary.sum(axis=(0, 1))
                auxiliary_square_sum += np.square(raw_auxiliary).sum(axis=(0, 1))
                total_points += raw_auxiliary.shape[0] * raw_auxiliary.shape[1]

            auxiliary_mean = auxiliary_sum / float(total_points)
            auxiliary_var = (
                auxiliary_square_sum / float(total_points) - np.square(auxiliary_mean)
            )
            auxiliary_std = np.sqrt(np.maximum(auxiliary_var, 0.0))
            safe_auxiliary_std = np.where(
                auxiliary_std < aggregate_eps,
                1.0,
                auxiliary_std,
            ).astype(np.float32)
            return ForecastNormalizationStats(
                aggregate_mode=aggregate_mode,
                aggregate_eps=aggregate_eps,
                auxiliary_mean=auxiliary_mean.reshape(1, 1, -1).astype(np.float32),
                auxiliary_std=safe_auxiliary_std.reshape(1, 1, -1),
            )

        if aggregate_mode == LEGACY_NORMALIZATION_MODE:
            total_feature_points = 0
            feature_sum = np.zeros((len(self.feature_names),), dtype=np.float64)
            feature_square_sum = np.zeros_like(feature_sum)
            target_sum = 0.0
            target_square_sum = 0.0
            total_target_points = 0

            for batch_indices in self._iter_batches(self.row_indices):
                raw_features = self._load_feature_batch(
                    batch_indices,
                    self.feature_channel_indices,
                    dtype=np.float64,
                )
                raw_targets = np.asarray(
                    self.target_store[batch_indices],
                    dtype=np.float64,
                )
                feature_sum += raw_features.sum(axis=(0, 1))
                feature_square_sum += np.square(raw_features).sum(axis=(0, 1))
                total_feature_points += raw_features.shape[0] * raw_features.shape[1]
                target_sum += float(raw_targets.sum())
                target_square_sum += float(np.square(raw_targets).sum())
                total_target_points += raw_targets.size

            feature_mean = feature_sum / float(total_feature_points)
            feature_var = (
                feature_square_sum / float(total_feature_points) - np.square(feature_mean)
            )
            feature_std = np.sqrt(np.maximum(feature_var, 0.0))
            safe_feature_std = np.where(
                feature_std < aggregate_eps,
                1.0,
                feature_std,
            ).astype(np.float32)
            target_mean = target_sum / float(total_target_points)
            target_var = (
                target_square_sum / float(total_target_points) - target_mean**2
            )
            target_std = float(np.sqrt(max(target_var, 0.0)))
            safe_target_std = 1.0 if target_std < aggregate_eps else target_std
            return ForecastNormalizationStats(
                aggregate_mode=aggregate_mode,
                aggregate_eps=aggregate_eps,
                feature_mean=feature_mean.reshape(1, 1, -1).astype(np.float32),
                feature_std=safe_feature_std.reshape(1, 1, -1),
                target_mean=float(target_mean),
                target_std=float(safe_target_std),
            )

        raise ValueError(f"不支持的 aggregate_mode: {aggregate_mode}")

    def __len__(self) -> int:
        return len(self.samples)

    def _normalize_input_window(
        self,
        raw_features: np.ndarray,
        raw_targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.normalization.auxiliary_mean is None or self.normalization.auxiliary_std is None:
            raise ValueError("input_window 模式缺少 auxiliary_mean / auxiliary_std")

        aggregate = raw_features[:, AGGREGATE_CHANNEL_INDEX].astype(np.float32)
        aggregate_mean_value = float(aggregate.mean())
        aggregate_std_value = float(aggregate.std())
        safe_aggregate_std_value = (
            1.0
            if aggregate_std_value < self.normalization.aggregate_eps
            else aggregate_std_value
        )

        normalized_aggregate = (
            (aggregate - aggregate_mean_value) / safe_aggregate_std_value
        ).astype(np.float32)[:, np.newaxis]
        if raw_features.shape[1] == 1:
            normalized_features = normalized_aggregate
        else:
            normalized_auxiliary = (
                raw_features[:, 1:] - self.normalization.auxiliary_mean[0, 0, :]
            ) / self.normalization.auxiliary_std[0, 0, :]
            normalized_features = np.concatenate(
                [normalized_aggregate, normalized_auxiliary.astype(np.float32)],
                axis=-1,
            )
        normalized_targets = (
            (raw_targets - aggregate_mean_value) / safe_aggregate_std_value
        ).astype(np.float32)
        denorm_mean = np.asarray([aggregate_mean_value], dtype=np.float32)
        denorm_std = np.asarray([safe_aggregate_std_value], dtype=np.float32)
        return normalized_features, normalized_targets, denorm_mean, denorm_std

    def _normalize_global(
        self,
        raw_features: np.ndarray,
        raw_targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if (
            self.normalization.feature_mean is None
            or self.normalization.feature_std is None
            or self.normalization.target_mean is None
            or self.normalization.target_std is None
        ):
            raise ValueError("global 模式缺少完整的全局归一化参数")

        normalized_features = (
            raw_features - self.normalization.feature_mean[0, 0, :]
        ) / self.normalization.feature_std[0, 0, :]
        normalized_targets = (
            raw_targets - self.normalization.target_mean
        ) / self.normalization.target_std
        denorm_mean = np.asarray([self.normalization.target_mean], dtype=np.float32)
        denorm_std = np.asarray([self.normalization.target_std], dtype=np.float32)
        return (
            normalized_features.astype(np.float32),
            normalized_targets.astype(np.float32),
            denorm_mean,
            denorm_std,
        )

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row_index = int(self.row_indices[index])
        raw_features = np.take(
            np.asarray(self.feature_store[row_index], dtype=np.float32),
            self.feature_channel_indices,
            axis=-1,
        )
        raw_targets = np.asarray(self.target_store[row_index], dtype=np.float32)

        if self.normalization.aggregate_mode == DEFAULT_NORMALIZATION_MODE:
            (
                normalized_features,
                normalized_targets,
                denorm_mean,
                denorm_std,
            ) = self._normalize_input_window(raw_features, raw_targets)
        elif self.normalization.aggregate_mode == LEGACY_NORMALIZATION_MODE:
            (
                normalized_features,
                normalized_targets,
                denorm_mean,
                denorm_std,
            ) = self._normalize_global(raw_features, raw_targets)
        else:
            raise ValueError(
                f"不支持的 aggregate_mode: {self.normalization.aggregate_mode}"
            )

        return (
            torch.from_numpy(normalized_features),
            torch.from_numpy(normalized_targets),
            torch.from_numpy(denorm_mean),
            torch.from_numpy(denorm_std),
        )


def load_forecast_samples(
    data_path: Path,
    feature_names: list[str],
) -> list[ForecastSample]:
    if not feature_names:
        raise ValueError("feature_names 不能为空")
    invalid_feature_names = set(feature_names).difference(ALL_FEATURE_NAMES)
    if invalid_feature_names:
        raise ValueError(f"存在不支持的 feature_names: {sorted(invalid_feature_names)}")

    feature_path, target_path = resolve_forecast_array_paths(data_path)
    if not feature_path.exists() or not target_path.exists():
        raise FileNotFoundError(
            "未找到预测数组文件，请先重新执行 build-forecast 生成 .npy 数据集"
        )

    data_frame = pd.read_csv(data_path, usecols=["row_index", "house_id"])
    required_columns = {"row_index", "house_id"}
    missing_columns = required_columns.difference(data_frame.columns)
    if missing_columns:
        raise ValueError(
            "预测样本元数据缺少必要字段，请重新执行 build-forecast："
            f"{sorted(missing_columns)}"
        )

    samples: list[ForecastSample] = []
    for row in data_frame.itertuples(index=False):
        samples.append(
            ForecastSample(
                row_index=int(row.row_index),
                house_id=str(row.house_id),
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

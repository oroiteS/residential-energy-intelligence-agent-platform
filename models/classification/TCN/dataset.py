"""分类数据集加载与切分。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from classification.TCN.constants import INPUT_CHANNELS, LABELS, SEQUENCE_LENGTH

LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}


def _build_feature_columns(prefix: str, length: int) -> list[str]:
    return [f"{prefix}_{index:03d}" for index in range(length)]


AGGREGATE_COLUMNS = _build_feature_columns("aggregate", SEQUENCE_LENGTH)
ACTIVE_COLUMNS = _build_feature_columns("active_count", SEQUENCE_LENGTH)
BURST_COLUMNS = _build_feature_columns("burst_count", SEQUENCE_LENGTH)


@dataclass(slots=True)
class ClassificationSample:
    sample_id: str
    house_id: str
    date: str
    features: np.ndarray
    label_index: int


class ClassificationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """将日级分类样本转换为可训练的数据集。"""

    def __init__(
        self,
        samples: list[ClassificationSample],
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> None:
        if not samples:
            raise ValueError("分类数据集不能为空")

        self.samples = samples
        raw_features = np.stack([sample.features for sample in samples], axis=0).astype(np.float32)

        if mean is None:
            mean = raw_features.mean(axis=(0, 1), keepdims=True)
        if std is None:
            std = raw_features.std(axis=(0, 1), keepdims=True)

        safe_std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        self.mean = mean.astype(np.float32)
        self.std = safe_std
        self.features = (raw_features - self.mean) / self.std
        self.labels = np.array([sample.label_index for sample in samples], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        feature_tensor = torch.from_numpy(self.features[index])
        label_tensor = torch.tensor(self.labels[index], dtype=torch.long)
        return feature_tensor, label_tensor


def load_classification_samples(data_path: Path) -> list[ClassificationSample]:
    data_frame = pd.read_csv(data_path)

    required_columns = {
        "sample_id",
        "house_id",
        "date",
        "label_name",
        *AGGREGATE_COLUMNS,
        *ACTIVE_COLUMNS,
        *BURST_COLUMNS,
    }
    missing_columns = required_columns.difference(data_frame.columns)
    if missing_columns:
        raise ValueError(f"分类数据缺少必要字段: {sorted(missing_columns)}")

    samples: list[ClassificationSample] = []
    for row in data_frame.itertuples(index=False):
        label_name = str(row.label_name)
        if label_name not in LABEL_TO_INDEX:
            raise ValueError(f"未知分类标签: {label_name}")

        aggregate_values = np.asarray([getattr(row, column) for column in AGGREGATE_COLUMNS], dtype=np.float32)
        active_values = np.asarray([getattr(row, column) for column in ACTIVE_COLUMNS], dtype=np.float32)
        burst_values = np.asarray([getattr(row, column) for column in BURST_COLUMNS], dtype=np.float32)

        features = np.stack(
            [aggregate_values, active_values, burst_values],
            axis=-1,
        )
        if features.shape != (SEQUENCE_LENGTH, INPUT_CHANNELS):
            raise ValueError(f"样本形状不正确: {features.shape}")

        samples.append(
            ClassificationSample(
                sample_id=str(row.sample_id),
                house_id=str(row.house_id),
                date=str(row.date),
                features=features,
                label_index=LABEL_TO_INDEX[label_name],
            )
        )
    return samples


def split_samples(
    samples: list[ClassificationSample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[ClassificationSample], list[ClassificationSample], list[ClassificationSample]]:
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

"""KMeans 聚类数据加载与特征处理。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.process.classification.constants import AGGREGATE_COLUMNS


def load_day_feature_frame(data_path: Path) -> pd.DataFrame:
    data_frame = pd.read_csv(data_path)
    required_columns = {"sample_id", "house_id", "date", *AGGREGATE_COLUMNS}
    missing_columns = required_columns.difference(data_frame.columns)
    if missing_columns:
        raise ValueError(f"聚类输入缺少必要字段: {sorted(missing_columns)}")
    return data_frame


def extract_raw_curves(data_frame: pd.DataFrame) -> np.ndarray:
    return data_frame.loc[:, AGGREGATE_COLUMNS].to_numpy(dtype=np.float32)


def normalize_curves(curves: np.ndarray, normalization_mode: str) -> np.ndarray:
    if normalization_mode == "none":
        return curves.astype(np.float32)

    if normalization_mode == "sample_zscore":
        mean = curves.mean(axis=1, keepdims=True)
        std = curves.std(axis=1, keepdims=True)
        safe_std = np.where(std < 1e-6, 1.0, std)
        return ((curves - mean) / safe_std).astype(np.float32)

    if normalization_mode == "sample_minmax":
        min_value = curves.min(axis=1, keepdims=True)
        max_value = curves.max(axis=1, keepdims=True)
        scale = np.where((max_value - min_value) < 1e-6, 1.0, max_value - min_value)
        return ((curves - min_value) / scale).astype(np.float32)

    if normalization_mode == "day_mean":
        mean = curves.mean(axis=1, keepdims=True)
        safe_mean = np.where(np.abs(mean) < 1e-6, 1.0, mean)
        return (curves / safe_mean).astype(np.float32)

    raise ValueError(f"不支持的 normalization_mode: {normalization_mode}")

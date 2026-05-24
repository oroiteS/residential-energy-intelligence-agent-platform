"""KMeans 聚类数据读取。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from .config import FEATURE_COLUMNS
except ImportError:  # 兼容直接运行
    from config import FEATURE_COLUMNS


def load_features(features_path: Path) -> pd.DataFrame:
    """读取分类任务窗口特征表并校验 16 个行为特征。"""

    df = pd.read_csv(features_path, encoding="utf-8")
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"特征列缺失：{missing}")
    return df

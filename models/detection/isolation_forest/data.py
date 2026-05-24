"""Isolation Forest 异常检测数据读取。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from .config import FEATURE_COLUMNS
except ImportError:  # 兼容 `python detection/isolation_forest/data.py`
    from config import FEATURE_COLUMNS


def load_features(features_path: Path) -> pd.DataFrame:
    """读取异常检测窗口特征表。

    输入文件由 `data/detection/preprocess_detection.py` 生成，必须包含
    用户、窗口边界以及固定的 16 个行为特征。这里集中做列校验，保证
    后续模型训练阶段只处理已经满足格式约束的数据。
    """

    df = pd.read_csv(features_path, encoding="utf-8")
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"特征列缺失：{missing}")
    return df

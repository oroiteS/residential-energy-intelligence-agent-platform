"""XGBoost 分类模型数据读取与用户级切分。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

try:
    from .config import FEATURE_COLUMNS
except ImportError:  # 兼容直接运行
    from config import FEATURE_COLUMNS


def load_data(features_path: Path, labels_path: Path) -> pd.DataFrame:
    """合并窗口特征和 KMeans 产生的聚类标签。

    两个文件通过 user_id、window_start、window_end 对齐，
    确保分类器学习的是同一个 7 天窗口的特征和簇标签。
    """

    features = pd.read_csv(features_path, encoding="utf-8")
    labels = pd.read_csv(labels_path, encoding="utf-8")
    missing = [column for column in FEATURE_COLUMNS if column not in features.columns]
    if missing:
        raise ValueError(f"特征列缺失：{missing}")

    # cluster 是 KMeans 阶段输出的无监督类别编号。
    # XGBoost 训练阶段把它当作监督学习标签。
    return features.merge(
        labels[["user_id", "window_start", "window_end", "cluster"]],
        on=["user_id", "window_start", "window_end"],
    )


def split_by_user_majority_label(
    *,
    df: pd.DataFrame,
    test_ratio: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """按用户主标签分层划分训练/测试集合。

    同一用户的多个滑动窗口不能同时进入训练集和测试集，否则分类器会
    记住用户特征，导致测试指标高估。
    """

    # 每个用户可能拥有多个窗口和多个聚类标签。
    # 这里用该用户出现最多的标签作为用户级分层标签。
    user_labels = (
        df.groupby("user_id")["cluster"]
        .agg(lambda values: values.value_counts().idxmax())
        .reset_index()
    )
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed)
    user_train_idx, user_test_idx = next(splitter.split(user_labels[["user_id"]], user_labels["cluster"]))
    train_users = set(user_labels.iloc[user_train_idx]["user_id"])
    test_users = set(user_labels.iloc[user_test_idx]["user_id"])

    # 再把用户级划分映射回窗口级样本索引。
    train_idx = np.flatnonzero(df["user_id"].isin(train_users).to_numpy())
    test_idx = np.flatnonzero(df["user_id"].isin(test_users).to_numpy())
    if train_users & test_users:
        raise RuntimeError("用户级训练/测试划分失败：存在用户泄漏")
    return train_idx, test_idx

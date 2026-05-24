"""XGBoost 预测模型数据读取、特征选择和数据切分。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb


@dataclass(frozen=True)
class SplitData:
    """训练、验证、测试三份数据。"""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """读取监督学习样本表。

    该表由 data/forecast/preprocess_forecast.py 生成，
    默认是一行表示“过去 30 天历史 -> 未来 7 天目标”的一个样本。
    """

    return pd.read_csv(dataset_path)


def time_split(df: pd.DataFrame, date_column: str, validation_ratio: float, test_ratio: float) -> SplitData:
    """按预测开始日期切分训练、验证、测试集。

    时间序列任务不能随机切分，否则未来日期可能泄漏到训练集。
    这里按 forecast_start_date 排序后，前段训练、中段验证、后段测试。
    """

    data = df.copy()
    data[date_column] = pd.to_datetime(data[date_column])
    unique_dates = sorted(data[date_column].dropna().unique())
    if len(unique_dates) < 3:
        raise ValueError("可用于时间切分的日期数量太少")

    test_size = max(1, int(len(unique_dates) * test_ratio))
    valid_size = max(1, int(len(unique_dates) * validation_ratio))
    train_size = len(unique_dates) - valid_size - test_size
    if train_size <= 0:
        raise ValueError("训练集日期数量不足，请调小验证集或测试集比例")

    valid_start = unique_dates[train_size]
    test_start = unique_dates[train_size + valid_size]
    train = data[data[date_column] < valid_start].copy()
    valid = data[(data[date_column] >= valid_start) & (data[date_column] < test_start)].copy()
    test = data[data[date_column] >= test_start].copy()
    return SplitData(train=train, valid=valid, test=test)


def user_split(
    df: pd.DataFrame,
    user_column: str,
    validation_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> SplitData:
    """按用户划分训练/验证/测试，避免同一用户滑动窗口泄漏到不同集合。"""

    if user_column not in df.columns:
        raise ValueError(f"用户级切分需要字段 {user_column}")

    rng = np.random.default_rng(random_seed)
    users = np.array(sorted(df[user_column].dropna().unique()))
    if len(users) < 3:
        raise ValueError("可用于用户级切分的用户数量太少")
    rng.shuffle(users)

    test_size = max(1, int(len(users) * test_ratio))
    valid_size = max(1, int(len(users) * validation_ratio))
    train_size = len(users) - valid_size - test_size
    if train_size <= 0:
        raise ValueError("训练集用户数量不足，请调小验证集或测试集比例")

    train_users = set(users[:train_size])
    valid_users = set(users[train_size : train_size + valid_size])
    test_users = set(users[train_size + valid_size :])
    if train_users & valid_users or train_users & test_users or valid_users & test_users:
        raise RuntimeError("用户级训练/验证/测试划分失败：存在用户泄漏")

    return SplitData(
        train=df[df[user_column].isin(train_users)].copy(),
        valid=df[df[user_column].isin(valid_users)].copy(),
        test=df[df[user_column].isin(test_users)].copy(),
    )


def select_feature_columns(
    df: pd.DataFrame,
    *,
    meta_columns: list[str],
    target_columns: list[str],
    feature_config: dict[str, Any] | None = None,
) -> list[str]:
    """根据配置选择可由用户上传数据计算得到的特征列。

    meta_columns 是样本身份字段，target_columns 是预测目标；
    它们都不能作为输入特征，否则会造成目标泄漏或无意义记忆。
    """

    excluded = set(meta_columns) | set(target_columns)
    if feature_config:
        include_prefixes = tuple(feature_config.get("include_prefixes", []))
        include_columns = list(feature_config.get("include_columns", []))
        exclude_columns = set(feature_config.get("exclude_columns", []))
        prefixed_columns = [
            column
            for column in df.columns
            if include_prefixes and column.startswith(include_prefixes)
        ]
        feature_columns = prefixed_columns + include_columns
        feature_columns = [
            column
            for column in dict.fromkeys(feature_columns)
            if column not in excluded and column not in exclude_columns
        ]
        missing_columns = [column for column in feature_columns if column not in df.columns]
        if missing_columns:
            raise ValueError(f"配置中的特征列不存在：{missing_columns}")
    else:
        feature_columns = [column for column in df.columns if column not in excluded]

    non_numeric = [column for column in feature_columns if not pd.api.types.is_numeric_dtype(df[column])]
    if non_numeric:
        raise ValueError(f"特征列中存在非数值字段，请先处理：{non_numeric}")
    return feature_columns


def make_split(df: pd.DataFrame, config: dict[str, Any]) -> tuple[SplitData, str]:
    """按配置创建训练、验证、测试切分。

    支持 time 和 user 两种策略：
    - time 更符合真实未来预测；
    - user 更适合评估模型对未见用户的泛化能力。
    """

    split_config = config["split"]
    data_config = config["data"]
    train_config = config["training"]
    split_strategy = split_config.get("strategy", "user")
    if split_strategy == "time":
        split = time_split(
            df,
            date_column=data_config["date_column"],
            validation_ratio=float(split_config["validation_ratio"]),
            test_ratio=float(split_config["test_ratio"]),
        )
    elif split_strategy == "user":
        split = user_split(
            df,
            user_column="user_id",
            validation_ratio=float(split_config["validation_ratio"]),
            test_ratio=float(split_config["test_ratio"]),
            random_seed=int(train_config["random_seed"]),
        )
    else:
        raise ValueError(f"不支持的切分策略：{split_strategy}")
    return split, split_strategy


def to_dmatrix(frame: pd.DataFrame, feature_columns: list[str], target_column: str | None = None) -> xgb.DMatrix:
    """把 Pandas 数据转换为 XGBoost DMatrix。

    DMatrix 是 XGBoost 的高效内部数据结构，会保存特征矩阵、可选标签和特征名。
    """

    label = frame[target_column].to_numpy(dtype=float) if target_column else None
    return xgb.DMatrix(frame[feature_columns].to_numpy(dtype=float), label=label, feature_names=feature_columns)

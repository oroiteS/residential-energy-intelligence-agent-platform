"""LSTM 直接预测任务的数据处理函数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitData:
    """按时间划分后的训练、验证、测试数据表。"""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class FeatureSpec:
    """模型输入特征列定义。

    `sequence_columns` 按历史天数组织成二维列名列表；`future_columns`
    表示未来 7 天的日历特征；`static_columns` 表示历史窗口统计特征。
    """

    sequence_columns: list[list[str]]
    future_columns: list[str]
    static_columns: list[str]


def time_split(df: pd.DataFrame, date_column: str, validation_ratio: float, test_ratio: float) -> SplitData:
    """按预测开始日期切分训练、验证、测试集。

    神经网络训练也按时间切分，避免未来时间窗口进入训练集造成评估偏乐观。
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
    return SplitData(
        train=data[data[date_column] < valid_start].copy(),
        valid=data[(data[date_column] >= valid_start) & (data[date_column] < test_start)].copy(),
        test=data[data[date_column] >= test_start].copy(),
    )


def build_feature_spec(config: dict[str, Any]) -> FeatureSpec:
    """根据特征配置生成模型实际读取的列名。

    sequence_columns 形状是 input_days × 每日特征数；
    future_columns 是未来 7 天的日历特征展平结果；
    static_columns 是历史窗口统计特征。
    """

    input_days = int(config["input_days"])
    sequence_columns = [
        [template.format(day=day) for template in config["sequence_feature_templates"]]
        for day in range(1, input_days + 1)
    ]
    future_days = 7
    future_columns = [
        template.format(day=day)
        for day in range(1, future_days + 1)
        for template in config["future_feature_templates"]
    ]
    return FeatureSpec(
        sequence_columns=sequence_columns,
        future_columns=future_columns,
        static_columns=list(config["static_columns"]),
    )


def validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """校验样本表是否包含模型所需列。"""

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"缺少字段：{missing}")


def build_raw_arrays(
    frame: pd.DataFrame,
    spec: FeatureSpec,
    target_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """把数据表转换为序列特征、未来特征、静态特征和真实目标数组。

    输出形状：
    - sequence: 样本数 × 30 天 × 每日特征数；
    - future: 样本数 × 未来日历特征数；
    - static: 样本数 × 历史统计特征数；
    - actual: 样本数 × 21 个预测目标。
    """

    sequence = np.stack(
        [frame[day_columns].to_numpy(dtype=float) for day_columns in spec.sequence_columns],
        axis=1,
    )
    future = frame[spec.future_columns].to_numpy(dtype=float)
    static = frame[spec.static_columns].to_numpy(dtype=float)
    actual = frame[target_columns].to_numpy(dtype=float)
    return sequence, future, static, actual

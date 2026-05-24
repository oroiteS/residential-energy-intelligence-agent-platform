"""Transformer 残差 baseline 任务的数据处理函数。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    from .config import resolve_path
except ImportError:  # 兼容直接运行
    from config import resolve_path


@dataclass(frozen=True)
class SplitData:
    """按时间划分后的训练、验证、测试数据表。"""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class FeatureSpec:
    """模型输入特征列定义。"""

    sequence_columns: list[list[str]]
    future_columns: list[str]
    static_columns: list[str]


def time_split(df: pd.DataFrame, date_column: str, validation_ratio: float, test_ratio: float) -> SplitData:
    """按预测开始日期切分训练、验证、测试集。"""

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
    """根据特征配置生成模型实际读取的列名。"""

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


def read_xgboost_feature_columns(feature_file: Path) -> list[str]:
    """读取 XGBoost baseline 训练时保存的特征列清单。"""

    return [line.strip() for line in feature_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def predict_xgboost_baseline(
    frame: pd.DataFrame,
    *,
    model_dir: Path,
    feature_columns: list[str],
    xgboost_targets: list[str],
) -> np.ndarray:
    """用 7 个 XGBoost 模型预测 7 天总用电量，返回形状为 `(n, 7)` 的数组。"""

    validate_columns(frame, feature_columns)
    dmatrix = xgb.DMatrix(
        frame[feature_columns].to_numpy(dtype=float),
        feature_names=feature_columns,
    )
    predictions = []
    for target in xgboost_targets:
        model_path = model_dir / f"{target}.json"
        if not model_path.exists():
            raise FileNotFoundError(f"缺少 XGBoost baseline 模型：{model_path}")
        booster = xgb.Booster()
        booster.load_model(model_path)
        predictions.append(booster.predict(dmatrix))
    return np.stack(predictions, axis=1)


def get_baseline_predictions(
    split: SplitData,
    *,
    baseline_config: dict[str, Any],
    xgboost_targets: list[str],
) -> dict[str, np.ndarray]:
    """获取训练、验证、测试三份数据对应的 XGBoost 总用电 baseline。"""

    if not baseline_config.get("enabled", False):
        return {
            "train": np.zeros((len(split.train), len(xgboost_targets)), dtype=float),
            "valid": np.zeros((len(split.valid), len(xgboost_targets)), dtype=float),
            "test": np.zeros((len(split.test), len(xgboost_targets)), dtype=float),
        }

    cache_path = resolve_path(baseline_config["cache_file"])
    use_cache = bool(baseline_config.get("use_cache", True))
    if use_cache and cache_path.exists():
        loaded = np.load(cache_path)
        return {key: loaded[key] for key in ["train", "valid", "test"]}

    model_dir = resolve_path(baseline_config["xgboost_model_dir"])
    feature_file = resolve_path(baseline_config["xgboost_feature_file"])
    feature_columns = read_xgboost_feature_columns(feature_file)
    predictions = {
        "train": predict_xgboost_baseline(
            split.train,
            model_dir=model_dir,
            feature_columns=feature_columns,
            xgboost_targets=xgboost_targets,
        ),
        "valid": predict_xgboost_baseline(
            split.valid,
            model_dir=model_dir,
            feature_columns=feature_columns,
            xgboost_targets=xgboost_targets,
        ),
        "test": predict_xgboost_baseline(
            split.test,
            model_dir=model_dir,
            feature_columns=feature_columns,
            xgboost_targets=xgboost_targets,
        ),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as file:
        np.savez_compressed(file, train=predictions["train"], valid=predictions["valid"], test=predictions["test"])
    return predictions


def expand_baseline_to_21(xgboost_total: np.ndarray, hist_peak_ratio: np.ndarray) -> np.ndarray:
    """将 7 维 XGBoost 总用电 baseline 用 ratio 扩展为 21 维（总、峰、谷）。"""

    ratio = hist_peak_ratio.reshape(-1, 1)
    peak_baseline = xgboost_total * ratio
    valley_baseline = xgboost_total * (1.0 - ratio)
    return np.concatenate([xgboost_total, peak_baseline, valley_baseline], axis=1)


def build_raw_arrays(
    frame: pd.DataFrame,
    spec: FeatureSpec,
    target_columns: list[str],
    xgboost_baseline: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构建残差学习所需数组。

    返回顺序为 `(sequence, future, static, residual_21, actual_21, baseline_21)`。
    """

    sequence = np.stack(
        [frame[day_columns].to_numpy(dtype=float) for day_columns in spec.sequence_columns],
        axis=1,
    )
    future = frame[spec.future_columns].to_numpy(dtype=float)
    static = frame[spec.static_columns].to_numpy(dtype=float)

    hist_peak_ratio = frame["hist_peak_ratio"].to_numpy(dtype=float)
    baseline_21 = expand_baseline_to_21(xgboost_baseline, hist_peak_ratio)

    actual = frame[target_columns].to_numpy(dtype=float)
    residual = actual - baseline_21
    return sequence, future, static, residual, actual, baseline_21

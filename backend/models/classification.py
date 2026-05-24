from __future__ import annotations

import pickle
from functools import lru_cache
from typing import Any, Protocol, Sequence, cast

import numpy as np
import pandas as pd
import xgboost as xgb

from models.common import ARTIFACTS_DIR


# 分类模型的固定特征顺序。
# 训练和推理必须使用同一顺序，否则 XGBoost 输入含义会错位。
FEATURE_COLUMNS = [
    "avg_energy",
    "std_energy",
    "max_energy",
    "min_energy",
    "avg_peak",
    "avg_valley",
    "peak_valley_ratio",
    "peak_ratio",
    "valley_ratio",
    "load_factor",
    "workday_avg",
    "weekend_avg",
    "weekend_workday_ratio",
    "trend_rel",
    "volatility",
    "med_mean_ratio",
]


class LabelEncoderLike(Protocol):
    """分类标签编码器的最小协议。

    这里只声明推理阶段实际用到的属性和方法，避免代码依赖训练脚本的完整类型。
    """

    classes_: Any

    def inverse_transform(self, y: Sequence[int]) -> Any: ...


@lru_cache(maxsize=1)
def _load_classifier() -> tuple[xgb.XGBClassifier, LabelEncoderLike]:
    """加载 XGBoost 分类模型和标签编码器。

    使用 lru_cache 保证后端进程内只加载一次模型产物，
    避免每次接口请求都重复读取磁盘文件。
    """

    artifact_dir = ARTIFACTS_DIR / "classification" / "xgboost"
    model_path = artifact_dir / "xgboost_model.json"
    encoder_path = artifact_dir / "label_encoder.pkl"

    # label_encoder 用于把模型输出的类别编号还原为中文/业务标签。
    with encoder_path.open("rb") as file:
        label_encoder = cast(LabelEncoderLike, pickle.load(file))

    model = xgb.XGBClassifier(
        n_jobs=1,
    )
    model.load_model(model_path)
    return model, label_encoder


def classify_daily_window(window_rows: Sequence[dict]) -> dict:
    """对一个日级窗口进行用电行为分类。

    输入窗口由若干天的 total_kwh、peak_kwh、valley_kwh 组成；
    输出包括预测标签、置信度、各类别概率、解释文本和特征快照。
    """

    if not window_rows:
        raise ValueError("分类窗口不能为空")

    # 先把业务窗口转换为模型特征，再按 FEATURE_COLUMNS 固定顺序组装 DataFrame。
    features = extract_window_features(window_rows)
    model, label_encoder = _load_classifier()
    frame = pd.DataFrame([{column: features[column] for column in FEATURE_COLUMNS}])

    # predict_proba 返回每个类别的概率。
    # 最大概率对应最终标签，confidence 直接取该标签的概率。
    probabilities_array = model.predict_proba(frame)[0]
    pred_index = int(np.argmax(probabilities_array))
    label = str(label_encoder.inverse_transform([pred_index])[0])
    class_names = [str(item) for item in label_encoder.classes_]
    probabilities = {
        class_name: round(float(probabilities_array[index]), 4)
        for index, class_name in enumerate(class_names)
    }
    confidence = probabilities[label]

    return {
        "predicted_label": label,
        "confidence": confidence,
        "probabilities": probabilities,
        "explanation": _build_explanation(label, features),
        "feature_snapshot": {
            key: round(float(value), 4)
            for key, value in features.items()
        },
    }


def extract_window_features(window_rows: Sequence[dict]) -> dict[str, float]:
    """从日级用电窗口提取分类/检测共用特征。

    特征覆盖总量、峰谷结构、工作日/周末差异、趋势、波动和负荷均衡度；
    异常检测也复用这套特征，保证两个模型的解释口径一致。
    """

    # 三个核心序列分别表示每日总用电、峰时用电和谷时用电。
    energies = np.array([float(item["total_kwh"]) for item in window_rows], dtype=np.float64)
    peaks = np.array([float(item["peak_kwh"]) for item in window_rows], dtype=np.float64)
    valleys = np.array([float(item["valley_kwh"]) for item in window_rows], dtype=np.float64)
    dates = [item["date"] for item in window_rows]

    avg_e = float(energies.mean())
    std_e = float(energies.std())
    max_e = float(energies.max())
    min_e = float(energies.min())
    avg_peak = float(peaks.mean())
    avg_valley = float(valleys.mean())
    total_peak = float(peaks.sum())
    total_valley = float(valleys.sum())
    total_e = float(energies.sum())

    # 工作日/周末均值用于刻画作息差异。
    # 如果窗口里缺少某类日期，则回退到全窗口平均值，避免空数组产生 NaN。
    weekdays = np.array([date.weekday() for date in dates])
    workday_values = energies[weekdays < 5]
    weekend_values = energies[weekdays >= 5]
    workday_avg = float(workday_values.mean()) if len(workday_values) else avg_e
    weekend_avg = float(weekend_values.mean()) if len(weekend_values) else avg_e

    # 使用简单线性斜率描述窗口内总用电趋势。
    # trend_rel 会除以均值，减少不同总量级用户之间的尺度差异。
    n = len(energies)
    x_mean = (n - 1) / 2
    numerator = float(sum((i - x_mean) * (energies[i] - avg_e) for i in range(n)))
    denominator = float(sum((i - x_mean) ** 2 for i in range(n)))
    slope = numerator / (denominator + 1e-9)
    day_diffs = np.abs(np.diff(energies))

    return {
        "avg_energy": avg_e,
        "std_energy": std_e,
        "max_energy": max_e,
        "min_energy": min_e,
        "avg_peak": avg_peak,
        "avg_valley": avg_valley,
        "peak_valley_ratio": total_peak / (total_valley + 1e-6),
        "peak_ratio": total_peak / (total_e + 1e-6),
        "valley_ratio": total_valley / (total_e + 1e-6),
        "load_factor": avg_e / (max_e + 1e-6),
        "workday_avg": workday_avg,
        "weekend_avg": weekend_avg,
        "weekend_workday_ratio": weekend_avg / (workday_avg + 1e-6),
        "trend_rel": slope / (avg_e + 1e-6),
        "volatility": float(day_diffs.mean()) / (avg_e + 1e-6) if len(day_diffs) else 0.0,
        "med_mean_ratio": float(np.median(energies)) / (avg_e + 1e-6),
    }


def _build_explanation(label: str, features: dict[str, float]) -> str:
    """根据预测标签和关键特征生成简短解释。"""

    return (
        f"模型判定为{label}。"
        f"窗口日均用电 {features['avg_energy']:.2f} kWh，"
        f"峰时占比 {features['peak_ratio']:.1%}，"
        f"波动强度 {features['volatility']:.2f}。"
    )

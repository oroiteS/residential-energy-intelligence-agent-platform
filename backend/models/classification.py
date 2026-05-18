from __future__ import annotations

import pickle
from functools import lru_cache
from typing import Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

from models.common import ARTIFACTS_DIR


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


@lru_cache(maxsize=1)
def _load_classifier() -> tuple[xgb.XGBClassifier, object]:
    artifact_dir = ARTIFACTS_DIR / "classification" / "xgboost"
    model_path = artifact_dir / "xgboost_model.json"
    encoder_path = artifact_dir / "label_encoder.pkl"

    with encoder_path.open("rb") as file:
        label_encoder = pickle.load(file)

    model = xgb.XGBClassifier(
        n_jobs=1,
    )
    model.load_model(model_path)
    return model, label_encoder


def classify_daily_window(window_rows: Sequence[dict]) -> dict:
    if not window_rows:
        raise ValueError("分类窗口不能为空")

    features = extract_window_features(window_rows)
    model, label_encoder = _load_classifier()
    frame = pd.DataFrame([{column: features[column] for column in FEATURE_COLUMNS}])
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

    weekdays = np.array([date.weekday() for date in dates])
    workday_values = energies[weekdays < 5]
    weekend_values = energies[weekdays >= 5]
    workday_avg = float(workday_values.mean()) if len(workday_values) else avg_e
    weekend_avg = float(weekend_values.mean()) if len(weekend_values) else avg_e

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
    return (
        f"模型判定为{label}。"
        f"窗口日均用电 {features['avg_energy']:.2f} kWh，"
        f"峰时占比 {features['peak_ratio']:.1%}，"
        f"波动强度 {features['volatility']:.2f}。"
    )

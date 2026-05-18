from __future__ import annotations

import json
import math
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from functools import lru_cache

import numpy as np
import torch

from models.common import ARTIFACTS_DIR
from models.forecast_postprocess import classify_future_window, detect_future_window
from models.lstm_model import LSTMDirectForecaster


# macOS / CPU 环境下，PyTorch 的 MKL-DNN 后端在本项目的 LSTM 推理链路中会触发段错误。
# 这里在模型模块加载时直接关闭，确保后端预测接口稳定可用。
torch.backends.mkldnn.enabled = False
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


TARGET_COLUMNS = [
    *[f"y_energy_d{day:02d}" for day in range(1, 8)],
    *[f"y_peak_d{day:02d}" for day in range(1, 8)],
    *[f"y_valley_d{day:02d}" for day in range(1, 8)],
]


@lru_cache(maxsize=1)
def _load_lstm_artifacts() -> tuple[LSTMDirectForecaster, dict, dict]:
    artifact_dir = ARTIFACTS_DIR / "forecast" / "lstm"
    ckpt_path = artifact_dir / "checkpoints" / "best.ckpt"
    feature_spec = json.loads((artifact_dir / "feature_columns.json").read_text(encoding="utf-8"))
    scalers = dict(np.load(artifact_dir / "input_scalers.npz"))
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = checkpoint["hyper_parameters"]
    model = LSTMDirectForecaster(
        sequence_feature_size=int(hparams["sequence_feature_size"]),
        future_feature_size=int(hparams["future_feature_size"]),
        static_feature_size=int(hparams["static_feature_size"]),
        output_size=int(hparams["output_size"]),
        model_config=hparams["model_config"],
        target_mean=hparams["target_mean"],
        target_scale=hparams["target_scale"],
    )
    state_dict = {
        key.replace("model.", "", 1): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    model.model.load_state_dict(state_dict)
    model.eval()
    return model, feature_spec, scalers


def forecast_daily_series(
    history_rows: Sequence[dict],
    *,
    forecast_start: date,
    horizon_days: int,
) -> dict:
    if len(history_rows) < 30:
        raise ValueError("LSTM 预测至少需要 30 天历史日级数据")
    if horizon_days != 7:
        raise ValueError("当前 LSTM Direct 模型固定预测未来 7 天")

    model, _feature_spec, scalers = _load_lstm_artifacts()
    history = list(history_rows)[-30:]
    future_dates = [forecast_start + timedelta(days=index) for index in range(7)]

    raw_sequence = np.array([_sequence_features(item) for item in history], dtype=np.float32)
    raw_future = np.array([value for day in future_dates for value in _calendar_features(day)], dtype=np.float32)
    raw_static = np.array(_static_features(history), dtype=np.float32)

    sequence = _scale(raw_sequence, scalers["sequence_mean"], scalers["sequence_scale"])
    future = _scale(raw_future, scalers["future_mean"], scalers["future_scale"])
    static = _scale(raw_static, scalers["static_mean"], scalers["static_scale"])

    with torch.no_grad():
        prediction = model.predict_final(
            torch.from_numpy(sequence[None, :, :].astype(np.float32)),
            torch.from_numpy(future[None, :].astype(np.float32)),
            torch.from_numpy(static[None, :].astype(np.float32)),
        )[0].cpu().numpy()

    prediction = np.maximum(prediction, 0.0)
    total_values = prediction[0:7]
    peak_values = prediction[7:14]
    valley_values = prediction[14:21]
    series = []
    for index, target_date in enumerate(future_dates):
        total = float(total_values[index])
        peak = min(float(peak_values[index]), total)
        valley = min(float(valley_values[index]), max(total - peak, 0.0))
        series.append(
            {
                "date": target_date,
                "total_kwh": round(total, 4),
                "peak_kwh": round(peak, 4),
                "valley_kwh": round(valley, 4),
            }
        )

    classification = classify_future_window(series)
    future_detection = detect_future_window(series, history[-7:])

    risk_flags: list[str] = []
    if classification["predicted_label"] == "峰时集中型":
        risk_flags.append("peak_usage_risk")
    if classification["predicted_label"] == "高耗持续型":
        risk_flags.append("high_baseload")
    if future_detection["is_anomaly"]:
        risk_flags.append("abnormal_rise")

    total_sum = sum(item["total_kwh"] for item in series)
    peak_sum = sum(item["peak_kwh"] for item in series)
    valley_sum = sum(item["valley_kwh"] for item in series)

    return {
        "series": [
            {
                "date": item["date"].isoformat(),
                "predicted_total_kwh": item["total_kwh"],
                "predicted_peak_kwh": item["peak_kwh"],
                "predicted_valley_kwh": item["valley_kwh"],
            }
            for item in series
        ],
        "summary": {
            "forecast_start": datetime.combine(series[0]["date"], datetime.min.time()).isoformat(),
            "forecast_end": datetime.combine(series[-1]["date"], datetime.min.time()).isoformat(),
            "granularity": "daily",
            "schema_version": "v1",
            "forecast_horizon": "7d",
            "model_type": "lstm",
            "predicted_total_kwh": round(total_sum, 4),
            "predicted_peak_kwh": round(peak_sum, 4),
            "predicted_valley_kwh": round(valley_sum, 4),
            "predicted_avg_daily_kwh": round(total_sum / len(series), 4),
            "predicted_peak_ratio": round(peak_sum / max(total_sum, 1e-6), 4),
            "predicted_valley_ratio": round(valley_sum / max(total_sum, 1e-6), 4),
            "risk_flags": risk_flags,
            "forecast_classification": {
                "schema_version": "v1",
                "model_type": "xgboost",
                "predicted_label": classification["predicted_label"],
                "label_display_name": classification["predicted_label"],
                "confidence": classification["confidence"],
                "probabilities": classification["probabilities"],
                "window_start": datetime.combine(series[0]["date"], datetime.min.time()).isoformat(),
                "window_end": datetime.combine(series[-1]["date"], datetime.min.time()).isoformat(),
                "source": "lstm_direct_forecast_window",
            },
            "future_detection": future_detection,
            "confidence_hint": "medium",
        },
        "classification": classification,
        "detection": future_detection,
    }


def _scale(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (values - mean.astype(np.float32)) / np.maximum(scale.astype(np.float32), 1e-6)


def _sequence_features(row: dict) -> list[float]:
    day = row["date"]
    weekday_sin, weekday_cos, is_weekend = _calendar_features(day)
    return [
        float(row["total_kwh"]),
        float(row["peak_kwh"]),
        float(row["valley_kwh"]),
        weekday_sin,
        weekday_cos,
        is_weekend,
    ]


def _calendar_features(day: date) -> tuple[float, float, float]:
    weekday = day.weekday()
    return (
        math.sin(2 * math.pi * weekday / 7.0),
        math.cos(2 * math.pi * weekday / 7.0),
        1.0 if weekday >= 5 else 0.0,
    )


def _static_features(history: Sequence[dict]) -> list[float]:
    total = np.array([float(item["total_kwh"]) for item in history], dtype=np.float64)
    peak = np.array([float(item["peak_kwh"]) for item in history], dtype=np.float64)
    valley = np.array([float(item["valley_kwh"]) for item in history], dtype=np.float64)
    weekdays = np.array([item["date"].weekday() for item in history])

    return [
        _mean(total[-7:]),
        _mean(total[-14:]),
        _mean(total[-30:]),
        _std(total[-7:]),
        _std(total[-30:]),
        float(total[-30:].min()),
        float(total[-30:].max()),
        float(total[-1]),
        _trend(total[-7:]),
        _mean(total[weekdays >= 5]),
        _mean(total[weekdays < 5]),
        _mean(total[weekdays >= 5]) - _mean(total[weekdays < 5]),
        float(peak.sum() / max(total.sum(), 1e-6)),
        _mean(peak[-7:]),
        _mean(peak[-30:]),
        _std(peak[-7:]),
        _std(peak[-30:]),
        float(peak[-30:].min()),
        float(peak[-30:].max()),
        float(peak[-1]),
        _trend(peak[-7:]),
        _mean(valley[-7:]),
        _mean(valley[-30:]),
        _std(valley[-7:]),
        _std(valley[-30:]),
        float(valley[-30:].min()),
        float(valley[-30:].max()),
        float(valley[-1]),
        _trend(valley[-7:]),
    ]


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else 0.0


def _std(values: np.ndarray) -> float:
    return float(values.std()) if len(values) else 0.0


def _trend(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    y = values.astype(np.float64)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    numerator = float(((x - x_mean) * (y - y_mean)).sum())
    denominator = float(((x - x_mean) ** 2).sum())
    return numerator / (denominator + 1e-9) / (y_mean + 1e-6)

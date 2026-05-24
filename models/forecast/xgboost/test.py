"""XGBoost 预测模型指标计算与结果写出。"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根误差，单位与原始 kWh 一致。"""

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方误差。"""

    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均绝对误差。"""

    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 MAPE；真实值全为 0 时返回 NaN。"""

    mask = np.abs(y_true) > 1e-8
    if not np.any(mask):
        return math.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_peak_valley_baseline(
    df_test: pd.DataFrame,
    target_columns: list[str],
) -> dict[str, Any]:
    """用历史峰时占比推导峰/谷预测，计算逐天 baseline 误差。

    XGBoost 只预测未来总量时，峰/谷无法直接得到；
    这里用 hist_peak_ratio 做一个可解释的峰谷拆分基线。
    """

    output_days = len(target_columns)
    peak_ratio = df_test["hist_peak_ratio"].to_numpy(dtype=float)

    per_day: list[dict[str, Any]] = []
    for day in range(1, output_days + 1):
        total_pred = df_test[f"y_energy_d{day:02d}"].to_numpy(dtype=float)
        peak_pred = total_pred * peak_ratio
        valley_pred = total_pred * (1.0 - peak_ratio)

        peak_true = df_test[f"y_peak_d{day:02d}"].to_numpy(dtype=float)
        valley_true = df_test[f"y_valley_d{day:02d}"].to_numpy(dtype=float)

        peak_mse_val = mse(peak_true, peak_pred)
        valley_mse_val = mse(valley_true, valley_pred)
        per_day.append(
            {
                "target": f"d{day:02d}",
                "peak_mse": round(peak_mse_val, 6),
                "peak_rmse": round(math.sqrt(peak_mse_val), 6),
                "peak_mae": round(mae(peak_true, peak_pred), 6),
                "peak_mape": round(mape(peak_true, peak_pred), 4),
                "valley_mse": round(valley_mse_val, 6),
                "valley_rmse": round(math.sqrt(valley_mse_val), 6),
                "valley_mae": round(mae(valley_true, valley_pred), 6),
                "valley_mape": round(mape(valley_true, valley_pred), 4),
            }
        )

    return {
        "method": "hist_peak_ratio_baseline",
        "per_day": per_day,
        "peak_avg_rmse": round(float(np.mean([day["peak_rmse"] for day in per_day])), 6),
        "peak_avg_mae": round(float(np.mean([day["peak_mae"] for day in per_day])), 6),
        "peak_avg_mape": round(float(np.mean([day["peak_mape"] for day in per_day])), 4),
        "valley_avg_rmse": round(float(np.mean([day["valley_rmse"] for day in per_day])), 6),
        "valley_avg_mae": round(float(np.mean([day["valley_mae"] for day in per_day])), 6),
        "valley_avg_mape": round(float(np.mean([day["valley_mape"] for day in per_day])), 4),
    }


def write_metrics(metrics: list[dict[str, Any]], metrics_path: Path) -> None:
    """同时写出 CSV 和 JSON 格式的逐目标指标。"""

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    with metrics_path.with_suffix(".json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)


def write_feature_columns(feature_columns: list[str], feature_path: Path) -> None:
    """保存训练实际使用的特征列，便于部署和论文复核。"""

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.write_text("\n".join(feature_columns) + "\n", encoding="utf-8")


def write_baseline_report(baseline: dict[str, Any], baseline_path: Path) -> None:
    """保存并打印峰谷 baseline 指标。"""

    with baseline_path.open("w", encoding="utf-8") as file:
        json.dump(baseline, file, ensure_ascii=False, indent=2)

    print("\n峰谷 baseline 指标（hist_peak_ratio 推导）：")
    header = (
        f"{'day':<6} {'peak_rmse':>10} {'peak_mae':>10} {'peak_mape':>10} "
        f"{'valley_rmse':>12} {'valley_mae':>11} {'valley_mape':>12}"
    )
    print(header)
    for day in baseline["per_day"]:
        print(
            f"{day['target']:<6} {day['peak_rmse']:>10.4f} {day['peak_mae']:>10.4f} {day['peak_mape']:>9.2f}%"
            f" {day['valley_rmse']:>12.4f} {day['valley_mae']:>11.4f} {day['valley_mape']:>11.2f}%"
        )
    print(
        f"{'avg':<6} {baseline['peak_avg_rmse']:>10.4f} {baseline['peak_avg_mae']:>10.4f} "
        f"{baseline['peak_avg_mape']:>9.2f}% {baseline['valley_avg_rmse']:>12.4f} "
        f"{baseline['valley_avg_mae']:>11.4f} {baseline['valley_avg_mape']:>11.2f}%"
    )

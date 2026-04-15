"""预测模型测试可视化工具。"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def compute_horizon_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """按预测步长计算 MAE 和 RMSE。"""

    errors = predictions - targets
    horizon_mae = np.mean(np.abs(errors), axis=0)
    horizon_rmse = np.sqrt(np.mean(np.square(errors), axis=0))
    return horizon_mae.astype(np.float32), horizon_rmse.astype(np.float32)


def select_case_indices(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, int]:
    """选择最佳、中位和最差样本用于个例可视化。"""

    sample_rmse = np.sqrt(
        np.mean(np.square(predictions - targets), axis=1)
    )
    sorted_indices = np.argsort(sample_rmse)
    median_position = len(sorted_indices) // 2
    return {
        "best": int(sorted_indices[0]),
        "median": int(sorted_indices[median_position]),
        "worst": int(sorted_indices[-1]),
    }


def plot_test_diagnostics(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_dir: Path,
    prefix: str = "test",
    scatter_sample_size: int = 10000,
) -> dict[str, str]:
    """绘制测试阶段常见诊断图。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_dir / f"{prefix}_diagnostics.png"
    case_path = output_dir / f"{prefix}_cases.png"

    horizon_mae, horizon_rmse = compute_horizon_metrics(predictions, targets)
    flattened_pred = predictions.reshape(-1)
    flattened_true = targets.reshape(-1)
    flattened_error = flattened_pred - flattened_true

    if len(flattened_pred) > scatter_sample_size:
        sample_indices = np.linspace(
            0,
            len(flattened_pred) - 1,
            num=scatter_sample_size,
            dtype=int,
        )
        scatter_true = flattened_true[sample_indices]
        scatter_pred = flattened_pred[sample_indices]
    else:
        scatter_true = flattened_true
        scatter_pred = flattened_pred

    figure, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    horizon_axis = np.arange(1, predictions.shape[1] + 1)

    axes[0].plot(horizon_axis, horizon_mae, label="MAE", linewidth=2)
    axes[0].plot(horizon_axis, horizon_rmse, label="RMSE", linewidth=2)
    axes[0].set_title("Horizon-wise Error")
    axes[0].set_xlabel("Forecast Horizon (15min step)")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].scatter(
        scatter_true,
        scatter_pred,
        s=8,
        alpha=0.15,
        edgecolors="none",
    )
    min_value = float(min(scatter_true.min(), scatter_pred.min()))
    max_value = float(max(scatter_true.max(), scatter_pred.max()))
    axes[1].plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        linewidth=1.5,
        color="black",
    )
    axes[1].set_title("Prediction vs Ground Truth")
    axes[1].set_xlabel("Ground Truth")
    axes[1].set_ylabel("Prediction")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    axes[2].hist(flattened_error, bins=80, color="#377eb8", alpha=0.8)
    axes[2].axvline(0.0, color="black", linestyle="--", linewidth=1.5)
    axes[2].set_title("Residual Distribution")
    axes[2].set_xlabel("Prediction - Ground Truth")
    axes[2].grid(True, linestyle="--", alpha=0.3)

    mean_true_curve = np.mean(targets, axis=0)
    mean_pred_curve = np.mean(predictions, axis=0)
    axes[3].plot(horizon_axis, mean_true_curve, label="ground_truth", linewidth=2)
    axes[3].plot(horizon_axis, mean_pred_curve, label="prediction", linewidth=2)
    axes[3].set_title("Mean Daily Forecast Curve")
    axes[3].set_xlabel("Forecast Horizon (15min step)")
    axes[3].grid(True, linestyle="--", alpha=0.4)
    axes[3].legend()

    figure.tight_layout()
    figure.savefig(diagnostics_path, dpi=160)
    plt.close(figure)

    case_indices = select_case_indices(predictions, targets)
    case_titles = [
        ("best", "Best Sample"),
        ("median", "Median Sample"),
        ("worst", "Worst Sample"),
    ]
    case_figure, case_axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    for axis, (case_key, case_title) in zip(case_axes, case_titles):
        sample_index = case_indices[case_key]
        sample_rmse = float(
            np.sqrt(
                np.mean(
                    np.square(predictions[sample_index] - targets[sample_index])
                )
            )
        )
        axis.plot(
            horizon_axis,
            targets[sample_index],
            label="ground_truth",
            linewidth=2,
        )
        axis.plot(
            horizon_axis,
            predictions[sample_index],
            label="prediction",
            linewidth=2,
        )
        axis.set_title(f"{case_title} (sample={sample_index}, rmse={sample_rmse:.2f})")
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend()
    case_axes[-1].set_xlabel("Forecast Horizon (15min step)")
    case_figure.tight_layout()
    case_figure.savefig(case_path, dpi=160)
    plt.close(case_figure)

    return {
        "diagnostics": str(diagnostics_path),
        "cases": str(case_path),
    }

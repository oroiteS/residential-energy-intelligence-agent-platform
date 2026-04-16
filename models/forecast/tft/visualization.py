"""TFT 训练与测试可视化。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def plot_training_history(
    metrics_csv_path: Path,
    output_dir: Path,
) -> list[Path]:
    if not metrics_csv_path.exists():
        raise FileNotFoundError(f"未找到训练日志文件: {metrics_csv_path}")

    metrics_df = pd.read_csv(metrics_csv_path)
    if metrics_df.empty or "epoch" not in metrics_df.columns:
        raise ValueError(f"训练日志为空或缺少 epoch 字段: {metrics_csv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    created_paths: list[Path] = []

    epoch_df = (
        metrics_df.sort_values("epoch")
        .groupby("epoch", as_index=False)
        .last()
    )
    metric_groups = [
        ("loss", "Loss", ["train_loss_epoch", "val_loss"]),
        ("mae", "MAE", ["train_mae_epoch", "val_mae", "val_baseline_mae"]),
        ("rmse", "RMSE", ["train_rmse_epoch", "val_rmse", "val_baseline_rmse"]),
        ("diff_mae", "Diff MAE", ["train_diff_mae_epoch", "val_diff_mae"]),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    for axis, (_, title, metric_names) in zip(axes.flatten(), metric_groups, strict=False):
        for metric_name in metric_names:
            if metric_name in epoch_df.columns:
                valid_df = epoch_df[["epoch", metric_name]].dropna()
                if not valid_df.empty:
                    axis.plot(
                        valid_df["epoch"],
                        valid_df[metric_name],
                        marker="o",
                        linewidth=2,
                        label=metric_name,
                    )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend()
    figure.tight_layout()
    combined_path = output_dir / "training_curves.png"
    figure.savefig(combined_path, dpi=160)
    plt.close(figure)
    created_paths.append(combined_path)

    step_metrics = ["train_loss_step", "train_mae_step"]
    step_df = metrics_df[["step", *[name for name in step_metrics if name in metrics_df.columns]]].copy()
    if len(step_df.columns) > 1:
        figure, axis = plt.subplots(figsize=(12, 6))
        for metric_name in step_df.columns:
            if metric_name == "step":
                continue
            valid_df = step_df[["step", metric_name]].dropna()
            if not valid_df.empty:
                axis.plot(
                    valid_df["step"],
                    valid_df[metric_name],
                    linewidth=1.5,
                    label=metric_name,
                    alpha=0.9,
                )
        axis.set_title("Step Metrics")
        axis.set_xlabel("Step")
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend()
        figure.tight_layout()
        step_path = output_dir / "training_step_curves.png"
        figure.savefig(step_path, dpi=160)
        plt.close(figure)
        created_paths.append(step_path)

    return created_paths


def summarize_model_parameters(
    model: torch.nn.Module,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    parameter_rows: list[dict[str, Any]] = []
    total_params = 0
    trainable_params = 0
    for name, parameter in model.named_parameters():
        param_count = int(parameter.numel())
        total_params += param_count
        if parameter.requires_grad:
            trainable_params += param_count
        data = parameter.detach().float().cpu()
        parameter_rows.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "count": param_count,
                "requires_grad": bool(parameter.requires_grad),
                "mean": float(data.mean().item()),
                "std": float(data.std().item()) if data.numel() > 1 else 0.0,
                "min": float(data.min().item()),
                "max": float(data.max().item()),
                "l2_norm": float(torch.linalg.vector_norm(data).item()),
            }
        )

    summary = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": total_params - trainable_params,
        "parameter_count": len(parameter_rows),
        "parameters": parameter_rows,
    }
    summary_path = output_dir / "parameter_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def plot_parameter_distributions(
    model: torch.nn.Module,
    output_dir: Path,
    top_k: int = 12,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created_paths: list[Path] = []
    parameter_stats = []
    flattened_weights: list[np.ndarray] = []
    for name, parameter in model.named_parameters():
        if parameter.numel() == 0:
            continue
        values = parameter.detach().float().cpu().numpy().reshape(-1)
        flattened_weights.append(values)
        parameter_stats.append(
            {
                "name": name,
                "count": int(parameter.numel()),
                "l2_norm": float(np.linalg.norm(values)),
            }
        )

    if flattened_weights:
        figure, axis = plt.subplots(figsize=(12, 6))
        concatenated = np.concatenate(flattened_weights, axis=0)
        axis.hist(concatenated, bins=100, color="#287271", alpha=0.85)
        axis.set_title("All Parameter Distribution")
        axis.set_xlabel("Parameter Value")
        axis.set_ylabel("Frequency")
        axis.grid(True, linestyle="--", alpha=0.3)
        figure.tight_layout()
        histogram_path = output_dir / "parameter_histogram.png"
        figure.savefig(histogram_path, dpi=160)
        plt.close(figure)
        created_paths.append(histogram_path)

    if parameter_stats:
        top_items = sorted(parameter_stats, key=lambda item: item["l2_norm"], reverse=True)[:top_k]
        figure, axis = plt.subplots(figsize=(14, 8))
        axis.barh(
            [item["name"] for item in reversed(top_items)],
            [item["l2_norm"] for item in reversed(top_items)],
            color="#8ab17d",
        )
        axis.set_title(f"Top {len(top_items)} Parameter L2 Norms")
        axis.set_xlabel("L2 Norm")
        axis.grid(True, axis="x", linestyle="--", alpha=0.3)
        figure.tight_layout()
        norm_path = output_dir / "parameter_norms.png"
        figure.savefig(norm_path, dpi=160)
        plt.close(figure)
        created_paths.append(norm_path)
    return created_paths


def plot_test_household_metrics(
    per_house_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if per_house_df.empty:
        return []

    created_paths: list[Path] = []
    sorted_df = per_house_df.sort_values("mae_improvement", ascending=False).reset_index(drop=True)

    figure, axis = plt.subplots(figsize=(14, 8))
    x_positions = np.arange(len(sorted_df))
    axis.bar(x_positions - 0.2, sorted_df["baseline_mae"], width=0.4, label="baseline_mae", color="#e76f51")
    axis.bar(x_positions + 0.2, sorted_df["model_mae"], width=0.4, label="model_mae", color="#2a9d8f")
    axis.set_title("Per-Household MAE Comparison")
    axis.set_xlabel("Household (sorted by improvement)")
    axis.set_ylabel("MAE")
    axis.set_xticks([])
    axis.legend()
    axis.grid(True, axis="y", linestyle="--", alpha=0.3)
    figure.tight_layout()
    compare_path = output_dir / "household_mae_comparison.png"
    figure.savefig(compare_path, dpi=160)
    plt.close(figure)
    created_paths.append(compare_path)

    figure, axis = plt.subplots(figsize=(14, 6))
    axis.bar(x_positions, sorted_df["mae_improvement"], color="#264653")
    axis.axhline(0.0, color="gray", linestyle="--", linewidth=1.2)
    axis.set_title("Per-Household MAE Improvement (baseline - model)")
    axis.set_xlabel("Household (sorted)")
    axis.set_ylabel("MAE Improvement")
    axis.set_xticks([])
    axis.grid(True, axis="y", linestyle="--", alpha=0.3)
    figure.tight_layout()
    improvement_path = output_dir / "household_mae_improvement.png"
    figure.savefig(improvement_path, dpi=160)
    plt.close(figure)
    created_paths.append(improvement_path)

    return created_paths


def plot_prediction_examples(
    prediction_examples: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not prediction_examples:
        return []

    created_paths: list[Path] = []
    json_path = output_dir / "prediction_examples.json"
    json_path.write_text(
        json.dumps(prediction_examples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    created_paths.append(json_path)

    max_examples = min(len(prediction_examples), 12)
    columns = 2
    rows = int(np.ceil(max_examples / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(14, 4 * rows), sharex=True)
    flat_axes = np.atleast_1d(axes).reshape(-1)
    for axis in flat_axes[max_examples:]:
        axis.axis("off")

    for axis, example in zip(flat_axes, prediction_examples[:max_examples], strict=False):
        slots = np.arange(len(example["target"]))
        axis.plot(slots, example["target"], label="target", linewidth=2, color="#264653")
        axis.plot(slots, example["prediction"], label="prediction", linewidth=2, color="#2a9d8f")
        axis.plot(slots, example["baseline"], label="baseline", linewidth=1.5, color="#e76f51", alpha=0.9)
        axis.set_title(
            f"{example['house_id']} | {example['target_start']}\n"
            f"MAE={example['model_mae']:.2f}, baseline={example['baseline_mae']:.2f}"
        )
        axis.set_xlabel("15min Slot")
        axis.set_ylabel("Aggregate")
        axis.grid(True, linestyle="--", alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    plot_path = output_dir / "prediction_examples.png"
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    created_paths.append(plot_path)
    return created_paths


"""五模型 21 维预测结果对比主图生成。

energy（总）/ peak（峰）/ valley（谷）三类目标，每类 7 天。
XGBoost 仅输出 energy 指标，神经网络模型输出全部三类。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "forecast" / "comparison" / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"

OKABE_ITO = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]

TARGET_TYPE_LABELS = {
    "total": "Total (总用电)",
    "peak": "Peak (峰时)",
    "valley": "Valley (谷时)",
}


@dataclass(frozen=True)
class ModelSpec:
    """一个参与对比的模型配置。

    metrics_path 指向该模型测试阶段输出的指标文件；
    kind 用于区分 XGBoost 指标格式和神经网络指标格式。
    """

    key: str
    label: str
    metrics_path: Path
    kind: str
    color: str


MODEL_SPECS = [
    ModelSpec(
        key="xgboost",
        label="XGBoost",
        metrics_path=PROJECT_ROOT / "forecast" / "xgboost" / "output" / "metrics.csv",
        kind="xgboost",
        color=OKABE_ITO[4],
    ),
    ModelSpec(
        key="lstm_baseline",
        label="LSTM Baseline",
        metrics_path=PROJECT_ROOT / "forecast" / "lstm_baseline" / "output" / "test_metrics.csv",
        kind="nn",
        color=OKABE_ITO[0],
    ),
    ModelSpec(
        key="transformer_baseline",
        label="Transformer Baseline",
        metrics_path=PROJECT_ROOT / "forecast" / "transformer_baseline" / "output" / "test_metrics.csv",
        kind="nn",
        color=OKABE_ITO[2],
    ),
    ModelSpec(
        key="lstm_direct",
        label="LSTM Direct",
        metrics_path=PROJECT_ROOT / "forecast" / "lstm" / "output" / "test_metrics.csv",
        kind="nn",
        color=OKABE_ITO[5],
    ),
    ModelSpec(
        key="transformer_direct",
        label="Transformer Direct",
        metrics_path=PROJECT_ROOT / "forecast" / "transformer" / "output" / "test_metrics.csv",
        kind="nn",
        color=OKABE_ITO[1],
    ),
]

METRICS = ["mse", "rmse", "mae", "mape"]
HORIZON_PLOT_METRICS = ["rmse", "mae", "mape"]


def apply_plot_style() -> None:
    """设置论文/报告风格的 Matplotlib 样式。"""

    sns.set_theme(style="ticks", context="paper", font_scale=1.1)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["PingFang SC", "Heiti SC", "Arial", "Helvetica", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "savefig.dpi": 300,
            "figure.constrained_layout.use": True,
        }
    )


def save_figure(fig: plt.Figure, base_path: Path) -> list[Path]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for suffix in [".png", ".pdf"]:
        output = base_path.with_suffix(suffix)
        fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
        saved.append(output)
    plt.close(fig)
    return saved


def extract_horizon(target_name: str) -> int:
    match = re.search(r"(\d+)$", target_name)
    if match is None:
        raise ValueError(f"无法从目标列中解析预测步长：{target_name}")
    return int(match.group(1))


def extract_target_type(target_name: str) -> str:
    if "energy" in target_name:
        return "total"
    elif "peak" in target_name:
        return "peak"
    elif "valley" in target_name:
        return "valley"
    raise ValueError(f"无法解析目标类型：{target_name}")


def load_metrics(spec: ModelSpec) -> pd.DataFrame | None:
    """读取单个模型的测试指标并统一列格式。"""

    if not spec.metrics_path.exists():
        return None

    frame = pd.read_csv(spec.metrics_path)

    # XGBoost 训练脚本输出 test_rmse/test_mae 等列；
    # 神经网络测试脚本输出 rmse/mae 等列，因此这里统一成同一格式。
    if spec.kind == "xgboost":
        frame = frame.rename(
            columns={
                "test_mse": "mse",
                "test_rmse": "rmse",
                "test_mae": "mae",
                "test_mape": "mape",
            }
        )[["target", "mse", "rmse", "mae", "mape"]]
    else:
        frame = frame[["target", "mse", "rmse", "mae", "mape"]]

    frame["horizon"] = frame["target"].map(extract_horizon)
    frame["target_type"] = frame["target"].map(extract_target_type)
    frame["model_key"] = spec.key
    frame["model_label"] = spec.label
    frame["color"] = spec.color
    return frame


def load_available_results() -> pd.DataFrame:
    """读取所有已经存在的模型指标文件。"""

    frames = [frame for spec in MODEL_SPECS if (frame := load_metrics(spec)) is not None]
    if not frames:
        raise FileNotFoundError("没有找到任何可用的模型测试指标文件")
    results = pd.concat(frames, ignore_index=True)
    model_order = [spec.label for spec in MODEL_SPECS if spec.metrics_path.exists()]
    results["model_label"] = pd.Categorical(
        results["model_label"], categories=model_order, ordered=True
    )
    return results.sort_values(["model_label", "horizon"]).reset_index(drop=True)


def available_model_labels(results: pd.DataFrame) -> list[str]:
    return [spec.label for spec in MODEL_SPECS if spec.label in set(results["model_label"].astype(str))]


def available_target_types(results: pd.DataFrame) -> list[str]:
    types_in_data = set(results["target_type"].astype(str))
    return [t for t in ["total", "peak", "valley"] if t in types_in_data]


def write_summary_tables(results: pd.DataFrame) -> list[Path]:
    """写出长表和平均指标表，方便论文或答辩引用。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = []

    long_path = OUTPUT_DIR / "metrics_long.csv"
    results.to_csv(long_path, index=False, encoding="utf-8")
    saved.append(long_path)

    average = (
        results.groupby(["model_key", "model_label", "target_type"], observed=True)[METRICS]
        .mean()
        .reset_index()
    )
    average_path = OUTPUT_DIR / "metrics_average.csv"
    average.to_csv(average_path, index=False, encoding="utf-8")
    saved.append(average_path)
    return saved


def plot_average_metric_bars(results: pd.DataFrame) -> list[Path]:
    """总用电平均指标柱状图（与 XGBoost 公平对比）。"""
    energy_data = results[results["target_type"] == "total"]
    average = energy_data.groupby("model_label", observed=True)[METRICS].mean().reset_index()
    palette = {spec.label: spec.color for spec in MODEL_SPECS if spec.metrics_path.exists()}

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8))
    axes = axes.flatten()
    titles = {
        "mse": "Average MSE",
        "rmse": "Average RMSE",
        "mae": "Average MAE",
        "mape": "Average MAPE (%)",
    }

    for ax, metric in zip(axes, METRICS, strict=True):
        sns.barplot(
            data=average,
            x="model_label",
            y=metric,
            hue="model_label",
            palette=palette,
            legend=False,
            ax=ax,
        )
        ax.set_title(titles[metric])
        ax.set_xlabel("")
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Model Comparison: Average Metrics (Total Energy)", fontsize=11, fontweight="bold")
    return save_figure(fig, FIGURE_DIR / "comparison_average_metrics")


def plot_horizon_lines(results: pd.DataFrame) -> list[Path]:
    """按 target_type 分行，展示各模型逐日误差。"""
    palette = {spec.label: spec.color for spec in MODEL_SPECS if spec.metrics_path.exists()}
    types = available_target_types(results)
    n_types = len(types)

    saved: list[Path] = []
    for metric in HORIZON_PLOT_METRICS:
        fig, axes = plt.subplots(1, n_types, figsize=(4.5 * n_types, 3.5), squeeze=False)
        for ax, target_type in zip(axes[0], types, strict=True):
            type_data = results[results["target_type"] == target_type]
            sns.lineplot(
                data=type_data,
                x="horizon",
                y=metric,
                hue="model_label",
                style="model_label",
                markers=True,
                dashes=False,
                palette=palette,
                ax=ax,
            )
            ax.set_title(TARGET_TYPE_LABELS.get(target_type, target_type))
            ax.set_xlabel("Forecast day")
            ax.set_ylabel(metric.upper())
            ax.set_xticks(range(1, 8))
            if ax != axes[0, -1]:
                ax.legend_.remove()
            else:
                ax.legend(title="", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")

        fig.suptitle(f"{metric.upper()} by Forecast Horizon", fontsize=11, fontweight="bold")
        saved.extend(save_figure(fig, FIGURE_DIR / f"comparison_horizon_{metric}"))
    return saved


def plot_heatmaps(results: pd.DataFrame) -> list[Path]:
    """每类目标一张热力图（模型 × 预测日）。"""
    model_order = available_model_labels(results)
    types = available_target_types(results)

    saved: list[Path] = []
    for metric in ["rmse", "mape"]:
        fig, axes = plt.subplots(1, len(types), figsize=(5.0 * len(types), 4.0), squeeze=False)
        for ax, target_type in zip(axes[0], types, strict=True):
            type_data = results[results["target_type"] == target_type]
            pivot = type_data.pivot(
                index="model_label", columns="horizon", values=metric
            ).reindex(model_order)
            sns.heatmap(
                pivot,
                cmap="viridis",
                annot=True,
                fmt=".3f",
                linewidths=0.5,
                cbar_kws={"label": metric.upper()},
                ax=ax,
            )
            ax.set_title(TARGET_TYPE_LABELS.get(target_type, target_type))
            ax.set_xlabel("Forecast day")
            ax.set_ylabel("")

        fig.suptitle(f"{metric.upper()} Heatmap", fontsize=11, fontweight="bold")
        saved.extend(save_figure(fig, FIGURE_DIR / f"comparison_heatmap_{metric}"))
    return saved


def _rank_frame(results: pd.DataFrame) -> pd.DataFrame:
    """在每个 (target_type, horizon) 组内排序。"""
    frames: list[pd.DataFrame] = []
    for metric in METRICS:
        ranked = results[["model_label", "target_type", "horizon", metric]].copy()
        ranked["metric"] = metric
        ranked["rank"] = ranked.groupby(["target_type", "horizon"])[metric].rank(
            method="min", ascending=True
        )
        frames.append(ranked)
    return pd.concat(frames, ignore_index=True)


def plot_rank_summary(results: pd.DataFrame) -> list[Path]:
    """总用电排名汇总（与 XGBoost 公平对比）。"""
    energy_data = results[results["target_type"] == "total"]
    rank_frame = _rank_frame(energy_data)
    avg_rank = (
        rank_frame.groupby(["model_label", "metric"], observed=True)["rank"].mean().reset_index()
    )
    best_count = (
        rank_frame[rank_frame["rank"] == 1]
        .groupby(["model_label", "metric"], observed=True)
        .size()
        .reset_index(name="win_count")
    )

    palette = {
        "mse": OKABE_ITO[0],
        "rmse": OKABE_ITO[4],
        "mae": OKABE_ITO[2],
        "mape": OKABE_ITO[6],
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))

    sns.barplot(
        data=avg_rank,
        x="model_label",
        y="rank",
        hue="metric",
        palette=palette,
        ax=axes[0],
    )
    axes[0].set_title("Average Rank by Metric (Total Energy)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Average rank (lower is better)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(title="", frameon=False, ncol=2)

    best_pivot = (
        best_count.pivot(index="model_label", columns="metric", values="win_count")
        .fillna(0)
        .reindex(available_model_labels(energy_data))
        .reindex(columns=METRICS)
    )
    best_pivot.plot(
        kind="bar",
        stacked=True,
        color=[palette[metric] for metric in METRICS],
        ax=axes[1],
        width=0.8,
    )
    axes[1].set_title("Best Horizon Count (Total Energy)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Number of best horizons")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(title="", frameon=False, ncol=2)

    fig.suptitle("Model Comparison: Rank Summary (Total Energy)", fontsize=11, fontweight="bold")
    return save_figure(fig, FIGURE_DIR / "comparison_rank_summary")


def generate_comparison_figures() -> list[Path]:
    """读取已有测试结果并生成所有对比图。

    生成内容包括平均指标柱状图、预测步长折线图、热力图和排名汇总图。
    XGBoost 只参与 total 目标的公平对比，神经网络模型参与 total/peak/valley 三类目标。
    """
    apply_plot_style()
    results = load_available_results()

    saved: list[Path] = []
    saved.extend(write_summary_tables(results))
    saved.extend(plot_average_metric_bars(results))
    saved.extend(plot_horizon_lines(results))
    saved.extend(plot_heatmaps(results))
    saved.extend(plot_rank_summary(results))
    return saved


def format_saved_paths(paths: Iterable[Path]) -> str:
    return "\n".join(str(path) for path in paths)

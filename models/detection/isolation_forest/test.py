"""Isolation Forest 异常检测评估与结果落盘。"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

try:
    from .config import FEATURE_COLUMNS, IsolationForestPaths
except ImportError:  # 兼容 `python detection/isolation_forest/test.py`
    from config import FEATURE_COLUMNS, IsolationForestPaths


matplotlib.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def compute_deviation(df: pd.DataFrame, anomaly_mask: np.ndarray) -> pd.DataFrame:
    """计算异常样本各特征偏离正常均值的标准差倍数。"""

    normal = df.loc[~anomaly_mask, FEATURE_COLUMNS]
    normal_mean = normal.mean()
    normal_std = normal.std().replace(0, 1e-8)
    anomaly = df.loc[anomaly_mask, FEATURE_COLUMNS]
    return ((anomaly - normal_mean) / normal_std).abs()


def plot_score_distribution(scores: np.ndarray, threshold: float, output_dir: Path) -> Path:
    """绘制异常分数分布直方图。"""

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=80, color="steelblue", alpha=0.75, edgecolor="white", linewidth=0.3)
    ax.axvline(
        x=threshold,
        color="tomato",
        linestyle="--",
        linewidth=1.5,
        label=f"阈值 ({threshold:.4f})",
    )
    ax.set_xlabel("异常分数（越低越异常）")
    ax.set_ylabel("样本数")
    ax.set_title("Isolation Forest 异常分数分布")
    ax.legend()
    plt.tight_layout()
    out = output_dir / "score_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_feature_deviation(deviation: pd.DataFrame, output_dir: Path) -> Path:
    """绘制异常样本各特征的偏离度箱线图。"""

    out = output_dir / "feature_deviation.png"
    if deviation.empty:
        return out

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(
        [deviation[column].values for column in FEATURE_COLUMNS],
        tick_labels=FEATURE_COLUMNS,
        patch_artist=True,
        boxprops={"facecolor": "tomato", "alpha": 0.5},
        flierprops={"markersize": 3},
    )
    ax.set_ylabel("偏离标准差倍数（绝对值）")
    ax.set_title("异常样本各特征偏离度")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def build_summary(
    df: pd.DataFrame,
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    deviation: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """构建供论文结果检查和 LLM 上下文使用的异常检测摘要。"""

    n_total = len(df)
    n_anomaly = int((labels == -1).sum())
    anomaly_pct = n_anomaly / n_total * 100

    top_feature_means: dict[str, float] = {}
    if not deviation.empty:
        top_feature_means = deviation.mean().sort_values(ascending=False).head(5).to_dict()
        top_feature_means = {key: round(value, 2) for key, value in top_feature_means.items()}

    return {
        "model": "IsolationForest",
        "contamination": float(config["model"]["contamination"]),
        "n_estimators": int(config["model"]["n_estimators"]),
        "n_total_samples": n_total,
        "n_anomaly_samples": n_anomaly,
        "anomaly_pct": round(anomaly_pct, 2),
        "score_threshold": round(threshold, 6),
        "score_min": round(float(scores.min()), 6),
        "score_max": round(float(scores.max()), 6),
        "score_mean": round(float(scores.mean()), 6),
        "top_deviating_features": top_feature_means,
    }


def save_outputs(
    *,
    df: pd.DataFrame,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    model: IsolationForest,
    config: dict[str, Any],
    paths: IsolationForestPaths,
) -> dict[str, Any]:
    """保存异常分数、异常样本、图表、摘要和模型文件。"""

    paths.output_dir.mkdir(parents=True, exist_ok=True)

    scores_df = df[["user_id", "window_start", "window_end"]].copy()
    scores_df["anomaly_score"] = scores
    scores_df["is_anomaly"] = labels == -1
    scores_df.to_csv(paths.anomaly_scores_path, index=False, encoding="utf-8", float_format="%.6f")

    anomaly_df = scores_df[scores_df["is_anomaly"]].sort_values("anomaly_score").copy()
    anomaly_df.to_csv(paths.anomaly_samples_path, index=False, encoding="utf-8", float_format="%.6f")

    anomaly_mask = labels == -1
    deviation = compute_deviation(df, anomaly_mask)
    dist_path = plot_score_distribution(scores, threshold, paths.output_dir)
    dev_path = plot_feature_deviation(deviation, paths.output_dir)

    summary = build_summary(df, scores, labels, threshold, deviation, config)
    with paths.anomaly_summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    with paths.model_path.open("wb") as file:
        pickle.dump(model, file)

    print(f"异常分数已写入：{paths.anomaly_scores_path}")
    print(f"异常样本已写入：{paths.anomaly_samples_path}")
    print(f"分数分布图已保存：{dist_path}")
    print(f"特征偏离度图已保存：{dev_path}")
    print(f"异常摘要已写入：{paths.anomaly_summary_path}")
    print(f"模型已保存：{paths.model_path}")
    return summary

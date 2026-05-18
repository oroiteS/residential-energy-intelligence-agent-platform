#!/usr/bin/env python3
"""Isolation Forest 无监督异常检测。

对 window_features.csv 训练 IsolationForest，输出：
- anomaly_scores.csv：所有样本的异常分数和标签
- anomaly_samples.csv：仅包含被标记为异常的样本
- anomaly_summary.json：供 LLM 上下文使用的异常摘要
- score_distribution.png：异常分数分布图
- feature_deviation.png：异常样本各特征偏离度图
- isolation_forest.pkl：供推理复用
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest

matplotlib.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

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


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_features(features_path: Path) -> pd.DataFrame:
    df = pd.read_csv(features_path, encoding="utf-8")
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"特征列缺失：{missing}")
    return df


def compute_deviation(
    df: pd.DataFrame,
    anomaly_mask: np.ndarray,
) -> pd.DataFrame:
    """计算异常样本各特征偏离正常均值的标准差倍数。"""
    normal = df.loc[~anomaly_mask, FEATURE_COLUMNS]
    normal_mean = normal.mean()
    normal_std = normal.std().replace(0, 1e-8)
    anomaly = df.loc[anomaly_mask, FEATURE_COLUMNS]
    deviation = (anomaly - normal_mean) / normal_std
    return deviation.abs()


def plot_score_distribution(scores: np.ndarray, threshold: float, output_dir: Path) -> Path:
    """绘制异常分数分布直方图。"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=80, color="steelblue", alpha=0.75, edgecolor="white", linewidth=0.3)
    ax.axvline(x=threshold, color="tomato", linestyle="--", linewidth=1.5,
               label=f"阈值 ({threshold:.4f})")
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
    if deviation.empty:
        return output_dir / "feature_deviation.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(
        [deviation[col].values for col in FEATURE_COLUMNS],
        tick_labels=FEATURE_COLUMNS,
        patch_artist=True,
        boxprops={"facecolor": "tomato", "alpha": 0.5},
        flierprops={"markersize": 3},
    )
    ax.set_ylabel("偏离标准差倍数（绝对值）")
    ax.set_title("异常样本各特征偏离度")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    out = output_dir / "feature_deviation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def build_summary(
    df: pd.DataFrame,
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    deviation: pd.DataFrame,
    cfg: dict,
) -> dict:
    n_total = len(df)
    n_anomaly = int((labels == -1).sum())
    anomaly_pct = n_anomaly / n_total * 100

    top_feature_means = {}
    if not deviation.empty:
        top_feature_means = deviation.mean().sort_values(ascending=False).head(5).to_dict()
        top_feature_means = {k: round(v, 2) for k, v in top_feature_means.items()}

    return {
        "model": "IsolationForest",
        "contamination": float(cfg["model"]["contamination"]),
        "n_estimators": int(cfg["model"]["n_estimators"]),
        "n_total_samples": n_total,
        "n_anomaly_samples": n_anomaly,
        "anomaly_pct": round(anomaly_pct, 2),
        "score_threshold": round(threshold, 6),
        "score_min": round(float(scores.min()), 6),
        "score_max": round(float(scores.max()), 6),
        "score_mean": round(float(scores.mean()), 6),
        "top_deviating_features": top_feature_means,
    }


def parse_args() -> argparse.Namespace:
    default_config = Path(__file__).resolve().parent / "config" / "default.yaml"
    parser = argparse.ArgumentParser(description="Isolation Forest 异常检测")
    parser.add_argument("--config", type=Path, default=default_config)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    project_root = Path(__file__).resolve().parents[2]
    features_path = project_root / cfg["data"]["features_path"]
    output_dir = project_root / cfg["output"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_features(features_path)
    print(f"加载样本数：{len(df)}")

    X = df[FEATURE_COLUMNS].values.astype(np.float64)

    model = IsolationForest(
        contamination=float(cfg["model"]["contamination"]),
        n_estimators=int(cfg["model"]["n_estimators"]),
        max_samples=float(cfg["model"]["max_samples"]),
        random_state=int(cfg["model"]["random_seed"]),
        n_jobs=-1,
    )
    labels = model.fit_predict(X)
    scores = model.decision_function(X)

    # threshold = scores 中使 label=-1 的最大分数（低于此值视为异常）
    anomaly_scores = scores[labels == -1]
    threshold = anomaly_scores.max() if len(anomaly_scores) > 0 else float("inf")
    print(f"异常分数阈值：{threshold:.6f}")

    n_anomaly = int((labels == -1).sum())
    print(f"异常样本数：{n_anomaly} / {len(df)}（{n_anomaly / len(df) * 100:.2f}%）")

    # 保存全部样本的异常分数
    scores_df = df[["user_id", "window_start", "window_end"]].copy()
    scores_df["anomaly_score"] = scores
    scores_df["is_anomaly"] = labels == -1
    scores_path = output_dir / cfg["output"]["anomaly_scores_file"]
    scores_df.to_csv(scores_path, index=False, encoding="utf-8", float_format="%.6f")
    print(f"异常分数已写入：{scores_path}")

    # 仅保存异常样本，按异常分数升序（越靠前越异常）
    anomaly_df = scores_df[scores_df["is_anomaly"]].sort_values("anomaly_score").copy()
    anomaly_path = output_dir / cfg["output"]["anomaly_samples_file"]
    anomaly_df.to_csv(anomaly_path, index=False, encoding="utf-8", float_format="%.6f")
    print(f"异常样本已写入：{anomaly_path}")

    # 特征偏离度
    anomaly_mask = labels == -1
    deviation = compute_deviation(df, anomaly_mask)

    # 图表
    dist_path = plot_score_distribution(scores, threshold, output_dir)
    print(f"分数分布图已保存：{dist_path}")
    dev_path = plot_feature_deviation(deviation, output_dir)
    print(f"特征偏离度图已保存：{dev_path}")

    # 摘要
    summary = build_summary(df, scores, labels, threshold, deviation, cfg)
    summary_path = output_dir / cfg["output"]["anomaly_summary_file"]
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"异常摘要已写入：{summary_path}")

    # 模型
    model_path = output_dir / cfg["output"]["model_file"]
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"模型已保存：{model_path}")


if __name__ == "__main__":
    main()

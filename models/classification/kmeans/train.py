#!/usr/bin/env python3
"""KMeans 居民用电行为聚类。

对 window_features.csv 做 StandardScaler + KMeans，输出：
- cluster_result.csv：每个样本的聚类标签
- cluster_centers.csv：各簇中心在原始特征空间的均值
- cluster_summary.json：供 LLM 上下文使用的摘要
- elbow_silhouette.png：手肘法 + 轮廓系数图
- cluster_profiles.png：各簇特征分布图
- scaler.pkl / kmeans_model.pkl：供推理复用
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt

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


def select_k(X_scaled: np.ndarray, k: int, k_range: range, output_dir: Path) -> None:
    """绘制手肘图 + 轮廓系数图，帮助人工确认 k 值。"""
    inertias, silhouettes = [], []
    for ki in k_range:
        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels, sample_size=min(10000, len(X_scaled))))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(list(k_range), inertias, "o-", color="steelblue")
    ax1.axvline(x=k, color="tomato", linestyle="--", label=f"k={k}")
    ax1.set_xlabel("聚类数 k")
    ax1.set_ylabel("惯性（Inertia）")
    ax1.set_title("手肘法")
    ax1.legend()

    ax2.plot(list(k_range), silhouettes, "s-", color="seagreen")
    ax2.axvline(x=k, color="tomato", linestyle="--", label=f"k={k}")
    ax2.set_xlabel("聚类数 k")
    ax2.set_ylabel("轮廓系数")
    ax2.set_title("轮廓系数")
    ax2.legend()

    plt.tight_layout()
    out = output_dir / "elbow_silhouette.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"手肘图已保存：{out}")


def plot_cluster_profiles(
    standardized_centers_df: pd.DataFrame,
    output_dir: Path,
    name_map: dict[int, str] | None = None,
) -> None:
    """绘制标准化空间下的聚类画像图。"""
    n_clusters = len(standardized_centers_df)
    colors = ["steelblue", "seagreen", "tomato", "goldenrod", "mediumpurple"]

    fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 5), sharey=True)
    if n_clusters == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        row = standardized_centers_df.iloc[i]
        cluster_id = int(row["cluster"])
        label = name_map.get(cluster_id, f"簇{cluster_id}") if name_map else f"簇{cluster_id}"
        ax.barh(FEATURE_COLUMNS, row[FEATURE_COLUMNS], color=colors[i % len(colors)], alpha=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{label}\n（{int(row['size'])} 样本）")
        ax.set_xlabel("标准化特征值（Z-score）")

    plt.suptitle("各聚类中心标准化特征画像", fontsize=13)
    plt.tight_layout()
    out = output_dir / "cluster_profiles.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"特征分布图已保存：{out}")


def run_kmeans(
    df: pd.DataFrame,
    cfg: dict,
    output_dir: Path,
) -> tuple[np.ndarray, StandardScaler, KMeans]:
    k = cfg["cluster"]["k"]
    k_range = range(cfg["cluster"]["k_range_min"], cfg["cluster"]["k_range_max"] + 1)
    n_init = cfg["cluster"]["n_init"]
    seed = cfg["cluster"]["random_seed"]

    X = df[FEATURE_COLUMNS].values.astype(np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"正在计算 k={k_range.start}~{k_range.stop - 1} 的手肘/轮廓系数...")
    select_k(X_scaled, k, k_range, output_dir)

    km = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
    labels = km.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels, sample_size=min(10000, len(X_scaled)))
    print(f"k={k} 轮廓系数：{sil:.4f}")

    return labels, scaler, km


def build_centers_df(
    df: pd.DataFrame,
    labels: np.ndarray,
    km: KMeans,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_tmp = df[["user_id"] + FEATURE_COLUMNS].copy()
    df_tmp["cluster"] = labels

    raw_centers = df_tmp.groupby("cluster")[FEATURE_COLUMNS].mean().reset_index()
    sizes = df_tmp.groupby("cluster").size().reset_index(name="size")
    raw_centers = raw_centers.merge(sizes, on="cluster")

    standardized_centers = pd.DataFrame(km.cluster_centers_, columns=FEATURE_COLUMNS)
    standardized_centers.insert(0, "cluster", range(len(standardized_centers)))
    standardized_centers["size"] = raw_centers["size"].values

    return raw_centers, standardized_centers


def parse_args() -> argparse.Namespace:
    default_config = Path(__file__).resolve().parent / "config" / "default.yaml"
    parser = argparse.ArgumentParser(description="KMeans 居民用电行为聚类")
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

    labels, scaler, km = run_kmeans(df, cfg, output_dir)

    # 读取已有摘要中的人工标注字段，重跑时不覆盖
    LABEL_KEYS = ("name", "description", "advice_focus")
    summary_path = output_dir / "cluster_summary.json"
    existing_labels: dict[str, dict] = {}
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            for key, val in json.load(f).items():
                existing_labels[key] = {k: val[k] for k in LABEL_KEYS if k in val}

    # 检测所有簇是否已命名，决定 cluster_result.csv 使用名称还是数字
    k = cfg["cluster"]["k"]
    name_map: dict[int, str] = {}
    for c in range(k):
        name = existing_labels.get(f"cluster_{c}", {}).get("name", "").strip()
        if name:
            name_map[c] = name

    all_named = len(name_map) == k
    result_df = df[["user_id", "window_start", "window_end"]].copy()
    if all_named:
        result_df["cluster"] = [name_map[lb] for lb in labels]
        print("已检测到所有簇命名，cluster 列将使用名称标签。")
    else:
        result_df["cluster"] = labels
        unnamed = [c for c in range(k) if c not in name_map]
        print(f"簇 {unnamed} 尚未命名，cluster 列使用数字标签。")
        print(f"请在 {summary_path} 中填写各簇的 name 字段后重新运行。")

    result_df.to_csv(output_dir / "cluster_result.csv", index=False, encoding="utf-8")
    print(f"聚类结果已写入：{output_dir / 'cluster_result.csv'}")

    raw_centers, standardized_centers = build_centers_df(df, labels, km)
    raw_centers.to_csv(output_dir / "cluster_centers.csv", index=False, encoding="utf-8", float_format="%.4f")
    print(f"簇中心已写入：{output_dir / 'cluster_centers.csv'}")
    standardized_centers.to_csv(
        output_dir / "cluster_centers_standardized.csv",
        index=False,
        encoding="utf-8",
        float_format="%.4f",
    )
    print(f"标准化簇中心已写入：{output_dir / 'cluster_centers_standardized.csv'}")

    plot_cluster_profiles(standardized_centers, output_dir, name_map)

    for c, cnt in sorted(result_df["cluster"].value_counts().items()):
        print(f"  {c}：{cnt} 样本（{cnt / len(result_df) * 100:.1f}%）")

    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(output_dir / "kmeans_model.pkl", "wb") as f:
        pickle.dump(km, f)

    summary = {}
    for _, row in raw_centers.iterrows():
        c = int(row["cluster"])
        key = f"cluster_{c}"
        # name/description/advice_focus 优先保留已有内容，不存在时给空字符串占位
        entry: dict = {
            "name": "",
            "description": "",
            "advice_focus": "",
        }
        entry.update(existing_labels.get(key, {}))
        entry.update({
            "size": int(row["size"]),
            "avg_energy": round(row["avg_energy"], 3),
            "peak_ratio": round(row["peak_ratio"], 3),
            "valley_ratio": round(row["valley_ratio"], 3),
            "load_factor": round(row["load_factor"], 3),
            "weekend_workday_ratio": round(row["weekend_workday_ratio"], 3),
            "volatility": round(row["volatility"], 3),
        })
        summary[key] = entry

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"聚类摘要已写入：{summary_path}")


if __name__ == "__main__":
    main()

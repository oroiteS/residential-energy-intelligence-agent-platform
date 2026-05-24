"""KMeans 聚类结果评估、图表和文件输出。"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from .config import FEATURE_COLUMNS, KMeansPaths
except ImportError:  # 兼容直接运行
    from config import FEATURE_COLUMNS, KMeansPaths


matplotlib.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

LABEL_KEYS = ("name", "description", "advice_focus")


def select_k(features_scaled: np.ndarray, k: int, k_range: range, output_dir: Path) -> None:
    """绘制手肘图和轮廓系数图，帮助人工确认 k 值。"""

    inertias: list[float] = []
    silhouettes: list[float] = []
    for ki in k_range:
        model = KMeans(n_clusters=ki, random_state=42, n_init=10)
        labels = model.fit_predict(features_scaled)
        inertias.append(float(model.inertia_))
        silhouettes.append(float(silhouette_score(features_scaled, labels, sample_size=min(10000, len(features_scaled)))))

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
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
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

    for index, ax in enumerate(axes):
        row = standardized_centers_df.iloc[index]
        cluster_id = int(row["cluster"])
        label = name_map.get(cluster_id, f"簇{cluster_id}") if name_map else f"簇{cluster_id}"
        ax.barh(FEATURE_COLUMNS, row[FEATURE_COLUMNS], color=colors[index % len(colors)], alpha=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{label}\n（{int(row['size'])} 样本）")
        ax.set_xlabel("标准化特征值（Z-score）")

    plt.suptitle("各聚类中心标准化特征画像", fontsize=13)
    plt.tight_layout()
    out = output_dir / "cluster_profiles.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"特征分布图已保存：{out}")


def build_centers_df(
    df: pd.DataFrame,
    labels: np.ndarray,
    model: KMeans,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成原始特征空间和标准化空间下的簇中心表。"""

    df_tmp = df[["user_id"] + FEATURE_COLUMNS].copy()
    df_tmp["cluster"] = labels

    raw_centers = df_tmp.groupby("cluster")[FEATURE_COLUMNS].mean().reset_index()
    sizes = df_tmp.groupby("cluster").size().reset_index(name="size")
    raw_centers = raw_centers.merge(sizes, on="cluster")

    standardized_centers = pd.DataFrame(model.cluster_centers_, columns=FEATURE_COLUMNS)
    standardized_centers.insert(0, "cluster", range(len(standardized_centers)))
    standardized_centers["size"] = raw_centers["size"].values
    return raw_centers, standardized_centers


def load_existing_labels(summary_path: Path) -> dict[str, dict[str, Any]]:
    """读取已有簇解释字段，重跑时保留人工命名结果。"""

    if not summary_path.exists():
        return {}

    with summary_path.open(encoding="utf-8") as file:
        summary = json.load(file)
    return {
        key: {label_key: value[label_key] for label_key in LABEL_KEYS if label_key in value}
        for key, value in summary.items()
    }


def build_name_map(existing_labels: dict[str, dict[str, Any]], k: int) -> dict[int, str]:
    """从已有摘要中提取完整的簇名称映射。"""

    name_map: dict[int, str] = {}
    for cluster_id in range(k):
        name = existing_labels.get(f"cluster_{cluster_id}", {}).get("name", "").strip()
        if name:
            name_map[cluster_id] = name
    return name_map


def write_summary(
    raw_centers: pd.DataFrame,
    existing_labels: dict[str, dict[str, Any]],
    summary_path: Path,
) -> None:
    """写出聚类摘要，并保留人工维护的解释字段。"""

    summary: dict[str, dict[str, Any]] = {}
    for _, row in raw_centers.iterrows():
        cluster_id = int(row["cluster"])
        key = f"cluster_{cluster_id}"
        entry: dict[str, Any] = {
            "name": "",
            "description": "",
            "advice_focus": "",
        }
        entry.update(existing_labels.get(key, {}))
        entry.update(
            {
                "size": int(row["size"]),
                "avg_energy": round(row["avg_energy"], 3),
                "peak_ratio": round(row["peak_ratio"], 3),
                "valley_ratio": round(row["valley_ratio"], 3),
                "load_factor": round(row["load_factor"], 3),
                "weekend_workday_ratio": round(row["weekend_workday_ratio"], 3),
                "volatility": round(row["volatility"], 3),
            }
        )
        summary[key] = entry

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f"聚类摘要已写入：{summary_path}")


def save_outputs(
    *,
    df: pd.DataFrame,
    labels: np.ndarray,
    scaler: StandardScaler,
    model: KMeans,
    features_scaled: np.ndarray,
    config: dict[str, Any],
    paths: KMeansPaths,
) -> None:
    """保存 KMeans 聚类结果、模型、标准化器、图表和摘要。"""

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    cluster_config = config["cluster"]
    k = int(cluster_config["k"])
    k_range = range(int(cluster_config["k_range_min"]), int(cluster_config["k_range_max"]) + 1)

    print(f"正在计算 k={k_range.start}~{k_range.stop - 1} 的手肘/轮廓系数...")
    select_k(features_scaled, k, k_range, paths.output_dir)

    existing_labels = load_existing_labels(paths.cluster_summary_path)
    name_map = build_name_map(existing_labels, k)
    all_named = len(name_map) == k

    result_df = df[["user_id", "window_start", "window_end"]].copy()
    if all_named:
        result_df["cluster"] = [name_map[int(label)] for label in labels]
        print("已检测到所有簇命名，cluster 列将使用名称标签。")
    else:
        result_df["cluster"] = labels
        unnamed = [cluster_id for cluster_id in range(k) if cluster_id not in name_map]
        print(f"簇 {unnamed} 尚未命名，cluster 列使用数字标签。")
        print(f"请在 {paths.cluster_summary_path} 中填写各簇的 name 字段后重新运行。")

    result_df.to_csv(paths.cluster_result_path, index=False, encoding="utf-8")
    print(f"聚类结果已写入：{paths.cluster_result_path}")

    raw_centers, standardized_centers = build_centers_df(df, labels, model)
    raw_centers.to_csv(paths.cluster_centers_path, index=False, encoding="utf-8", float_format="%.4f")
    standardized_centers.to_csv(
        paths.cluster_centers_standardized_path,
        index=False,
        encoding="utf-8",
        float_format="%.4f",
    )
    print(f"簇中心已写入：{paths.cluster_centers_path}")
    print(f"标准化簇中心已写入：{paths.cluster_centers_standardized_path}")

    plot_cluster_profiles(standardized_centers, paths.output_dir, name_map)

    for cluster_name, count in sorted(result_df["cluster"].value_counts().items()):
        print(f"  {cluster_name}：{count} 样本（{count / len(result_df) * 100:.1f}%）")

    with paths.scaler_path.open("wb") as file:
        pickle.dump(scaler, file)
    with paths.model_path.open("wb") as file:
        pickle.dump(model, file)

    write_summary(raw_centers, existing_labels, paths.cluster_summary_path)

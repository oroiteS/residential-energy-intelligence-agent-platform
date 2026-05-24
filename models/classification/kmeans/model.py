"""KMeans 聚类模型构造与训练。"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def scale_features(features: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """标准化行为特征。

    KMeans 基于欧氏距离工作，不同量纲的特征会直接影响簇中心位置，
    因此训练前必须使用 `StandardScaler` 统一尺度。
    """

    scaler = StandardScaler()
    return scaler.fit_transform(features), scaler


def build_model(config: dict[str, Any]) -> KMeans:
    """根据配置创建 KMeans 聚类模型。"""

    cluster_config = config["cluster"]
    return KMeans(
        n_clusters=int(cluster_config["k"]),
        random_state=int(cluster_config["random_seed"]),
        n_init=int(cluster_config["n_init"]),
    )


def run_kmeans(
    features: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, StandardScaler, KMeans, np.ndarray, float]:
    """训练 KMeans 并返回标签、标准化器、模型、标准化特征和轮廓系数。

    轮廓系数用于衡量聚类分离程度，取值越高说明同簇样本越相似、
    不同簇之间越分开。这里抽样最多 10000 条计算，避免全量样本过慢。
    """

    features_scaled, scaler = scale_features(features)
    model = build_model(config)
    labels = model.fit_predict(features_scaled)
    score = silhouette_score(features_scaled, labels, sample_size=min(10000, len(features_scaled)))
    return labels, scaler, model, features_scaled, float(score)

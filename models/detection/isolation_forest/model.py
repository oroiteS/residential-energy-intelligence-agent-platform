"""Isolation Forest 模型构造与推理。"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest


def build_model(config: dict[str, Any]) -> IsolationForest:
    """根据配置创建 Isolation Forest 模型。

    Isolation Forest 是无监督异常检测模型，`contamination` 表示预期异常
    比例，会直接影响最终被标记为异常的样本数量。
    """

    model_config = config["model"]
    return IsolationForest(
        contamination=float(model_config["contamination"]),
        n_estimators=int(model_config["n_estimators"]),
        max_samples=float(model_config["max_samples"]),
        random_state=int(model_config["random_seed"]),
        n_jobs=-1,
    )


def fit_predict(model: IsolationForest, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """训练模型并返回标签、异常分数和异常阈值。

    `decision_function` 分数越低越异常；阈值取被模型标记为异常的样本中
    分数最高的一个，便于报告解释“低于该分数视为异常”。
    """

    labels = model.fit_predict(features)
    scores = model.decision_function(features)
    anomaly_scores = scores[labels == -1]
    threshold = float(anomaly_scores.max()) if len(anomaly_scores) > 0 else float("inf")
    return labels, scores, threshold

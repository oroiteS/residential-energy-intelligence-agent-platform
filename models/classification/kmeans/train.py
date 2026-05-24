#!/usr/bin/env python3
"""KMeans 居民用电行为聚类训练函数。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from .config import FEATURE_COLUMNS, build_paths, load_config
    from .data import load_features
    from .model import run_kmeans
    from .test import save_outputs
except ImportError:  # 兼容 `python classification/kmeans/train.py`
    import importlib

    _config = importlib.import_module("config")
    _data = importlib.import_module("data")
    _model = importlib.import_module("model")
    _test = importlib.import_module("test")
    FEATURE_COLUMNS = getattr(_config, "FEATURE_COLUMNS")
    build_paths = getattr(_config, "build_paths")
    load_config = getattr(_config, "load_config")
    load_features = getattr(_data, "load_features")
    run_kmeans = getattr(_model, "run_kmeans")
    save_outputs = getattr(_test, "save_outputs")


def train(config_path: Path) -> None:
    """执行 KMeans 聚类训练并保存全部结果。

    输入是 classification 预处理阶段生成的 7 天窗口 16 维行为特征；
    输出包括每个窗口的聚类标签、聚类中心、标准化器和 KMeans 模型。
    """

    config = load_config(config_path)
    paths = build_paths(config)

    df = load_features(paths.features_path)
    print(f"加载样本数：{len(df)}")

    # KMeans 只使用数值行为特征，不直接使用 user_id 或窗口日期。
    # user_id/window_start/window_end 会在保存结果时用于回连样本身份。
    features = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    labels, scaler, model, features_scaled, silhouette = run_kmeans(features, config)
    print(f"k={config['cluster']['k']} 轮廓系数：{silhouette:.4f}")

    save_outputs(
        df=df,
        labels=labels,
        scaler=scaler,
        model=model,
        features_scaled=features_scaled,
        config=config,
        paths=paths,
    )


def main() -> None:
    """兼容旧入口：直接运行 train.py 时仍然启动完整训练。"""

    try:
        from .main import parse_args
    except ImportError:
        import importlib

        parse_args = getattr(importlib.import_module("main"), "parse_args")

    args = parse_args()
    train(args.config)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Isolation Forest 无监督异常检测训练函数。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from .config import FEATURE_COLUMNS, build_paths, load_config
    from .data import load_features
    from .model import build_model, fit_predict
    from .test import save_outputs
except ImportError:  # 兼容 `python detection/isolation_forest/train.py`
    import importlib

    _config = importlib.import_module("config")
    _data = importlib.import_module("data")
    _model = importlib.import_module("model")
    _test = importlib.import_module("test")
    FEATURE_COLUMNS = getattr(_config, "FEATURE_COLUMNS")
    build_paths = getattr(_config, "build_paths")
    load_config = getattr(_config, "load_config")
    load_features = getattr(_data, "load_features")
    build_model = getattr(_model, "build_model")
    fit_predict = getattr(_model, "fit_predict")
    save_outputs = getattr(_test, "save_outputs")


def train(config_path: Path) -> dict:
    """执行 Isolation Forest 训练并保存全部结果。

    训练阶段只负责串联配置、数据、模型和评估输出；具体的配置读取、
    数据校验、模型构造和结果保存分别放在独立模块中，便于后续单独
    测试或替换实现。
    """

    config = load_config(config_path)
    paths = build_paths(config)

    df = load_features(paths.features_path)
    print(f"加载样本数：{len(df)}")

    # Isolation Forest 使用与分类任务相同的 16 维窗口特征。
    # 它不需要标签，因此适合在缺少人工异常标注的场景下先筛选疑似异常窗口。
    features = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    model = build_model(config)
    labels, scores, threshold = fit_predict(model, features)

    print(f"异常分数阈值：{threshold:.6f}")
    n_anomaly = int((labels == -1).sum())
    print(f"异常样本数：{n_anomaly} / {len(df)}（{n_anomaly / len(df) * 100:.2f}%）")

    return save_outputs(
        df=df,
        labels=labels,
        scores=scores,
        threshold=threshold,
        model=model,
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

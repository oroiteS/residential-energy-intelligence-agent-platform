#!/usr/bin/env python3
"""XGBoost 预测模型训练函数。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from .config import build_paths, load_config
    from .data import load_dataset, make_split, select_feature_columns
    from .model import train_one_target
    from .test import evaluate_peak_valley_baseline, write_baseline_report, write_feature_columns, write_metrics
except ImportError:  # 兼容 `python forecast/xgboost/train.py`
    import importlib

    _config = importlib.import_module("config")
    _data = importlib.import_module("data")
    _model = importlib.import_module("model")
    _test = importlib.import_module("test")
    build_paths = getattr(_config, "build_paths")
    load_config = getattr(_config, "load_config")
    load_dataset = getattr(_data, "load_dataset")
    make_split = getattr(_data, "make_split")
    select_feature_columns = getattr(_data, "select_feature_columns")
    train_one_target = getattr(_model, "train_one_target")
    evaluate_peak_valley_baseline = getattr(_test, "evaluate_peak_valley_baseline")
    write_baseline_report = getattr(_test, "write_baseline_report")
    write_feature_columns = getattr(_test, "write_feature_columns")
    write_metrics = getattr(_test, "write_metrics")


def train(config_path: Path, *, no_resume: bool = False) -> list[dict[str, Any]]:
    """训练 30 天预测 7 天的 XGBoost 回归器集合。

    XGBoost baseline 只直接训练 target_columns 中配置的目标；
    当前主线通常是未来 7 天总用电量，即 7 个独立回归器。
    """

    config = load_config(config_path)
    paths = build_paths(config)

    data_config = config["data"]
    feature_config = config.get("features")
    train_config = config["training"]

    df = load_dataset(str(paths.dataset_path))
    target_columns = list(data_config["target_columns"])
    meta_columns = list(data_config["meta_columns"])
    feature_columns = select_feature_columns(
        df,
        meta_columns=meta_columns,
        target_columns=target_columns,
        feature_config=feature_config,
    )
    split, split_strategy = make_split(df, config)

    print(f"数据集：{paths.dataset_path}")
    print(f"切分策略：{split_strategy}")
    print(f"样本数：train={len(split.train)}, valid={len(split.valid)}, test={len(split.test)}")
    print(
        "用户数："
        f"train={split.train['user_id'].nunique()}, "
        f"valid={split.valid['user_id'].nunique()}, "
        f"test={split.test['user_id'].nunique()}"
    )
    print(f"特征数：{len(feature_columns)}")
    print(f"目标数：{len(target_columns)}")

    write_feature_columns(feature_columns, paths.feature_path)

    # 所有目标模型共用同一套特征列和切分结果，但每个目标单独训练。
    params = dict(config["xgboost_params"])
    params["seed"] = int(train_config["random_seed"])
    resume = bool(train_config["resume"]) and not no_resume

    split_summary = {
        "split_strategy": split_strategy,
        "train_samples": len(split.train),
        "valid_samples": len(split.valid),
        "test_samples": len(split.test),
        "train_users": int(split.train["user_id"].nunique()),
        "valid_users": int(split.valid["user_id"].nunique()),
        "test_users": int(split.test["user_id"].nunique()),
    }

    metrics: list[dict[str, Any]] = []
    for target_column in target_columns:
        # 每个 y_energy_dXX 单独训练一个 Booster，便于观察不同预测步长的误差。
        result = train_one_target(
            target_column=target_column,
            split=split,
            feature_columns=feature_columns,
            params=params,
            num_boost_round=int(train_config["num_boost_round"]),
            early_stopping_rounds=int(train_config["early_stopping_rounds"]),
            checkpoint_interval=int(train_config["checkpoint_interval"]),
            resume=resume,
            model_dir=paths.model_dir,
            checkpoint_dir=paths.checkpoint_dir,
        )
        metrics.append({**split_summary, **result})

    write_metrics(metrics, paths.metrics_path)
    print(f"指标已保存：{paths.metrics_path}")
    print(f"特征列已保存：{paths.feature_path}")

    # 峰谷 baseline 不单独训练模型，而是用历史峰时占比从总量预测中推导峰/谷。
    baseline = evaluate_peak_valley_baseline(split.test, target_columns)
    write_baseline_report(baseline, paths.baseline_path)
    return metrics


def main() -> None:
    """兼容直接运行 train.py 的训练入口。"""

    try:
        from .main import parse_args
    except ImportError:
        import importlib

        parse_args = getattr(importlib.import_module("main"), "parse_args")

    args = parse_args()
    train(args.config, no_resume=args.no_resume)


if __name__ == "__main__":
    main()

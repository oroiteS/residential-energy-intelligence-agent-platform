#!/usr/bin/env python3
"""XGBoost 居民用电行为分类器训练函数。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

try:
    from .config import FEATURE_COLUMNS, build_paths, load_config
    from .data import load_data, split_by_user_majority_label
    from .model import build_model
    from .test import evaluate_model, save_outputs
except ImportError:  # 兼容 `python classification/xgboost/train.py`
    import importlib

    _config = importlib.import_module("config")
    _data = importlib.import_module("data")
    _model = importlib.import_module("model")
    _test = importlib.import_module("test")
    FEATURE_COLUMNS = getattr(_config, "FEATURE_COLUMNS")
    build_paths = getattr(_config, "build_paths")
    load_config = getattr(_config, "load_config")
    load_data = getattr(_data, "load_data")
    split_by_user_majority_label = getattr(_data, "split_by_user_majority_label")
    build_model = getattr(_model, "build_model")
    evaluate_model = getattr(_test, "evaluate_model")
    save_outputs = getattr(_test, "save_outputs")


def train_and_evaluate(df: pd.DataFrame, config: dict[str, Any], output_dir: Path) -> tuple[xgb.XGBClassifier, LabelEncoder, dict[str, Any]]:
    """训练 XGBoost 分类器并返回模型、标签编码器和评估报告。

    这里的监督标签来自 KMeans 聚类结果，相当于学习“给定 16 维行为特征，
    如何快速预测该窗口属于哪个居民用电行为类别”。
    """

    model_config = config["model"]
    training_config = config["training"]
    x = df[FEATURE_COLUMNS]
    y = df["cluster"]
    groups = df["user_id"]

    # XGBoost 多分类需要整数类别编号。
    # LabelEncoder 负责在聚类标签和整数编号之间转换，并会随模型一起保存。
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    assert y_enc is not None
    n_classes = len(label_encoder.classes_)

    # 按用户划分训练/测试，避免同一用户的不同窗口同时出现在两边造成指标虚高。
    train_idx, test_idx = split_by_user_majority_label(
        df=df,
        test_ratio=float(training_config.get("test_ratio", 0.2)),
        random_seed=int(model_config["random_seed"]),
    )
    x_train = x.iloc[train_idx]
    y_train = y_enc[train_idx]
    train_groups = groups.iloc[train_idx]
    x_test = x.iloc[test_idx]
    y_test = y_enc[test_idx]

    model = build_model(config, n_classes)
    cv_folds = int(training_config["cv_folds"])

    # StratifiedGroupKFold 同时满足“类别分层”和“用户分组”。
    # 这样交叉验证更接近真实新用户/新窗口分类场景。
    cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=model_config["random_seed"])
    cv_scores = cross_val_score(model, x_train, y_train, groups=train_groups, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"{cv_folds}折用户分组交叉验证准确率：{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model.fit(x_train, y_train)
    report = evaluate_model(
        model=model,
        label_encoder=label_encoder,
        x_test=x_test,
        y_test=y_test,
        output_dir=output_dir,
    )
    report.update(
        {
            "split_strategy": "stratified_user_holdout",
            "train_samples": int(len(train_idx)),
            "test_samples": int(len(test_idx)),
            "train_users": int(train_groups.nunique()),
            "test_users": int(groups.iloc[test_idx].nunique()),
            "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
            "cv_accuracy_std": round(float(cv_scores.std()), 4),
        }
    )
    return model, label_encoder, report


def train(config_path: Path) -> dict:
    """执行 XGBoost 分类训练并保存模型和评估产物。

    输入：
    - window_features.csv: 7 天窗口 16 维行为特征；
    - cluster_result.csv: KMeans 产生的窗口聚类标签。

    输出：
    - xgboost_model.json；
    - label_encoder.pkl；
    - classifier_report.json 和评估图表。
    """

    config = load_config(config_path)
    paths = build_paths(config)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(paths.features_path, paths.labels_path)
    print(
        f"样本数：{len(df)}，用户数：{df['user_id'].nunique()}，"
        f"类别数：{df['cluster'].nunique()}，类别分布：{df['cluster'].value_counts().to_dict()}"
    )

    model, label_encoder, report = train_and_evaluate(df, config, paths.output_dir)
    save_outputs(model=model, label_encoder=label_encoder, report=report, paths=paths)
    return report


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

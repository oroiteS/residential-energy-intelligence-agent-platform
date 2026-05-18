#!/usr/bin/env python3
"""XGBoost 居民用电行为分类器训练。

用 KMeans 产生的聚类标签作为监督信号，训练 XGBoost 多分类器。
训练好的模型可用于：给新用户上传的 7 天数据直接推断行为类别。

输出：
- xgboost_model.json：训练好的模型
- classifier_report.json：准确率、F1、混淆矩阵、特征重要性
- feature_importance.png：特征重要性条形图
- confusion_matrix.png：混淆矩阵图
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

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


def load_data(features_path: Path, labels_path: Path) -> pd.DataFrame:
    feats = pd.read_csv(features_path, encoding="utf-8")
    labels = pd.read_csv(labels_path, encoding="utf-8")
    df = feats.merge(
        labels[["user_id", "window_start", "window_end", "cluster"]],
        on=["user_id", "window_start", "window_end"],
    )
    return df


def plot_feature_importance(model: xgb.XGBClassifier, output_dir: Path) -> None:
    importance = model.get_booster().get_fscore()
    if not importance:
        return
    imp_df = pd.DataFrame(
        {"feature": list(importance.keys()), "importance": list(importance.values())}
    ).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp_df["feature"], imp_df["importance"], color="steelblue", alpha=0.85)
    ax.set_xlabel("特征重要性（F-score）")
    ax.set_title("XGBoost 特征重要性")
    plt.tight_layout()
    out = output_dir / "feature_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"特征重要性图已保存：{out}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output_dir: Path) -> None:
    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title("混淆矩阵（样本数）")
    thresh = cm.max() / 2
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10,
            )
    plt.tight_layout()
    out = output_dir / "confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"混淆矩阵图已保存：{out}")


def plot_confusion_matrix_prob(
    y_true: pd.Series,
    y_prob: np.ndarray,
    class_names: list[str],
    output_dir: Path,
) -> None:
    """按真实标签分组，计算各类别的平均预测概率矩阵。
    行 = 真实类别，列 = 预测为各类的平均概率，对角线越接近 1 表示模型越确信。
    """
    n_classes = len(class_names)
    prob_matrix = np.zeros((n_classes, n_classes))
    for i, c in enumerate(class_names):
        mask = y_true == c
        prob_matrix[i] = y_prob[mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(prob_matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, format="%.2f")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("预测概率分布")
    ax.set_ylabel("真实标签")
    ax.set_title("平均预测概率矩阵")
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j, i, f"{prob_matrix[i, j]:.2f}",
                ha="center", va="center",
                color="white" if prob_matrix[i, j] > 0.5 else "black",
                fontsize=10,
            )
    plt.tight_layout()
    out = output_dir / "average_probability_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"概率混淆矩阵图已保存：{out}")


def plot_classification_metrics(
    report: dict,
    class_names: list[str],
    output_dir: Path,
) -> None:
    metrics = ["precision", "recall", "f1-score"]
    metric_labels = ["Precision", "Recall", "F1"]
    values = np.array(
        [
            [float(report[class_name][metric]) for metric in metrics]
            for class_name in class_names
        ],
        dtype=float,
    )

    x = np.arange(len(class_names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5.8))
    colors = ["#4c78a8", "#59a14f", "#f28e2b"]
    for index, label in enumerate(metric_labels):
        bars = ax.bar(
            x + (index - 1) * width,
            values[:, index],
            width,
            label=label,
            color=colors[index],
            alpha=0.9,
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)

    accuracy = float(report["accuracy"])
    macro_f1 = float(report["macro avg"]["f1-score"])
    weighted_f1 = float(report["weighted avg"]["f1-score"])
    ax.set_title(
        f"分类模型指标总览  Accuracy={accuracy:.4f}  Macro-F1={macro_f1:.4f}  Weighted-F1={weighted_f1:.4f}"
    )
    ax.set_ylabel("指标值")
    ax.set_ylim(0.94, 1.005)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.28), frameon=False)
    plt.tight_layout()
    out = output_dir / "classification_metrics_overview.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"分类指标总览图已保存：{out}")


def train_and_evaluate(
    df: pd.DataFrame,
    cfg: dict,
    output_dir: Path,
) -> tuple[xgb.XGBClassifier, LabelEncoder, dict]:
    mc = cfg["model"]
    tc = cfg["training"]
    X = df[FEATURE_COLUMNS]
    y = df["cluster"]
    groups = df["user_id"]

    # 支持字符串标签：统一用 LabelEncoder 转为整数给 XGBoost
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)
    n_classes = len(class_names)

    base_params = dict(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=n_classes,
        n_estimators=mc["n_estimators"],
        max_depth=mc["max_depth"],
        learning_rate=mc["learning_rate"],
        subsample=mc["subsample"],
        colsample_bytree=mc["colsample_bytree"],
        min_child_weight=mc["min_child_weight"],
        random_state=mc["random_seed"],
        n_jobs=-1,
        verbosity=0,
    )

    train_idx, test_idx = split_by_user_majority_label(
        df=df,
        test_ratio=float(tc.get("test_ratio", 0.2)),
        random_seed=int(mc["random_seed"]),
    )
    X_train = X.iloc[train_idx]
    y_train = y_enc[train_idx]
    train_groups = groups.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y_enc[test_idx]

    model = xgb.XGBClassifier(**base_params)
    cv_folds = int(tc["cv_folds"])
    cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=mc["random_seed"])
    cv_scores = cross_val_score(model, X_train, y_train, groups=train_groups, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"{cv_folds}折用户分组交叉验证准确率：{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model.fit(X_train, y_train)

    y_pred_enc = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    # 还原为原始标签（字符串或数字）用于报告和图表
    y_test_label = le.inverse_transform(y_test)
    y_pred = le.inverse_transform(y_pred_enc)

    test_acc = accuracy_score(y_test_label, y_pred)
    cm = confusion_matrix(y_test_label, y_pred, labels=class_names)
    report = classification_report(y_test_label, y_pred, labels=class_names, output_dict=True)

    print(f"独立用户测试集准确率：{test_acc:.4f}")
    print(classification_report(y_test_label, y_pred, labels=class_names))

    plot_feature_importance(model, output_dir)
    plot_confusion_matrix(cm, class_names, output_dir)
    # y_prob 的列顺序与 le.classes_ 一致，直接传入
    plot_confusion_matrix_prob(pd.Series(y_test_label), y_prob, class_names, output_dir)
    plot_classification_metrics(report, class_names, output_dir)

    result = {
        "class_names": class_names,
        "split_strategy": "stratified_user_holdout",
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "train_users": int(train_groups.nunique()),
        "test_users": int(groups.iloc[test_idx].nunique()),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "test_accuracy": round(test_acc, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "feature_importance": {
            k: float(v)
            for k, v in sorted(
                model.get_booster().get_fscore().items(),
                key=lambda x: -x[1],
            )
        },
    }
    return model, le, result


def split_by_user_majority_label(
    *,
    df: pd.DataFrame,
    test_ratio: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """按用户划分训练/测试，避免同一用户的滑动窗口同时出现在两侧。"""
    user_labels = (
        df.groupby("user_id")["cluster"]
        .agg(lambda values: values.value_counts().idxmax())
        .reset_index()
    )
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed)
    user_train_idx, user_test_idx = next(splitter.split(user_labels[["user_id"]], user_labels["cluster"]))
    train_users = set(user_labels.iloc[user_train_idx]["user_id"])
    test_users = set(user_labels.iloc[user_test_idx]["user_id"])
    train_idx = np.flatnonzero(df["user_id"].isin(train_users).to_numpy())
    test_idx = np.flatnonzero(df["user_id"].isin(test_users).to_numpy())
    if train_users & test_users:
        raise RuntimeError("用户级训练/测试划分失败：存在用户泄漏")
    return train_idx, test_idx


def parse_args() -> argparse.Namespace:
    default_config = Path(__file__).resolve().parent / "config" / "default.yaml"
    parser = argparse.ArgumentParser(description="训练 XGBoost 居民用电行为分类器")
    parser.add_argument("--config", type=Path, default=default_config)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    project_root = Path(__file__).resolve().parents[2]
    features_path = project_root / cfg["data"]["features_path"]
    labels_path = project_root / cfg["data"]["labels_path"]
    output_dir = project_root / cfg["output"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(features_path, labels_path)
    print(f"样本数：{len(df)}，用户数：{df['user_id'].nunique()}，类别数：{df['cluster'].nunique()}，类别分布：{df['cluster'].value_counts().to_dict()}")

    model, le, report = train_and_evaluate(df, cfg, output_dir)

    model_path = output_dir / "xgboost_model.json"
    model.save_model(model_path)
    print(f"模型已保存：{model_path}")

    with open(output_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print(f"LabelEncoder 已保存：{output_dir / 'label_encoder.pkl'}")

    report_path = output_dir / "classifier_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"评估报告已保存：{report_path}")


if __name__ == "__main__":
    main()

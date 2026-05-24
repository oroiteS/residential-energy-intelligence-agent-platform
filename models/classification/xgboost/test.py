"""XGBoost 分类模型评估、图表和文件输出。"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

try:
    from .config import XGBoostClassifierPaths
except ImportError:  # 兼容直接运行
    from config import XGBoostClassifierPaths


matplotlib.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def plot_feature_importance(model: xgb.XGBClassifier, output_dir: Path) -> None:
    """绘制 XGBoost F-score 特征重要性图。"""

    importance = model.get_booster().get_fscore()
    if not importance:
        return

    importance_df = pd.DataFrame(
        {"feature": list(importance.keys()), "importance": list(importance.values())}
    ).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance_df["feature"], importance_df["importance"], color="steelblue", alpha=0.85)
    ax.set_xlabel("特征重要性（F-score）")
    ax.set_title("XGBoost 特征重要性")
    plt.tight_layout()
    out = output_dir / "feature_importance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"特征重要性图已保存：{out}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output_dir: Path) -> None:
    """绘制样本数形式的混淆矩阵。"""

    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(image)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title("混淆矩阵（样本数）")
    threshold = cm.max() / 2
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=10,
            )
    plt.tight_layout()
    out = output_dir / "confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"混淆矩阵图已保存：{out}")


def plot_confusion_matrix_prob(
    y_true: pd.Series,
    y_prob: np.ndarray,
    class_names: list[str],
    output_dir: Path,
) -> None:
    """按真实标签分组，绘制平均预测概率矩阵。"""

    n_classes = len(class_names)
    prob_matrix = np.zeros((n_classes, n_classes))
    for index, class_name in enumerate(class_names):
        mask = y_true == class_name
        prob_matrix[index] = y_prob[mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(prob_matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(image, format="%.2f")
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
                j,
                i,
                f"{prob_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if prob_matrix[i, j] > 0.5 else "black",
                fontsize=10,
            )
    plt.tight_layout()
    out = output_dir / "average_probability_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"概率混淆矩阵图已保存：{out}")


def plot_classification_metrics(
    report: dict[str, Any],
    class_names: list[str],
    output_dir: Path,
) -> None:
    """绘制每个类别的 Precision、Recall 和 F1 指标。"""

    metrics = ["precision", "recall", "f1-score"]
    metric_labels = ["Precision", "Recall", "F1"]
    values = np.array(
        [[float(report[class_name][metric]) for metric in metrics] for class_name in class_names],
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
    ax.set_ylim(0.0, 1.005)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.28), frameon=False)
    plt.tight_layout()
    out = output_dir / "classification_metrics_overview.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"分类指标总览图已保存：{out}")


def evaluate_model(
    *,
    model: xgb.XGBClassifier,
    label_encoder: LabelEncoder,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    """评估测试集并生成报告和图表。"""

    raw_class_names = list(label_encoder.classes_)
    class_names = [str(class_name) for class_name in raw_class_names]
    y_pred_enc = model.predict(x_test)
    y_prob = model.predict_proba(x_test)
    y_test_label = label_encoder.inverse_transform(y_test)
    y_pred = label_encoder.inverse_transform(y_pred_enc)

    test_acc = accuracy_score(y_test_label, y_pred)
    cm = confusion_matrix(y_test_label, y_pred, labels=raw_class_names)
    report = cast(
        dict[str, Any],
        classification_report(
            y_test_label,
            y_pred,
            labels=raw_class_names,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    )

    print(f"独立用户测试集准确率：{test_acc:.4f}")
    print(
        classification_report(
            y_test_label,
            y_pred,
            labels=raw_class_names,
            target_names=class_names,
            zero_division=0,
        )
    )

    plot_feature_importance(model, output_dir)
    plot_confusion_matrix(cm, class_names, output_dir)
    plot_confusion_matrix_prob(pd.Series(y_test_label).astype(str), y_prob, class_names, output_dir)
    plot_classification_metrics(report, class_names, output_dir)
    feature_scores = model.get_booster().get_fscore()
    feature_importance: dict[str, float] = {}
    for key, raw_value in sorted(
        feature_scores.items(),
        key=lambda item: float(item[1][0]) if isinstance(item[1], list) else float(item[1]),
        reverse=True,
    ):
        value = raw_value[0] if isinstance(raw_value, list) else raw_value
        feature_importance[str(key)] = float(value)

    return {
        "class_names": class_names,
        "test_accuracy": round(float(test_acc), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "feature_importance": feature_importance,
    }


def save_outputs(
    *,
    model: xgb.XGBClassifier,
    label_encoder: LabelEncoder,
    report: dict[str, Any],
    paths: XGBoostClassifierPaths,
) -> None:
    """保存分类模型、LabelEncoder 和评估报告。"""

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(paths.model_path)
    print(f"模型已保存：{paths.model_path}")

    with paths.label_encoder_path.open("wb") as file:
        pickle.dump(label_encoder, file)
    print(f"LabelEncoder 已保存：{paths.label_encoder_path}")

    with paths.report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(f"评估报告已保存：{paths.report_path}")

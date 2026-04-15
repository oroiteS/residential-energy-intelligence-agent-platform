"""XGBoost 训练与评估公共逻辑。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd

from classification.xgboost.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from classification.xgboost.constants import TABULAR_FEATURE_NAMES
from classification.xgboost.dataset import (
    TabularClassificationSample,
    infer_label_vocabulary,
    load_training_samples,
    samples_to_xy,
    split_samples,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

try:
    import xgboost as xgb
except Exception as exc:  # pragma: no cover - 运行时依赖保护
    xgb = None
    XGBOOST_IMPORT_ERROR = exc
else:
    XGBOOST_IMPORT_ERROR = None


def ensure_xgboost_available() -> None:
    if xgb is None:
        raise ImportError(
            "当前环境的 xgboost 不可用，请先完成 xgboost 及其运行时依赖安装"
            "（例如 macOS 上的 libomp），再运行 XGBoost 分类任务。"
        ) from XGBOOST_IMPORT_ERROR


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def compute_macro_f1(predictions: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    f1_scores: list[float] = []
    for class_index in range(num_classes):
        true_positive = np.logical_and(predictions == class_index, targets == class_index).sum()
        false_positive = np.logical_and(predictions == class_index, targets != class_index).sum()
        false_negative = np.logical_and(predictions != class_index, targets == class_index).sum()

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return float(sum(f1_scores) / num_classes)


def compute_metrics(probabilities: np.ndarray, targets: np.ndarray, num_classes: int) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    target_index = targets.astype(np.int64)
    selected_probabilities = np.clip(
        probabilities[np.arange(len(target_index)), target_index],
        1e-12,
        1.0,
    )
    log_loss = float(-np.log(selected_probabilities).mean())
    return {
        "loss": log_loss,
        "accuracy": float((predictions == target_index).mean()),
        "macro_f1": compute_macro_f1(predictions, target_index, num_classes=num_classes),
    }


def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target_index, prediction_index in zip(targets.astype(np.int64), predictions.astype(np.int64), strict=False):
        confusion[target_index, prediction_index] += 1
    return confusion


def compute_per_class_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    label_names: list[str],
) -> list[dict[str, float | int | str]]:
    per_class_metrics: list[dict[str, float | int | str]] = []
    for class_index, label_name in enumerate(label_names):
        true_positive = int(np.logical_and(predictions == class_index, targets == class_index).sum())
        false_positive = int(np.logical_and(predictions == class_index, targets != class_index).sum())
        false_negative = int(np.logical_and(predictions != class_index, targets == class_index).sum())
        support = int((targets == class_index).sum())

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class_metrics.append(
            {
                "label_name": label_name,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1_score),
                "support": support,
            }
        )
    return per_class_metrics


def create_split_matrices(
    data_config: DataConfig,
    seed: int,
) -> dict[str, object]:
    data_frame = pd.read_csv(data_config.data_path)
    label_vocabulary = infer_label_vocabulary(data_frame)
    samples = load_training_samples(data_config.data_path)
    train_samples, val_samples, test_samples = split_samples(
        samples=samples,
        train_ratio=data_config.train_ratio,
        val_ratio=data_config.val_ratio,
        seed=seed,
    )
    x_train, y_train, train_metadata = samples_to_xy(train_samples)
    x_val, y_val, val_metadata = samples_to_xy(val_samples)
    x_test, y_test, test_metadata = samples_to_xy(test_samples)
    return {
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "train_metadata": train_metadata,
        "val_metadata": val_metadata,
        "test_metadata": test_metadata,
        "label_names": label_vocabulary.label_names,
    }


def build_training_params(
    model_config: ModelConfig,
    train_config: TrainConfig,
    num_classes: int,
) -> dict[str, object]:
    params: dict[str, object] = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eta": model_config.learning_rate,
        "max_depth": model_config.max_depth,
        "subsample": model_config.subsample,
        "colsample_bytree": model_config.colsample_bytree,
        "min_child_weight": model_config.min_child_weight,
        "gamma": model_config.gamma,
        "alpha": model_config.reg_alpha,
        "lambda": model_config.reg_lambda,
        "max_bin": model_config.max_bin,
        "tree_method": model_config.tree_method,
        "eval_metric": train_config.eval_metric,
        "seed": train_config.seed,
        "verbosity": 0,
    }
    if model_config.nthread > 0:
        params["nthread"] = model_config.nthread
    return params


def create_dmatrix(
    features: np.ndarray,
    labels: np.ndarray | None = None,
) -> "xgb.DMatrix":
    ensure_xgboost_available()
    matrix = xgb.DMatrix(
        data=features,
        label=labels,
        feature_names=list(TABULAR_FEATURE_NAMES),
    )
    return matrix


def resolve_iteration_range(booster: "xgb.Booster", fallback_rounds: int | None = None) -> tuple[int, int]:
    best_iteration = getattr(booster, "best_iteration", None)
    if best_iteration is not None and int(best_iteration) >= 0:
        return (0, int(best_iteration) + 1)
    total_rounds = fallback_rounds if fallback_rounds is not None else booster.num_boosted_rounds()
    return (0, int(total_rounds))


def predict_probabilities(
    booster: "xgb.Booster",
    features: np.ndarray,
    fallback_rounds: int | None = None,
) -> np.ndarray:
    dmatrix = create_dmatrix(features)
    return booster.predict(
        dmatrix,
        iteration_range=resolve_iteration_range(booster, fallback_rounds=fallback_rounds),
    )


def save_json_summary(output_path: Path, payload: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_model_bundle(
    output_dir: Path,
    booster: "xgb.Booster",
    experiment_config: ExperimentConfig,
    best_metrics: dict[str, float],
    evals_result: dict[str, dict[str, list[float]]],
    label_names: list[str],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "best_model.json"
    metadata_path = output_dir / "model_metadata.json"
    booster.save_model(model_path)

    importance_map = booster.get_score(importance_type="gain")
    importance_items = sorted(
        (
            {"feature_name": feature_name, "gain": float(gain)}
            for feature_name, gain in importance_map.items()
        ),
        key=lambda item: item["gain"],
        reverse=True,
    )
    save_json_summary(
        metadata_path,
        {
            "labels": label_names,
            "tabular_feature_names": list(TABULAR_FEATURE_NAMES),
            "config": experiment_config.to_dict(),
            "metrics": best_metrics,
            "best_iteration": getattr(booster, "best_iteration", None),
            "best_score": (
                float(getattr(booster, "best_score"))
                if getattr(booster, "best_score", None) is not None
                else None
            ),
            "evals_result": evals_result,
            "feature_importance_gain": importance_items,
        },
    )
    return model_path, metadata_path


def load_model(model_path: Path) -> "xgb.Booster":
    ensure_xgboost_available()
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster


def load_metadata(metadata_path: Path) -> dict[str, object]:
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_training_history(
    evals_result: dict[str, dict[str, list[float]]],
    output_path: Path,
) -> Path:
    plt = _import_matplotlib()

    train_metrics = evals_result.get("train", {})
    val_metrics = evals_result.get("val", {})
    metric_name = next(iter(train_metrics.keys()), None)
    if metric_name is None or metric_name not in val_metrics:
        raise ValueError("训练曲线缺少可绘制的评估指标")

    train_values = [float(value) for value in train_metrics[metric_name]]
    val_values = [float(value) for value in val_metrics[metric_name]]
    rounds = list(range(1, len(train_values) + 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(rounds, train_values, label=f"train_{metric_name}", linewidth=2)
    axes[0].plot(rounds, val_values, label=f"val_{metric_name}", linewidth=2)
    axes[0].set_ylabel(metric_name)
    axes[0].set_title("XGBoost Training Curves")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    metric_gap = np.asarray(val_values, dtype=np.float32) - np.asarray(train_values, dtype=np.float32)
    axes[1].plot(rounds, metric_gap, label=f"val_minus_train_{metric_name}", linewidth=2, color="darkorange")
    axes[1].axhline(0.0, linestyle="--", linewidth=1.2, color="gray")
    axes[1].set_xlabel("Boosting Round")
    axes[1].set_ylabel("Gap")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def plot_feature_importance(
    importance_items: list[dict[str, float | str]],
    output_path: Path,
    top_k: int = 20,
) -> Path:
    plt = _import_matplotlib()
    top_items = importance_items[:top_k]
    if not top_items:
        raise ValueError("特征重要性为空，无法绘图")

    feature_names = [str(item["feature_name"]) for item in reversed(top_items)]
    gains = [float(item["gain"]) for item in reversed(top_items)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 8))
    axis.barh(feature_names, gains, color="#2a9d8f")
    axis.set_xlabel("Gain")
    axis.set_title(f"Top {len(top_items)} Feature Importance")
    axis.grid(True, axis="x", linestyle="--", alpha=0.4)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def plot_label_distribution(
    distributions: dict[str, np.ndarray],
    label_names: list[str],
    output_path: Path,
) -> Path:
    plt = _import_matplotlib()

    split_names = list(distributions.keys())
    x_positions = np.arange(len(label_names), dtype=np.float32)
    bar_width = 0.8 / max(len(split_names), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 6))
    for split_index, split_name in enumerate(split_names):
        counts = distributions[split_name]
        offset = (split_index - (len(split_names) - 1) / 2) * bar_width
        axis.bar(
            x_positions + offset,
            counts,
            width=bar_width,
            label=split_name,
            alpha=0.9,
        )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(label_names, rotation=20, ha="right")
    axis.set_ylabel("Sample Count")
    axis.set_title("Label Distribution by Split")
    axis.grid(True, axis="y", linestyle="--", alpha=0.4)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    label_names: list[str],
    output_path: Path,
    normalize: bool = False,
) -> Path:
    plt = _import_matplotlib()

    matrix = confusion_matrix.astype(np.float64)
    if normalize:
        row_sum = matrix.sum(axis=1, keepdims=True)
        safe_row_sum = np.where(row_sum == 0.0, 1.0, row_sum)
        matrix = matrix / safe_row_sum

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="Blues", aspect="auto")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axis.set_xticks(np.arange(len(label_names)))
    axis.set_yticks(np.arange(len(label_names)))
    axis.set_xticklabels(label_names, rotation=25, ha="right")
    axis.set_yticklabels(label_names)
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.set_title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")

    threshold = matrix.max() * 0.55 if matrix.size else 0.0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            display_value = f"{matrix[row_index, column_index]:.2f}" if normalize else str(int(matrix[row_index, column_index]))
            axis.text(
                column_index,
                row_index,
                display_value,
                ha="center",
                va="center",
                color="white" if matrix[row_index, column_index] > threshold else "black",
                fontsize=9,
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def plot_per_class_metrics(
    per_class_metrics: list[dict[str, float | int | str]],
    output_path: Path,
) -> Path:
    plt = _import_matplotlib()

    label_names = [str(item["label_name"]) for item in per_class_metrics]
    precision = [float(item["precision"]) for item in per_class_metrics]
    recall = [float(item["recall"]) for item in per_class_metrics]
    f1_scores = [float(item["f1"]) for item in per_class_metrics]
    supports = [int(item["support"]) for item in per_class_metrics]
    x_positions = np.arange(len(label_names), dtype=np.float32)
    bar_width = 0.24

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    axes[0].bar(x_positions - bar_width, precision, width=bar_width, label="precision", color="#457b9d")
    axes[0].bar(x_positions, recall, width=bar_width, label="recall", color="#2a9d8f")
    axes[0].bar(x_positions + bar_width, f1_scores, width=bar_width, label="f1", color="#e76f51")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Per-class Metrics")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].bar(x_positions, supports, color="#6d597a")
    axes[1].set_ylabel("Support")
    axes[1].set_xlabel("Label")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(label_names, rotation=20, ha="right")

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path

"""XGBoost 训练与评估公共逻辑。"""

from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np

from classification.XGBoost.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from classification.XGBoost.constants import LABELS, TABULAR_FEATURE_NAMES
from classification.XGBoost.dataset import (
    TabularClassificationSample,
    load_training_samples,
    samples_to_xy,
    split_samples,
)

try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover - 运行时依赖保护
    xgb = None
    XGBOOST_IMPORT_ERROR = exc
else:
    XGBOOST_IMPORT_ERROR = None


def ensure_xgboost_available() -> None:
    if xgb is None:
        raise ImportError(
            "当前环境未安装 xgboost，请先在 models 目录执行依赖安装后再运行 XGBoost 分类任务。"
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


def compute_metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
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
        "macro_f1": compute_macro_f1(predictions, target_index, num_classes=len(LABELS)),
    }


def create_split_matrices(
    data_config: DataConfig,
    seed: int,
) -> dict[str, object]:
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
    }


def build_training_params(model_config: ModelConfig, train_config: TrainConfig) -> dict[str, object]:
    params: dict[str, object] = {
        "objective": "multi:softprob",
        "num_class": model_config.num_classes,
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
            "labels": LABELS,
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

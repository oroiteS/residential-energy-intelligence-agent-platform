"""XGBoost 分类训练入口。"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classification.xgboost.config import DEFAULT_CONFIG_PATH, load_experiment_config
from classification.xgboost.engine import (
    build_training_params,
    compute_confusion_matrix,
    compute_metrics,
    compute_per_class_metrics,
    create_dmatrix,
    create_split_matrices,
    ensure_xgboost_available,
    predict_probabilities,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_label_distribution,
    plot_per_class_metrics,
    plot_training_history,
    save_json_summary,
    save_model_bundle,
    set_seed,
)


def run_training(experiment_config) -> dict[str, object]:
    ensure_xgboost_available()
    train_config = experiment_config.train
    model_config = experiment_config.model
    log = tqdm.write

    log("[阶段] 开始准备 XGBoost 训练")
    set_seed(train_config.seed)
    split_payload = create_split_matrices(
        data_config=experiment_config.data,
        seed=train_config.seed,
    )
    x_train = split_payload["x_train"]
    y_train = split_payload["y_train"]
    x_val = split_payload["x_val"]
    y_val = split_payload["y_val"]
    x_test = split_payload["x_test"]
    label_names = split_payload["label_names"]

    log(
        f"[数据] train={len(x_train)} val={len(x_val)} test={len(x_test)} "
        f"feature_dim={x_train.shape[1]}"
    )

    dtrain = create_dmatrix(x_train, y_train)
    dval = create_dmatrix(x_val, y_val)
    evals_result: dict[str, dict[str, list[float]]] = {}
    training_params = build_training_params(
        model_config,
        train_config,
        num_classes=len(label_names),
    )
    log(
        f"[训练] num_boost_round={model_config.num_boost_round} "
        f"early_stopping_rounds={train_config.early_stopping_rounds} "
        f"early_stopping_min_delta={train_config.early_stopping_min_delta} "
        f"eval_metric={train_config.eval_metric} "
        f"tree_method={model_config.tree_method}"
    )

    import xgboost as xgb

    early_stopping_callback = xgb.callback.EarlyStopping(
        rounds=train_config.early_stopping_rounds,
        metric_name=train_config.eval_metric,
        data_name="val",
        save_best=True,
        min_delta=train_config.early_stopping_min_delta,
    )

    start_time = time.time()
    booster = xgb.train(
        params=training_params,
        dtrain=dtrain,
        num_boost_round=model_config.num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        callbacks=[early_stopping_callback],
        evals_result=evals_result,
        verbose_eval=train_config.verbose_eval,
    )
    elapsed_seconds = time.time() - start_time
    log("[阶段] 开始整理训练与验证指标")

    train_probabilities = predict_probabilities(
        booster,
        x_train,
        fallback_rounds=model_config.num_boost_round,
    )
    val_probabilities = predict_probabilities(
        booster,
        x_val,
        fallback_rounds=model_config.num_boost_round,
    )
    train_metrics = compute_metrics(train_probabilities, y_train, num_classes=len(label_names))
    val_metrics = compute_metrics(val_probabilities, y_val, num_classes=len(label_names))
    val_predictions = val_probabilities.argmax(axis=1)
    val_confusion_matrix = compute_confusion_matrix(
        predictions=val_predictions,
        targets=y_val,
        num_classes=len(label_names),
    )
    val_per_class_metrics = compute_per_class_metrics(
        predictions=val_predictions,
        targets=y_val,
        label_names=label_names,
    )
    best_metrics = {
        "best_iteration": getattr(booster, "best_iteration", None),
        "best_score": (
            float(getattr(booster, "best_score"))
            if getattr(booster, "best_score", None) is not None
            else None
        ),
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "train_macro_f1": train_metrics["macro_f1"],
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "elapsed_seconds": elapsed_seconds,
    }

    model_path, metadata_path = save_model_bundle(
        output_dir=train_config.output_dir,
        booster=booster,
        experiment_config=experiment_config,
        best_metrics=best_metrics,
        evals_result=evals_result,
        label_names=label_names,
    )
    importance_items = sorted(
        (
            {"feature_name": feature_name, "gain": float(gain)}
            for feature_name, gain in booster.get_score(importance_type="gain").items()
        ),
        key=lambda item: item["gain"],
        reverse=True,
    )
    training_curves_path = plot_training_history(
        evals_result=evals_result,
        output_path=train_config.output_dir / "training_curves.png",
    )
    feature_importance_path = plot_feature_importance(
        importance_items=importance_items,
        output_path=train_config.output_dir / "feature_importance.png",
    )
    label_distribution_path = plot_label_distribution(
        distributions={
            "train": np.bincount(y_train, minlength=len(label_names)),
            "val": np.bincount(y_val, minlength=len(label_names)),
            "test": np.bincount(split_payload["y_test"], minlength=len(label_names)),
        },
        label_names=label_names,
        output_path=train_config.output_dir / "split_label_distribution.png",
    )
    val_confusion_path = plot_confusion_matrix(
        confusion_matrix=val_confusion_matrix,
        label_names=label_names,
        output_path=train_config.output_dir / "val_confusion_matrix.png",
        normalize=False,
    )
    val_confusion_normalized_path = plot_confusion_matrix(
        confusion_matrix=val_confusion_matrix,
        label_names=label_names,
        output_path=train_config.output_dir / "val_confusion_matrix_normalized.png",
        normalize=True,
    )
    val_class_metrics_path = plot_per_class_metrics(
        per_class_metrics=val_per_class_metrics,
        output_path=train_config.output_dir / "val_per_class_metrics.png",
    )
    log(
        f"[产物] checkpoint={model_path} "
        f"metadata={metadata_path}"
    )
    save_json_summary(
        train_config.output_dir / "training_summary.json",
        {
            "best_metrics": best_metrics,
            "evals_result": evals_result,
            "labels": label_names,
            "val_confusion_matrix": val_confusion_matrix.tolist(),
            "val_per_class_metrics": val_per_class_metrics,
            "config": experiment_config.to_dict(),
            "runtime": {
                "library": "xgboost",
                "eval_metric": train_config.eval_metric,
            },
            "artifacts": {
                "checkpoint": str(model_path),
                "metadata": str(metadata_path),
                "training_curves": str(training_curves_path),
                "feature_importance": str(feature_importance_path),
                "split_label_distribution": str(label_distribution_path),
                "val_confusion_matrix": str(val_confusion_path),
                "val_confusion_matrix_normalized": str(val_confusion_normalized_path),
                "val_per_class_metrics": str(val_class_metrics_path),
            },
        },
    )

    log(
        "训练完成，"
        f"best_iteration={best_metrics['best_iteration']} "
        f"train_acc={best_metrics['train_accuracy']:.4f} "
        f"train_f1={best_metrics['train_macro_f1']:.4f} "
        f"val_acc={best_metrics['val_accuracy']:.4f} "
        f"val_f1={best_metrics['val_macro_f1']:.4f}"
    )
    return {
        "best_metrics": best_metrics,
        "evals_result": evals_result,
    }


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    experiment_config = load_experiment_config(config_path=config_path)
    run_training(experiment_config)


if __name__ == "__main__":
    main()

"""XGBoost 分类训练入口。"""

from __future__ import annotations

from pathlib import Path
import sys
import time

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classification.XGBoost.config import DEFAULT_CONFIG_PATH, load_experiment_config
from classification.XGBoost.engine import (
    build_training_params,
    compute_metrics,
    create_dmatrix,
    create_split_matrices,
    ensure_xgboost_available,
    predict_probabilities,
    save_json_summary,
    save_model_bundle,
    set_seed,
)


def run_training(experiment_config) -> dict[str, object]:
    ensure_xgboost_available()
    train_config = experiment_config.train
    model_config = experiment_config.model

    set_seed(train_config.seed)
    split_payload = create_split_matrices(
        data_config=experiment_config.data,
        seed=train_config.seed,
    )
    x_train = split_payload["x_train"]
    y_train = split_payload["y_train"]
    x_val = split_payload["x_val"]
    y_val = split_payload["y_val"]

    dtrain = create_dmatrix(x_train, y_train)
    dval = create_dmatrix(x_val, y_val)
    evals_result: dict[str, dict[str, list[float]]] = {}
    training_params = build_training_params(model_config, train_config)

    import xgboost as xgb

    start_time = time.time()
    booster = xgb.train(
        params=training_params,
        dtrain=dtrain,
        num_boost_round=model_config.num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=train_config.early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=train_config.verbose_eval,
    )
    elapsed_seconds = time.time() - start_time

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
    train_metrics = compute_metrics(train_probabilities, y_train)
    val_metrics = compute_metrics(val_probabilities, y_val)
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
    )
    save_json_summary(
        train_config.output_dir / "training_summary.json",
        {
            "best_metrics": best_metrics,
            "evals_result": evals_result,
            "config": experiment_config.to_dict(),
            "runtime": {
                "library": "xgboost",
                "eval_metric": train_config.eval_metric,
            },
            "artifacts": {
                "checkpoint": str(model_path),
                "metadata": str(metadata_path),
            },
        },
    )

    print(
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

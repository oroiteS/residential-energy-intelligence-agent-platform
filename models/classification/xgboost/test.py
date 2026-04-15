"""XGBoost 分类测试入口。"""

from __future__ import annotations

from pathlib import Path
import sys
import time

from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classification.xgboost.config import DEFAULT_CONFIG_PATH, load_experiment_config
from classification.xgboost.engine import (
    compute_confusion_matrix,
    compute_metrics,
    compute_per_class_metrics,
    create_split_matrices,
    ensure_xgboost_available,
    load_model,
    load_metadata,
    predict_probabilities,
    plot_confusion_matrix,
    plot_per_class_metrics,
    save_json_summary,
    set_seed,
)


def run_test(experiment_config) -> dict[str, object]:
    ensure_xgboost_available()
    test_config = experiment_config.test
    train_config = experiment_config.train
    log = tqdm.write

    log("[阶段] 开始准备 XGBoost 测试")
    set_seed(train_config.seed)
    model_path = test_config.checkpoint_path or (train_config.output_dir / "best_model.json")
    metadata_path = model_path.with_name("model_metadata.json")
    booster = load_model(model_path)
    metadata = load_metadata(metadata_path)
    log(f"[测试] checkpoint={model_path} metadata={metadata_path}")

    split_payload = create_split_matrices(
        data_config=experiment_config.data,
        seed=train_config.seed,
    )
    x_test = split_payload["x_test"]
    y_test = split_payload["y_test"]
    label_names = split_payload["label_names"]
    log(f"[测试] samples={len(x_test)} best_iteration={metadata.get('best_iteration')}")

    start_time = time.time()
    probabilities = predict_probabilities(
        booster,
        x_test,
        fallback_rounds=experiment_config.model.num_boost_round,
    )
    predictions = probabilities.argmax(axis=1)
    metrics = compute_metrics(probabilities, y_test, num_classes=len(label_names))
    confusion_matrix = compute_confusion_matrix(
        predictions=predictions,
        targets=y_test,
        num_classes=len(label_names),
    )
    per_class_metrics = compute_per_class_metrics(
        predictions=predictions,
        targets=y_test,
        label_names=label_names,
    )
    metrics["elapsed_seconds"] = time.time() - start_time
    confusion_path = plot_confusion_matrix(
        confusion_matrix=confusion_matrix,
        label_names=label_names,
        output_path=test_config.output_dir / "test_confusion_matrix.png",
        normalize=False,
    )
    confusion_normalized_path = plot_confusion_matrix(
        confusion_matrix=confusion_matrix,
        label_names=label_names,
        output_path=test_config.output_dir / "test_confusion_matrix_normalized.png",
        normalize=True,
    )
    per_class_metrics_path = plot_per_class_metrics(
        per_class_metrics=per_class_metrics,
        output_path=test_config.output_dir / "test_per_class_metrics.png",
    )

    summary = {
        "checkpoint_path": str(model_path),
        "metrics": metrics,
        "labels": label_names,
        "confusion_matrix": confusion_matrix.tolist(),
        "per_class_metrics": per_class_metrics,
        "config": experiment_config.to_dict(),
        "runtime": {
            "library": "xgboost",
            "best_iteration": metadata.get("best_iteration"),
        },
        "artifacts": {
            "confusion_matrix": str(confusion_path),
            "confusion_matrix_normalized": str(confusion_normalized_path),
            "per_class_metrics": str(per_class_metrics_path),
        },
    }
    save_json_summary(test_config.output_dir / "test_summary.json", summary)

    log(
        "测试完成，"
        f"test_loss={metrics['loss']:.4f} "
        f"test_acc={metrics['accuracy']:.4f} "
        f"test_f1={metrics['macro_f1']:.4f}"
    )
    return summary


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    experiment_config = load_experiment_config(config_path=config_path)
    run_test(experiment_config)


if __name__ == "__main__":
    main()

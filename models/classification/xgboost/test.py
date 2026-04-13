"""XGBoost 分类测试入口。"""

from __future__ import annotations

from pathlib import Path
import sys
import time

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classification.XGBoost.config import DEFAULT_CONFIG_PATH, load_experiment_config
from classification.XGBoost.engine import (
    compute_metrics,
    create_split_matrices,
    ensure_xgboost_available,
    load_model,
    load_metadata,
    predict_probabilities,
    save_json_summary,
    set_seed,
)


def run_test(experiment_config) -> dict[str, object]:
    ensure_xgboost_available()
    test_config = experiment_config.test
    train_config = experiment_config.train

    set_seed(train_config.seed)
    model_path = test_config.checkpoint_path or (train_config.output_dir / "best_model.json")
    metadata_path = model_path.with_name("model_metadata.json")
    booster = load_model(model_path)
    metadata = load_metadata(metadata_path)

    split_payload = create_split_matrices(
        data_config=experiment_config.data,
        seed=train_config.seed,
    )
    x_test = split_payload["x_test"]
    y_test = split_payload["y_test"]

    start_time = time.time()
    probabilities = predict_probabilities(
        booster,
        x_test,
        fallback_rounds=experiment_config.model.num_boost_round,
    )
    metrics = compute_metrics(probabilities, y_test)
    metrics["elapsed_seconds"] = time.time() - start_time

    summary = {
        "checkpoint_path": str(model_path),
        "metrics": metrics,
        "config": experiment_config.to_dict(),
        "runtime": {
            "library": "xgboost",
            "best_iteration": metadata.get("best_iteration"),
        },
    }
    save_json_summary(test_config.output_dir / "test_summary.json", summary)

    print(
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

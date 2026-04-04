"""Decoder-only Transformer 预测测试入口。"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forecast.GPT.config import DEFAULT_CONFIG_PATH, detect_device, load_experiment_config
from forecast.GPT.engine import (
    build_criterion,
    build_model,
    checkpoint_to_normalization,
    create_eval_loader,
    create_split_datasets,
    describe_loss,
    evaluate,
    load_checkpoint,
    save_json_summary,
    set_seed,
)


def run_test(experiment_config) -> dict[str, object]:
    test_config = experiment_config.test
    train_config = experiment_config.train

    set_seed(train_config.seed)

    device_name = detect_device()
    device = torch.device(device_name)
    checkpoint_path = test_config.checkpoint_path or (
        train_config.output_dir / "best_model.pt"
    )
    checkpoint = load_checkpoint(checkpoint_path, device)
    normalization = checkpoint_to_normalization(checkpoint)

    _, _, test_dataset = create_split_datasets(
        data_config=experiment_config.data,
        seed=train_config.seed,
        normalization=normalization,
    )
    test_loader = create_eval_loader(test_dataset, batch_size=test_config.batch_size)

    model = build_model(experiment_config.model).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = build_criterion(train_config)
    loss_name = describe_loss(train_config)

    start_time = time.time()
    metrics = evaluate(model, test_loader, criterion, device)
    metrics["elapsed_seconds"] = time.time() - start_time

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "metrics": metrics,
        "config": experiment_config.to_dict(),
        "runtime": {
            "device": device_name,
            "loss": loss_name,
            "aggregate_normalization": normalization.aggregate_mode,
        },
    }
    save_json_summary(test_config.output_dir / "test_summary.json", summary)

    print(
        "测试完成，"
        f"test_loss={metrics['loss']:.4f} "
        f"test_mae={metrics['mae']:.4f} "
        f"test_rmse={metrics['rmse']:.4f} "
        f"test_smape={metrics['smape']:.4f} "
        f"test_wape={metrics['wape']:.4f} "
        f"loss={loss_name} "
        f"device={device_name}"
    )
    return summary


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    experiment_config = load_experiment_config(config_path=config_path)
    run_test(experiment_config)


if __name__ == "__main__":
    main()


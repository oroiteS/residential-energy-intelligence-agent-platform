#!/usr/bin/env python3
"""PyTorch Lightning 测试入口。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .data import LSTMForecastDataModule, load_config, resolve_path
    from .model import LSTMResidualForecaster
except ImportError:  # 兼容 `python forecast/lstm/test.py`
    from data import LSTMForecastDataModule, load_config, resolve_path
    from model import LSTMResidualForecaster

from forecast.visualization.comparison import generate_comparison_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试 LSTM 残差预测模型")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "default.yaml",
        help="测试配置 YAML 路径",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="模型 checkpoint 路径，默认自动寻找 best")
    return parser.parse_args()


def find_checkpoint(config: dict, explicit_checkpoint: Path | None) -> Path:
    if explicit_checkpoint is not None:
        return explicit_checkpoint

    configured_checkpoint = config.get("test", {}).get("checkpoint_path")
    if configured_checkpoint:
        return resolve_path(configured_checkpoint)

    checkpoint_dir = resolve_path(config["output"]["output_dir"]) / config["output"]["checkpoint_dir"]
    mode = config.get("test", {}).get("checkpoint_mode", "auto")
    if mode not in {"auto", "best", "last"}:
        raise ValueError(f"未知 checkpoint_mode：{mode}")

    if mode in {"auto", "best"}:
        best = sorted(checkpoint_dir.glob("best-*.ckpt"))
        if best:
            return best[-1]
        if mode == "best":
            raise FileNotFoundError(f"没有找到 best checkpoint：{checkpoint_dir}")

    if mode in {"auto", "last"}:
        last = checkpoint_dir / "last.ckpt"
        if last.exists():
            return last
        if mode == "last":
            raise FileNotFoundError(f"没有找到 last checkpoint：{last}")

    raise FileNotFoundError(f"没有找到可测试的 checkpoint：{checkpoint_dir}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_columns: list[str]) -> list[dict[str, float | str]]:
    rows = []
    for index, target in enumerate(target_columns):
        true = y_true[:, index]
        pred = y_pred[:, index]
        mse = float(np.mean((true - pred) ** 2))
        mask = np.abs(true) > 1e-8
        mape = float(np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100) if np.any(mask) else float("nan")
        rows.append(
            {
                "target": target,
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "mae": float(np.mean(np.abs(true - pred))),
                "mape": mape,
            }
        )
    return rows


def write_metrics(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with path.with_suffix(".json").open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)


def write_predictions(
    path: Path,
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_columns: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for target in target_columns:
        fieldnames.extend([f"{target}_true", f"{target}_pred"])

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for true_row, pred_row in zip(y_true, y_pred, strict=True):
            row = {}
            for index, target in enumerate(target_columns):
                row[f"{target}_true"] = float(true_row[index])
                row[f"{target}_pred"] = float(pred_row[index])
            writer.writerow(row)


def test(config_path: Path, checkpoint_path: Path | None = None) -> None:
    config = load_config(config_path)
    datamodule = LSTMForecastDataModule(config)
    datamodule.setup("test")
    ckpt = find_checkpoint(config, checkpoint_path)
    # 兼容 PyTorch 2.6+：旧 checkpoint 的 hparams 里可能包含 numpy 对象。
    # checkpoint 来自本项目训练产物，属于可信来源。
    model = LSTMResidualForecaster.load_from_checkpoint(ckpt, weights_only=False)
    model.eval()

    predictions = []
    actuals = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            batch = {key: value.to(device) for key, value in batch.items()}
            predictions.append(model.predict_final(batch).cpu().numpy())
            actuals.append(batch["actual"].cpu().numpy())

    y_pred = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(actuals, axis=0)

    rows = compute_metrics(y_true, y_pred, datamodule.target_columns)
    test_config = config.get("test", {})
    metrics_file = test_config.get("metrics_file", config["output"]["metrics_file"])
    metrics_path = resolve_path(config["output"]["output_dir"]) / metrics_file
    write_metrics(metrics_path, rows)
    print(f"测试指标已保存：{metrics_path}")

    print("\n逐目标指标：")
    header = f"{'target':<22} {'rmse':>10} {'mae':>10} {'mape':>10}"
    print(header)
    for r in rows:
        print(f"{r['target']:<22} {r['rmse']:>10.4f} {r['mae']:>10.4f} {r['mape']:>9.2f}%")

    if bool(test_config.get("save_predictions", False)):
        predictions_path = resolve_path(config["output"]["output_dir"]) / test_config["predictions_file"]
        write_predictions(
            predictions_path,
            y_true=y_true,
            y_pred=y_pred,
            target_columns=datamodule.target_columns,
        )
        print(f"测试预测明细已保存：{predictions_path}")

    try:
        saved = generate_comparison_figures()
        print(f"五模型对比主图已刷新，共生成 {len(saved)} 个文件")
    except Exception as exc:
        print(f"刷新五模型对比主图失败：{exc}")


def main() -> None:
    args = parse_args()
    test(args.config, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()

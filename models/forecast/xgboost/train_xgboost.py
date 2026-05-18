#!/usr/bin/env python3
"""XGBoost 预测模型训练脚本。

使用 `30 天历史 -> 未来 7 天` 的监督学习样本表，分别训练 7 个
XGBoost 回归器。这样早停、断点恢复和单日误差分析都比较清楚。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "缺少训练依赖。请先运行 `uv sync` 安装 pyproject.toml 中的依赖。"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SplitData:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


class BoosterCheckpoint(xgb.callback.TrainingCallback):
    """定期保存 Booster，支持中断后继续训练。"""

    def __init__(self, checkpoint_path: Path, interval: int) -> None:
        self.checkpoint_path = checkpoint_path
        self.interval = interval

    def after_iteration(self, model: xgb.Booster, epoch: int, evals_log: dict[str, Any]) -> bool:
        if self.interval > 0 and (epoch + 1) % self.interval == 0:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_model(self.checkpoint_path)
        return False


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def time_split(df: pd.DataFrame, date_column: str, validation_ratio: float, test_ratio: float) -> SplitData:
    data = df.copy()
    data[date_column] = pd.to_datetime(data[date_column])
    unique_dates = sorted(data[date_column].dropna().unique())
    if len(unique_dates) < 3:
        raise ValueError("可用于时间切分的日期数量太少")

    test_size = max(1, int(len(unique_dates) * test_ratio))
    valid_size = max(1, int(len(unique_dates) * validation_ratio))
    train_size = len(unique_dates) - valid_size - test_size
    if train_size <= 0:
        raise ValueError("训练集日期数量不足，请调小验证集或测试集比例")

    valid_start = unique_dates[train_size]
    test_start = unique_dates[train_size + valid_size]

    train = data[data[date_column] < valid_start].copy()
    valid = data[(data[date_column] >= valid_start) & (data[date_column] < test_start)].copy()
    test = data[data[date_column] >= test_start].copy()
    return SplitData(train=train, valid=valid, test=test)


def user_split(
    df: pd.DataFrame,
    user_column: str,
    validation_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> SplitData:
    """按用户划分训练/验证/测试，避免同一用户滑动窗口泄漏到不同集合。"""
    if user_column not in df.columns:
        raise ValueError(f"用户级切分需要字段 {user_column}")

    rng = np.random.default_rng(random_seed)
    users = np.array(sorted(df[user_column].dropna().unique()))
    if len(users) < 3:
        raise ValueError("可用于用户级切分的用户数量太少")
    rng.shuffle(users)

    test_size = max(1, int(len(users) * test_ratio))
    valid_size = max(1, int(len(users) * validation_ratio))
    train_size = len(users) - valid_size - test_size
    if train_size <= 0:
        raise ValueError("训练集用户数量不足，请调小验证集或测试集比例")

    train_users = set(users[:train_size])
    valid_users = set(users[train_size : train_size + valid_size])
    test_users = set(users[train_size + valid_size :])
    if train_users & valid_users or train_users & test_users or valid_users & test_users:
        raise RuntimeError("用户级训练/验证/测试划分失败：存在用户泄漏")

    return SplitData(
        train=df[df[user_column].isin(train_users)].copy(),
        valid=df[df[user_column].isin(valid_users)].copy(),
        test=df[df[user_column].isin(test_users)].copy(),
    )


def select_feature_columns(
    df: pd.DataFrame,
    *,
    meta_columns: list[str],
    target_columns: list[str],
    feature_config: dict[str, Any] | None = None,
) -> list[str]:
    excluded = set(meta_columns) | set(target_columns)
    if feature_config:
        include_prefixes = tuple(feature_config.get("include_prefixes", []))
        include_columns = list(feature_config.get("include_columns", []))
        exclude_columns = set(feature_config.get("exclude_columns", []))
        prefixed_columns = [
            column
            for column in df.columns
            if include_prefixes and column.startswith(include_prefixes)
        ]
        feature_columns = prefixed_columns + include_columns
        feature_columns = [
            column
            for column in dict.fromkeys(feature_columns)
            if column not in excluded and column not in exclude_columns
        ]
        missing_columns = [column for column in feature_columns if column not in df.columns]
        if missing_columns:
            raise ValueError(f"配置中的特征列不存在：{missing_columns}")
    else:
        feature_columns = [column for column in df.columns if column not in excluded]

    non_numeric = [
        column for column in feature_columns if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric:
        raise ValueError(f"特征列中存在非数值字段，请先处理：{non_numeric}")
    return feature_columns


def to_dmatrix(frame: pd.DataFrame, feature_columns: list[str], target_column: str | None = None) -> xgb.DMatrix:
    label = frame[target_column].to_numpy(dtype=float) if target_column else None
    return xgb.DMatrix(frame[feature_columns].to_numpy(dtype=float), label=label, feature_names=feature_columns)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.abs(y_true) > 1e-8
    if not np.any(mask):
        return math.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def train_one_target(
    *,
    target_column: str,
    split: SplitData,
    feature_columns: list[str],
    params: dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
    checkpoint_interval: int,
    resume: bool,
    model_dir: Path,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    target_name = safe_name(target_column)
    final_model_path = model_dir / f"{target_name}.json"
    checkpoint_path = checkpoint_dir / f"{target_name}.checkpoint.json"

    dtrain = to_dmatrix(split.train, feature_columns, target_column)
    dvalid = to_dmatrix(split.valid, feature_columns, target_column)
    dtest = to_dmatrix(split.test, feature_columns, target_column)

    resume_model = None
    if resume:
        if checkpoint_path.exists():
            resume_model = checkpoint_path
            print(f"检测到断点，继续训练：{checkpoint_path}")
        elif final_model_path.exists():
            resume_model = final_model_path
            print(f"检测到已有模型，继续训练：{final_model_path}")

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=early_stopping_rounds,
        xgb_model=str(resume_model) if resume_model else None,
        callbacks=[BoosterCheckpoint(checkpoint_path, checkpoint_interval)],
        verbose_eval=50,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(final_model_path)
    booster.save_model(checkpoint_path)

    test_true = split.test[target_column].to_numpy(dtype=float)
    test_pred = booster.predict(dtest)
    valid_true = split.valid[target_column].to_numpy(dtype=float)
    valid_pred = booster.predict(dvalid)

    result = {
        "target": target_column,
        "best_iteration": int(getattr(booster, "best_iteration", 0) or 0),
        "best_score": float(getattr(booster, "best_score", math.nan)),
        "valid_mse": mse(valid_true, valid_pred),
        "valid_rmse": rmse(valid_true, valid_pred),
        "valid_mae": mae(valid_true, valid_pred),
        "valid_mape": mape(valid_true, valid_pred),
        "test_mse": mse(test_true, test_pred),
        "test_rmse": rmse(test_true, test_pred),
        "test_mae": mae(test_true, test_pred),
        "test_mape": mape(test_true, test_pred),
        "model_path": str(final_model_path),
        "checkpoint_path": str(checkpoint_path),
    }
    print(
        f"{target_column} 完成："
        f"valid_rmse={result['valid_rmse']:.4f}, test_rmse={result['test_rmse']:.4f}"
    )
    return result


def evaluate_peak_valley_baseline(
    df_test: pd.DataFrame,
    target_columns: list[str],
) -> dict[str, Any]:
    """用历史峰时占比（中位数）推导峰/谷预测，计算逐天 baseline 误差。"""
    output_days = len(target_columns)
    peak_ratio = df_test["hist_peak_ratio"].to_numpy(dtype=float)

    per_day: list[dict[str, Any]] = []
    for day in range(1, output_days + 1):
        total_pred = df_test[f"y_energy_d{day:02d}"].to_numpy(dtype=float)
        peak_pred = total_pred * peak_ratio
        valley_pred = total_pred * (1.0 - peak_ratio)

        peak_true = df_test[f"y_peak_d{day:02d}"].to_numpy(dtype=float)
        valley_true = df_test[f"y_valley_d{day:02d}"].to_numpy(dtype=float)

        peak_mse_val = mse(peak_true, peak_pred)
        valley_mse_val = mse(valley_true, valley_pred)
        per_day.append({
            "target": f"d{day:02d}",
            "peak_mse": round(peak_mse_val, 6),
            "peak_rmse": round(math.sqrt(peak_mse_val), 6),
            "peak_mae": round(mae(peak_true, peak_pred), 6),
            "peak_mape": round(mape(peak_true, peak_pred), 4),
            "valley_mse": round(valley_mse_val, 6),
            "valley_rmse": round(math.sqrt(valley_mse_val), 6),
            "valley_mae": round(mae(valley_true, valley_pred), 6),
            "valley_mape": round(mape(valley_true, valley_pred), 4),
        })

    all_peak_mse = [d["peak_mse"] for d in per_day]
    all_valley_mse = [d["valley_mse"] for d in per_day]
    return {
        "method": "hist_peak_ratio_baseline",
        "per_day": per_day,
        "peak_avg_rmse": round(float(np.mean([d["peak_rmse"] for d in per_day])), 6),
        "peak_avg_mae": round(float(np.mean([d["peak_mae"] for d in per_day])), 6),
        "peak_avg_mape": round(float(np.mean([d["peak_mape"] for d in per_day])), 4),
        "valley_avg_rmse": round(float(np.mean([d["valley_rmse"] for d in per_day])), 6),
        "valley_avg_mae": round(float(np.mean([d["valley_mae"] for d in per_day])), 6),
        "valley_avg_mape": round(float(np.mean([d["valley_mape"] for d in per_day])), 4),
    }


def write_metrics(metrics: list[dict[str, Any]], metrics_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    json_path = metrics_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)


def write_feature_columns(feature_columns: list[str], feature_path: Path) -> None:
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.write_text("\n".join(feature_columns) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 30 天预测 7 天的 XGBoost 模型")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "default.yaml",
        help="训练配置 YAML 路径",
    )
    parser.add_argument("--no-resume", action="store_true", help="忽略已有断点，从头训练")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_config = config["data"]
    feature_config = config.get("features")
    split_config = config["split"]
    train_config = config["training"]
    output_config = config["output"]

    dataset_path = resolve_path(data_config["dataset_path"])
    output_dir = resolve_path(output_config["output_dir"])
    model_dir = output_dir / output_config["model_dir"]
    checkpoint_dir = output_dir / output_config["checkpoint_dir"]
    metrics_path = output_dir / output_config["metrics_file"]
    feature_path = output_dir / output_config["feature_file"]

    df = pd.read_csv(dataset_path)
    target_columns = list(data_config["target_columns"])
    meta_columns = list(data_config["meta_columns"])
    feature_columns = select_feature_columns(
        df,
        meta_columns=meta_columns,
        target_columns=target_columns,
        feature_config=feature_config,
    )
    split_strategy = split_config.get("strategy", "user")
    if split_strategy == "time":
        split = time_split(
            df,
            date_column=data_config["date_column"],
            validation_ratio=float(split_config["validation_ratio"]),
            test_ratio=float(split_config["test_ratio"]),
        )
    elif split_strategy == "user":
        split = user_split(
            df,
            user_column="user_id",
            validation_ratio=float(split_config["validation_ratio"]),
            test_ratio=float(split_config["test_ratio"]),
            random_seed=int(train_config["random_seed"]),
        )
    else:
        raise ValueError(f"不支持的切分策略：{split_strategy}")

    print(f"数据集：{dataset_path}")
    print(f"切分策略：{split_strategy}")
    print(f"样本数：train={len(split.train)}, valid={len(split.valid)}, test={len(split.test)}")
    print(
        "用户数："
        f"train={split.train['user_id'].nunique()}, "
        f"valid={split.valid['user_id'].nunique()}, "
        f"test={split.test['user_id'].nunique()}"
    )
    print(f"特征数：{len(feature_columns)}")
    print(f"目标数：{len(target_columns)}")

    write_feature_columns(feature_columns, feature_path)

    params = dict(config["xgboost_params"])
    params["seed"] = int(train_config["random_seed"])
    resume = bool(train_config["resume"]) and not args.no_resume

    metrics: list[dict[str, Any]] = []
    split_summary = {
        "split_strategy": split_strategy,
        "train_samples": len(split.train),
        "valid_samples": len(split.valid),
        "test_samples": len(split.test),
        "train_users": int(split.train["user_id"].nunique()),
        "valid_users": int(split.valid["user_id"].nunique()),
        "test_users": int(split.test["user_id"].nunique()),
    }
    for target_column in target_columns:
        result = train_one_target(
            target_column=target_column,
            split=split,
            feature_columns=feature_columns,
            params=params,
            num_boost_round=int(train_config["num_boost_round"]),
            early_stopping_rounds=int(train_config["early_stopping_rounds"]),
            checkpoint_interval=int(train_config["checkpoint_interval"]),
            resume=resume,
            model_dir=model_dir,
            checkpoint_dir=checkpoint_dir,
        )
        metrics.append({**split_summary, **result})

    write_metrics(metrics, metrics_path)
    print(f"指标已保存：{metrics_path}")
    print(f"特征列已保存：{feature_path}")

    # 峰谷 baseline 评估（仅需测试集，不依赖 XGBoost 预测）
    baseline = evaluate_peak_valley_baseline(split.test, target_columns)
    baseline_path = output_dir / "peak_valley_baseline.json"
    with baseline_path.open("w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)
    print(f"\n峰谷 baseline 指标（hist_peak_ratio 推导）：")
    header = f"{'day':<6} {'peak_rmse':>10} {'peak_mae':>10} {'peak_mape':>10} {'valley_rmse':>12} {'valley_mae':>11} {'valley_mape':>12}"
    print(header)
    for d in baseline["per_day"]:
        print(
            f"{d['target']:<6} {d['peak_rmse']:>10.4f} {d['peak_mae']:>10.4f} {d['peak_mape']:>9.2f}%"
            f" {d['valley_rmse']:>12.4f} {d['valley_mae']:>11.4f} {d['valley_mape']:>11.2f}%"
        )
    print(
        f"{'avg':<6} {baseline['peak_avg_rmse']:>10.4f} {baseline['peak_avg_mae']:>10.4f} {baseline['peak_avg_mape']:>9.2f}%"
        f" {baseline['valley_avg_rmse']:>12.4f} {baseline['valley_avg_mae']:>11.4f} {baseline['valley_avg_mape']:>11.2f}%"
    )


if __name__ == "__main__":
    main()

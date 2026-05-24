"""XGBoost 预测模型训练单元。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import xgboost as xgb
from xgboost.callback import TrainingCallback

try:
    from .data import SplitData, to_dmatrix
    from .test import mae, mape, mse, rmse
except ImportError:  # 兼容直接运行
    import importlib

    _data = importlib.import_module("data")
    _test = importlib.import_module("test")
    SplitData = getattr(_data, "SplitData")
    to_dmatrix = getattr(_data, "to_dmatrix")
    mae = getattr(_test, "mae")
    mape = getattr(_test, "mape")
    mse = getattr(_test, "mse")
    rmse = getattr(_test, "rmse")

class BoosterCheckpoint(TrainingCallback):
    """定期保存 Booster，支持中断后继续训练。

    XGBoost 训练可能耗时较长，checkpoint 可以避免中途中断后完全重训。
    """

    def __init__(self, checkpoint_path: Path, interval: int) -> None:
        self.checkpoint_path = checkpoint_path
        self.interval = interval

    def after_iteration(self, model: xgb.Booster, epoch: int, evals_log: dict[str, Any]) -> bool:
        if self.interval > 0 and (epoch + 1) % self.interval == 0:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_model(self.checkpoint_path)
        return False


def safe_name(name: str) -> str:
    """把目标列名转换成可作为文件名的安全字符串。"""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def train_one_target(
    *,
    target_column: str,
    split: Any,
    feature_columns: list[str],
    params: dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
    checkpoint_interval: int,
    resume: bool,
    model_dir: Path,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    """训练单个预测日的 XGBoost 回归器。

    当前预测任务采用“未来 7 天每天一个模型”的结构，这样每个目标的
    早停轮数、断点和测试误差都能单独观察。
    """

    target_name = safe_name(target_column)
    final_model_path = model_dir / f"{target_name}.json"
    checkpoint_path = checkpoint_dir / f"{target_name}.checkpoint.json"

    dtrain = to_dmatrix(split.train, feature_columns, target_column)
    dvalid = to_dmatrix(split.valid, feature_columns, target_column)
    dtest = to_dmatrix(split.test, feature_columns, target_column)

    # 支持从 checkpoint 或最终模型继续训练。
    # 这对多目标训练很有用，因为 7 个目标模型可以独立恢复。
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

    # 每个目标列单独保存一个模型文件。
    # 例如 y_energy_d01.json 表示未来第 1 天总用电量预测器。
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
        "best_score": float(getattr(booster, "best_score", float("nan"))),
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

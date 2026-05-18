"""LSTM 基线预测任务的数据模块。

XGBoost 预测 7 天总用电，ratio 推导峰/谷 → 21 维 baseline。
LSTM 学习这 21 维的残差。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import yaml
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SplitData:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class FeatureSpec:
    sequence_columns: list[list[str]]
    future_columns: list[str]
    static_columns: list[str]


class ForecastDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        *,
        sequence: np.ndarray,
        future: np.ndarray,
        static: np.ndarray,
        residual_target: np.ndarray,
        baseline: np.ndarray,
        actual: np.ndarray,
    ) -> None:
        self.sequence = torch.from_numpy(sequence.astype(np.float32))
        self.future = torch.from_numpy(future.astype(np.float32))
        self.static = torch.from_numpy(static.astype(np.float32))
        self.residual_target = torch.from_numpy(residual_target.astype(np.float32))
        self.baseline = torch.from_numpy(baseline.astype(np.float32))
        self.actual = torch.from_numpy(actual.astype(np.float32))

    def __len__(self) -> int:
        return len(self.actual)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "sequence": self.sequence[index],
            "future": self.future[index],
            "static": self.static[index],
            "residual_target": self.residual_target[index],
            "baseline": self.baseline[index],
            "actual": self.actual[index],
        }


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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
    return SplitData(
        train=data[data[date_column] < valid_start].copy(),
        valid=data[(data[date_column] >= valid_start) & (data[date_column] < test_start)].copy(),
        test=data[data[date_column] >= test_start].copy(),
    )


def build_feature_spec(config: dict[str, Any]) -> FeatureSpec:
    input_days = int(config["input_days"])
    sequence_columns = [
        [template.format(day=day) for template in config["sequence_feature_templates"]]
        for day in range(1, input_days + 1)
    ]
    future_days = 7
    future_columns = [
        template.format(day=day)
        for day in range(1, future_days + 1)
        for template in config["future_feature_templates"]
    ]
    return FeatureSpec(
        sequence_columns=sequence_columns,
        future_columns=future_columns,
        static_columns=list(config["static_columns"]),
    )


def validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"缺少字段：{missing}")


def read_xgboost_feature_columns(feature_file: Path) -> list[str]:
    return [
        line.strip()
        for line in feature_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def predict_xgboost_baseline(
    frame: pd.DataFrame,
    *,
    model_dir: Path,
    feature_columns: list[str],
    xgboost_targets: list[str],
) -> np.ndarray:
    """用 7 个 XGBoost 模型预测 7 天总用电量，返回 (n, 7)。"""
    validate_columns(frame, feature_columns)
    dmatrix = xgb.DMatrix(
        frame[feature_columns].to_numpy(dtype=float),
        feature_names=feature_columns,
    )
    predictions = []
    for target in xgboost_targets:
        model_path = model_dir / f"{target}.json"
        if not model_path.exists():
            raise FileNotFoundError(f"缺少 XGBoost baseline 模型：{model_path}")
        booster = xgb.Booster()
        booster.load_model(model_path)
        predictions.append(booster.predict(dmatrix))
    return np.stack(predictions, axis=1)


def get_baseline_predictions(
    split: SplitData,
    *,
    baseline_config: dict[str, Any],
    xgboost_targets: list[str],
) -> dict[str, np.ndarray]:
    """获取 XGBoost 7 天总用电预测，返回 (n, 7)。"""
    if not baseline_config.get("enabled", False):
        return {
            "train": np.zeros((len(split.train), len(xgboost_targets)), dtype=float),
            "valid": np.zeros((len(split.valid), len(xgboost_targets)), dtype=float),
            "test": np.zeros((len(split.test), len(xgboost_targets)), dtype=float),
        }

    cache_path = resolve_path(baseline_config["cache_file"])
    use_cache = bool(baseline_config.get("use_cache", True))
    if use_cache and cache_path.exists():
        loaded = np.load(cache_path)
        return {key: loaded[key] for key in ["train", "valid", "test"]}

    model_dir = resolve_path(baseline_config["xgboost_model_dir"])
    feature_file = resolve_path(baseline_config["xgboost_feature_file"])
    feature_columns = read_xgboost_feature_columns(feature_file)
    predictions = {
        "train": predict_xgboost_baseline(
            split.train,
            model_dir=model_dir,
            feature_columns=feature_columns,
            xgboost_targets=xgboost_targets,
        ),
        "valid": predict_xgboost_baseline(
            split.valid,
            model_dir=model_dir,
            feature_columns=feature_columns,
            xgboost_targets=xgboost_targets,
        ),
        "test": predict_xgboost_baseline(
            split.test,
            model_dir=model_dir,
            feature_columns=feature_columns,
            xgboost_targets=xgboost_targets,
        ),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **predictions)
    return predictions


def expand_baseline_to_21(xgboost_total: np.ndarray, hist_peak_ratio: np.ndarray) -> np.ndarray:
    """将 7 维 XGBoost 总用电 baseline 用 ratio 扩展为 21 维（总+峰+谷）。"""
    ratio = hist_peak_ratio.reshape(-1, 1)  # (n, 1)
    peak_baseline = xgboost_total * ratio
    valley_baseline = xgboost_total * (1.0 - ratio)
    return np.concatenate([xgboost_total, peak_baseline, valley_baseline], axis=1)


def build_raw_arrays(
    frame: pd.DataFrame,
    spec: FeatureSpec,
    target_columns: list[str],
    xgboost_baseline: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构建原始数组，返回 (sequence, future, static, residual_21, actual_21, baseline_21)。"""
    sequence = np.stack(
        [frame[day_columns].to_numpy(dtype=float) for day_columns in spec.sequence_columns],
        axis=1,
    )
    future = frame[spec.future_columns].to_numpy(dtype=float)
    static = frame[spec.static_columns].to_numpy(dtype=float)

    hist_peak_ratio = frame["hist_peak_ratio"].to_numpy(dtype=float)
    baseline_21 = expand_baseline_to_21(xgboost_baseline, hist_peak_ratio)

    actual = frame[target_columns].to_numpy(dtype=float)
    residual = actual - baseline_21
    return sequence, future, static, residual, actual, baseline_21


class LSTMForecastDataModule(L.LightningDataModule):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.target_columns = list(config["data"]["target_columns"])
        # 前 7 个目标为总用电（XGBoost 直接预测），后 14 个为峰/谷（ratio 推导）
        self.xgboost_targets = self.target_columns[:7]
        self.feature_spec = build_feature_spec(config["features"])
        self.sequence_feature_size = len(self.feature_spec.sequence_columns[0])
        self.future_feature_size = len(self.feature_spec.future_columns)
        self.static_feature_size = len(self.feature_spec.static_columns)
        self.output_size = len(self.target_columns)
        self.scalers: dict[str, StandardScaler] = {}

    def setup(self, stage: str | None = None) -> None:
        if hasattr(self, "train_dataset") and hasattr(self, "valid_dataset") and hasattr(self, "test_dataset"):
            return

        df = pd.read_csv(resolve_path(self.config["data"]["dataset_path"]))
        all_columns = [column for day in self.feature_spec.sequence_columns for column in day]
        all_columns += self.feature_spec.future_columns + self.feature_spec.static_columns + self.target_columns
        validate_columns(df, all_columns)

        split = time_split(
            df,
            date_column=self.config["data"]["date_column"],
            validation_ratio=float(self.config["split"]["validation_ratio"]),
            test_ratio=float(self.config["split"]["test_ratio"]),
        )
        baseline = get_baseline_predictions(
            split,
            baseline_config=self.config["baseline"],
            xgboost_targets=self.xgboost_targets,
        )

        train_raw = build_raw_arrays(split.train, self.feature_spec, self.target_columns, baseline["train"])
        valid_raw = build_raw_arrays(split.valid, self.feature_spec, self.target_columns, baseline["valid"])
        test_raw = build_raw_arrays(split.test, self.feature_spec, self.target_columns, baseline["test"])

        train_dataset, valid_dataset, test_dataset, self.scalers = self._make_datasets(
            train_raw=train_raw,
            valid_raw=valid_raw,
            test_raw=test_raw,
        )
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

    def _make_datasets(
        self,
        *,
        train_raw: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        valid_raw: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        test_raw: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[ForecastDataset, ForecastDataset, ForecastDataset, dict[str, StandardScaler]]:
        train_sequence, train_future, train_static, train_residual, _, _ = train_raw
        seq_scaler = StandardScaler()
        future_scaler = StandardScaler()
        static_scaler = StandardScaler()
        target_scaler = StandardScaler()

        n, steps, features = train_sequence.shape
        seq_scaler.fit(train_sequence.reshape(n * steps, features))
        future_scaler.fit(train_future)
        static_scaler.fit(train_static)
        target_scaler.fit(train_residual)

        def transform(
            raw: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            sequence, future, static, residual, actual, baseline = raw
            count, step_count, feature_count = sequence.shape
            sequence = seq_scaler.transform(sequence.reshape(count * step_count, feature_count)).reshape(
                count, step_count, feature_count
            )
            return (
                sequence,
                future_scaler.transform(future),
                static_scaler.transform(static),
                target_scaler.transform(residual),
                actual,
                baseline,
            )

        scalers = {
            "sequence": seq_scaler,
            "future": future_scaler,
            "static": static_scaler,
            "target": target_scaler,
        }

        def dataset(
            raw: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ) -> ForecastDataset:
            sequence, future, static, residual, actual, baseline = transform(raw)
            return ForecastDataset(
                sequence=sequence,
                future=future,
                static=static,
                residual_target=residual,
                baseline=baseline,
                actual=actual,
            )

        return dataset(train_raw), dataset(valid_raw), dataset(test_raw), scalers

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True, batch_size_key="train_batch_size")

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.valid_dataset, shuffle=False, batch_size_key="val_batch_size")

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False, batch_size_key="test_batch_size")

    def _loader(self, dataset: Dataset, *, shuffle: bool, batch_size_key: str) -> DataLoader:
        cfg = self.config["dataloader"]
        num_workers = int(cfg["num_workers"])
        return DataLoader(
            dataset,
            batch_size=int(cfg[batch_size_key]),
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=bool(cfg["persistent_workers"]) and num_workers > 0,
            pin_memory=bool(cfg["pin_memory"]),
        )

    def target_scaler_state(self) -> tuple[np.ndarray, np.ndarray]:
        scaler = self.scalers["target"]
        return scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32)

    def write_feature_file(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "sequence_columns": self.feature_spec.sequence_columns,
                    "future_columns": self.feature_spec.future_columns,
                    "static_columns": self.feature_spec.static_columns,
                    "target_columns": self.target_columns,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

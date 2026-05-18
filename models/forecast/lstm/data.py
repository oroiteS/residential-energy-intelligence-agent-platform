"""无 baseline LSTM 直接预测任务的数据模块。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
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
        target: np.ndarray,
        actual: np.ndarray,
    ) -> None:
        self.sequence = torch.from_numpy(sequence.astype(np.float32))
        self.future = torch.from_numpy(future.astype(np.float32))
        self.static = torch.from_numpy(static.astype(np.float32))
        self.target = torch.from_numpy(target.astype(np.float32))
        self.actual = torch.from_numpy(actual.astype(np.float32))

    def __len__(self) -> int:
        return len(self.actual)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "sequence": self.sequence[index],
            "future": self.future[index],
            "static": self.static[index],
            "target": self.target[index],
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


def build_raw_arrays(
    frame: pd.DataFrame,
    spec: FeatureSpec,
    target_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sequence = np.stack(
        [frame[day_columns].to_numpy(dtype=float) for day_columns in spec.sequence_columns],
        axis=1,
    )
    future = frame[spec.future_columns].to_numpy(dtype=float)
    static = frame[spec.static_columns].to_numpy(dtype=float)
    actual = frame[target_columns].to_numpy(dtype=float)
    return sequence, future, static, actual


class LSTMForecastDataModule(L.LightningDataModule):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.target_columns = list(config["data"]["target_columns"])
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
        train_raw = build_raw_arrays(split.train, self.feature_spec, self.target_columns)
        valid_raw = build_raw_arrays(split.valid, self.feature_spec, self.target_columns)
        test_raw = build_raw_arrays(split.test, self.feature_spec, self.target_columns)

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
        train_raw: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        valid_raw: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        test_raw: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[ForecastDataset, ForecastDataset, ForecastDataset, dict[str, StandardScaler]]:
        train_sequence, train_future, train_static, train_actual = train_raw
        seq_scaler = StandardScaler()
        future_scaler = StandardScaler()
        static_scaler = StandardScaler()
        target_scaler = StandardScaler()

        n, steps, features = train_sequence.shape
        seq_scaler.fit(train_sequence.reshape(n * steps, features))
        future_scaler.fit(train_future)
        static_scaler.fit(train_static)
        target_scaler.fit(train_actual)

        def transform(
            raw: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            sequence, future, static, actual = raw
            count, step_count, feature_count = sequence.shape
            sequence = seq_scaler.transform(sequence.reshape(count * step_count, feature_count)).reshape(
                count, step_count, feature_count
            )
            return (
                sequence,
                future_scaler.transform(future),
                static_scaler.transform(static),
                target_scaler.transform(actual),
                actual,
            )

        scalers = {
            "sequence": seq_scaler,
            "future": future_scaler,
            "static": static_scaler,
            "target": target_scaler,
        }

        def dataset(raw: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> ForecastDataset:
            sequence, future, static, target, actual = transform(raw)
            return ForecastDataset(
                sequence=sequence,
                future=future,
                static=static,
                target=target,
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

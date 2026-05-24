"""Transformer 直接预测任务的数据集与 LightningDataModule。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

try:
    from .config import resolve_path
    from .data import build_feature_spec, build_raw_arrays, time_split, validate_columns
except ImportError:  # 兼容 `python forecast/transformer/dataloader.py`
    import importlib

    _config = importlib.import_module("config")
    _data = importlib.import_module("data")
    resolve_path = getattr(_config, "resolve_path")
    build_feature_spec = getattr(_data, "build_feature_spec")
    build_raw_arrays = getattr(_data, "build_raw_arrays")
    time_split = getattr(_data, "time_split")
    validate_columns = getattr(_data, "validate_columns")


class ForecastDataset(Dataset[dict[str, torch.Tensor]]):
    """将直接预测样本数组封装为 PyTorch 可迭代数据集。"""

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


class TransformerForecastDataModule(L.LightningDataModule):
    """Transformer 直接预测任务的数据组织入口。"""

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
            raw: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            sequence, future, static, actual = raw
            count, step_count, feature_count = sequence.shape
            sequence = cast(np.ndarray, seq_scaler.transform(sequence.reshape(count * step_count, feature_count))).reshape(
                count, step_count, feature_count
            )
            return (
                sequence,
                cast(np.ndarray, future_scaler.transform(future)),
                cast(np.ndarray, static_scaler.transform(static)),
                cast(np.ndarray, target_scaler.transform(actual)),
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
        mean = cast(np.ndarray, scaler.mean_)
        scale = cast(np.ndarray, scaler.scale_)
        return mean.astype(np.float32), scale.astype(np.float32)

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

"""TFT LightningDataModule。"""

from __future__ import annotations

try:
    import lightning as L
except ModuleNotFoundError:  # pragma: no cover - 兼容旧包名
    import pytorch_lightning as L  # type: ignore[no-redef]
from torch.utils.data import DataLoader

from forecast.tft.config import DataConfig
from forecast.tft.dataset import (
    ForecastFeatureSpec,
    ForecastTftDataset,
    load_split_frames,
)


class ForecastTftDataModule(L.LightningDataModule):
    """封装预测任务的数据切分与 DataLoader。"""

    def __init__(self, config: DataConfig, seed: int) -> None:
        super().__init__()
        self.config = config
        self.seed = seed
        self.feature_spec: ForecastFeatureSpec | None = None
        self.train_dataset: ForecastTftDataset | None = None
        self.val_dataset: ForecastTftDataset | None = None
        self.test_dataset: ForecastTftDataset | None = None
        self.split_summary: dict[str, int] = {}

    def prepare_data(self) -> None:
        return

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None and self.val_dataset is not None and self.test_dataset is not None:
            return
        split_frames, feature_spec = load_split_frames(config=self.config, seed=self.seed)
        self.feature_spec = feature_spec
        self.train_dataset = ForecastTftDataset(split_frames["train"], self.config, feature_spec)
        self.val_dataset = ForecastTftDataset(split_frames["val"], self.config, feature_spec)
        self.test_dataset = ForecastTftDataset(split_frames["test"], self.config, feature_spec)
        self.split_summary = {
            "train_samples": len(self.train_dataset),
            "val_samples": len(self.val_dataset),
            "test_samples": len(self.test_dataset),
            "train_houses": split_frames["train"]["house_id"].nunique(),
            "val_houses": split_frames["val"]["house_id"].nunique(),
            "test_houses": split_frames["test"]["house_id"].nunique(),
        }

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("DataModule 尚未 setup")
        return self._build_dataloader(
            dataset=self.train_dataset,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("DataModule 尚未 setup")
        return self._build_dataloader(
            dataset=self.val_dataset,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("DataModule 尚未 setup")
        return self._build_dataloader(
            dataset=self.test_dataset,
            shuffle=False,
            drop_last=False,
        )

    def _build_dataloader(
        self,
        dataset: ForecastTftDataset,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader:
        dataloader_kwargs = {
            "batch_size": self.config.batch_size,
            "shuffle": shuffle,
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory,
            "persistent_workers": self.config.num_workers > 0,
            "drop_last": drop_last,
        }
        if self.config.num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = self.config.prefetch_factor
        return DataLoader(dataset, **dataloader_kwargs)

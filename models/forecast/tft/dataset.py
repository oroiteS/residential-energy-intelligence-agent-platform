"""TFT 预测任务数据集。"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from forecast.tft.config import DataConfig


DECODER_FEATURE_NAMES = (
    "slot_sin",
    "slot_cos",
    "weekday_sin",
    "weekday_cos",
    "baseline",
)
BASELINE_STAT_DIM = 9


@dataclass(slots=True)
class ForecastFeatureSpec:
    input_length: int
    target_length: int
    all_feature_names: tuple[str, ...]
    feature_dim: int
    aggregate_feature_index: int
    profile_feature_indices: tuple[int, ...]
    baseline_stat_dim: int


def _resolve_array_paths(
    metadata_path: Path,
    features_path: Path | None,
    targets_path: Path | None,
) -> tuple[Path, Path]:
    if features_path is not None and targets_path is not None:
        return features_path, targets_path
    stem = metadata_path.stem
    resolved_features = features_path or metadata_path.with_name(f"{stem}_features.npy")
    resolved_targets = targets_path or metadata_path.with_name(f"{stem}_targets.npy")
    return resolved_features, resolved_targets


def load_feature_spec(
    feature_spec_path: Path,
    aggregate_feature_name: str,
) -> ForecastFeatureSpec:
    payload = json.loads(feature_spec_path.read_text(encoding="utf-8"))
    feature_names = tuple(str(name) for name in payload.get("all_feature_names", []))
    profile_feature_names = tuple(str(name) for name in payload.get("profile_feature_names", []))
    if not profile_feature_names:
        profile_feature_names = tuple(
            name for name in feature_names if name.startswith("profile_prob_")
        )
    if not feature_names:
        raise ValueError(f"预测特征说明缺少 all_feature_names: {feature_spec_path}")
    if aggregate_feature_name not in feature_names:
        raise ValueError(
            f"预测特征说明中不存在 aggregate 特征 `{aggregate_feature_name}`: {feature_spec_path}"
        )
    if not profile_feature_names:
        raise ValueError(f"预测特征说明缺少 profile_feature_names: {feature_spec_path}")
    missing_profile_names = [
        feature_name for feature_name in profile_feature_names if feature_name not in feature_names
    ]
    if missing_profile_names:
        raise ValueError(
            "预测特征说明中的 profile_feature_names 不存在于 all_feature_names: "
            f"{missing_profile_names}"
        )
    return ForecastFeatureSpec(
        input_length=int(payload["input_length"]),
        target_length=int(payload["target_length"]),
        all_feature_names=feature_names,
        feature_dim=int(payload["feature_dim"]),
        aggregate_feature_index=feature_names.index(aggregate_feature_name),
        profile_feature_indices=tuple(feature_names.index(name) for name in profile_feature_names),
        baseline_stat_dim=BASELINE_STAT_DIM,
    )


def split_sample_index(
    metadata_df: pd.DataFrame,
    split_mode: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    rng = random.Random(seed)
    if metadata_df.empty:
        raise ValueError("预测元数据为空，无法切分数据集")

    if split_mode == "by_house":
        unique_houses = sorted(metadata_df["house_id"].astype(str).unique().tolist())
        if len(unique_houses) < 3:
            raise ValueError("按家庭切分至少需要 3 个不同家庭")
        shuffled_houses = unique_houses.copy()
        rng.shuffle(shuffled_houses)
        train_houses, val_houses, test_houses = _split_values(
            shuffled_houses,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        return {
            "train": metadata_df.loc[metadata_df["house_id"].isin(set(train_houses))].copy(),
            "val": metadata_df.loc[metadata_df["house_id"].isin(set(val_houses))].copy(),
            "test": metadata_df.loc[metadata_df["house_id"].isin(set(test_houses))].copy(),
        }

    shuffled_positions = list(range(len(metadata_df)))
    rng.shuffle(shuffled_positions)
    train_pos, val_pos, test_pos = _split_values(
        shuffled_positions,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    return {
        "train": metadata_df.iloc[train_pos].copy(),
        "val": metadata_df.iloc[val_pos].copy(),
        "test": metadata_df.iloc[test_pos].copy(),
    }


def _split_values(
    values: list[object],
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[object], list[object], list[object]]:
    total = len(values)
    if total < 3:
        raise ValueError("样本数过少，无法切分训练/验证/测试集")
    train_end = max(1, int(total * train_ratio))
    val_end = max(train_end + 1, int(total * (train_ratio + val_ratio)))
    val_end = min(val_end, total - 1)

    train_values = values[:train_end]
    val_values = values[train_end:val_end]
    test_values = values[val_end:]
    if not val_values:
        val_values = [train_values.pop()]
    if not test_values:
        test_values = [val_values.pop()]
    return train_values, val_values, test_values


def load_split_frames(
    config: DataConfig,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], ForecastFeatureSpec]:
    metadata_df = pd.read_csv(config.metadata_path)
    if "row_index" not in metadata_df.columns:
        raise ValueError(f"预测元数据缺少 row_index 字段: {config.metadata_path}")
    metadata_df["house_id"] = metadata_df["house_id"].astype(str)
    metadata_df["sample_id"] = metadata_df["sample_id"].astype(str)
    metadata_df["target_start"] = pd.to_datetime(metadata_df["target_start"])
    split_frames = split_sample_index(
        metadata_df=metadata_df,
        split_mode=config.split_mode,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=seed,
    )
    feature_spec = load_feature_spec(
        feature_spec_path=config.feature_spec_path,
        aggregate_feature_name=config.aggregate_feature_name,
    )
    return split_frames, feature_spec


class ForecastTftDataset(Dataset):
    """按样本提供 TFT 训练所需张量。"""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        config: DataConfig,
        feature_spec: ForecastFeatureSpec,
    ) -> None:
        super().__init__()
        self.metadata_df = metadata_df.reset_index(drop=True).copy()
        self.config = config
        self.feature_spec = feature_spec
        self.features_path, self.targets_path = _resolve_array_paths(
            metadata_path=config.metadata_path,
            features_path=config.features_path,
            targets_path=config.targets_path,
        )
        self.feature_store = np.load(self.features_path, mmap_mode="r")
        self.target_store = np.load(self.targets_path, mmap_mode="r")
        self.row_indices = self.metadata_df["row_index"].to_numpy(dtype=np.int64)
        self.sample_ids = self.metadata_df["sample_id"].astype(str).tolist()
        self.house_ids = self.metadata_df["house_id"].astype(str).tolist()
        self.target_starts = pd.to_datetime(self.metadata_df["target_start"]).dt.normalize().tolist()

        if self.feature_store.shape[1:] != (
            feature_spec.input_length,
            feature_spec.feature_dim,
        ):
            raise ValueError(
                "预测特征数组形状与 feature_spec 不一致: "
                f"{self.feature_store.shape} vs "
                f"({feature_spec.input_length}, {feature_spec.feature_dim})"
            )
        if self.target_store.shape[1] != feature_spec.target_length:
            raise ValueError(
                "预测目标数组长度与 feature_spec 不一致: "
                f"{self.target_store.shape[1]} vs {feature_spec.target_length}"
            )

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, index: int) -> dict[str, object]:
        row_index = int(self.row_indices[index])
        encoder_cont = np.asarray(self.feature_store[row_index], dtype=np.float32).copy()
        target = np.asarray(self.target_store[row_index], dtype=np.float32).copy()

        aggregate = encoder_cont[:, self.feature_spec.aggregate_feature_index].copy()
        baseline = self._build_baseline(aggregate)
        scale = max(float(np.quantile(aggregate, self.config.normalization_quantile)), 1.0)
        encoder_cont[:, self.feature_spec.aggregate_feature_index] /= scale
        aggregate_norm = encoder_cont[:, self.feature_spec.aggregate_feature_index].copy()
        baseline_norm = baseline / scale
        target_norm = target / scale
        target_residual_norm = target_norm - baseline_norm
        profile_prior = self._build_profile_prior(encoder_cont)
        baseline_stats = self._build_baseline_stats(aggregate_norm)
        decoder_known = self._build_decoder_known_features(
            target_start=self.target_starts[index],
            baseline=baseline_norm,
        )

        return {
            "encoder_cont": torch.from_numpy(encoder_cont),
            "decoder_known": torch.from_numpy(decoder_known),
            "baseline": torch.from_numpy(baseline_norm.astype(np.float32)),
            "target": torch.from_numpy(target_norm.astype(np.float32)),
            "target_residual": torch.from_numpy(target_residual_norm.astype(np.float32)),
            "profile_prior": torch.from_numpy(profile_prior),
            "baseline_stats": torch.from_numpy(baseline_stats),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "sample_id": self.sample_ids[index],
            "house_id": self.house_ids[index],
            "target_start": self.target_starts[index].date().isoformat(),
        }

    def _build_baseline(self, aggregate: np.ndarray) -> np.ndarray:
        first_day = aggregate[: self.feature_spec.target_length]
        last_day = aggregate[-self.feature_spec.target_length :]
        baseline = (
            self.config.baseline_week_weight * first_day
            + self.config.baseline_recent_weight * last_day
        )
        return baseline.astype(np.float32)

    def _build_decoder_known_features(
        self,
        target_start: pd.Timestamp,
        baseline: np.ndarray,
    ) -> np.ndarray:
        slots = np.arange(self.feature_spec.target_length, dtype=np.float32)
        slot_angle = 2.0 * np.pi * slots / float(self.feature_spec.target_length)
        weekday_index = float(pd.Timestamp(target_start).dayofweek)
        weekday_angle = 2.0 * np.pi * weekday_index / 7.0
        return np.stack(
            [
                np.sin(slot_angle).astype(np.float32),
                np.cos(slot_angle).astype(np.float32),
                np.full(self.feature_spec.target_length, np.sin(weekday_angle), dtype=np.float32),
                np.full(self.feature_spec.target_length, np.cos(weekday_angle), dtype=np.float32),
                baseline.astype(np.float32),
            ],
            axis=-1,
        )

    def _build_profile_prior(self, encoder_cont: np.ndarray) -> np.ndarray:
        profile_history = encoder_cont[:, self.feature_spec.profile_feature_indices]
        first_day_prior = profile_history[: self.feature_spec.target_length].mean(axis=0)
        last_day_prior = profile_history[-self.feature_spec.target_length :].mean(axis=0)
        profile_prior = (
            self.config.baseline_week_weight * first_day_prior
            + self.config.baseline_recent_weight * last_day_prior
        )
        profile_prior = np.clip(profile_prior, a_min=1e-6, a_max=None)
        profile_prior = profile_prior / np.sum(profile_prior)
        return profile_prior.astype(np.float32)

    def _build_baseline_stats(self, aggregate_norm: np.ndarray) -> np.ndarray:
        first_day = aggregate_norm[: self.feature_spec.target_length]
        last_day = aggregate_norm[-self.feature_spec.target_length :]
        week_matrix = aggregate_norm.reshape(-1, self.feature_spec.target_length)
        difference = last_day - first_day
        recent_ramp = np.diff(last_day, prepend=last_day[:1])
        day_totals = week_matrix.sum(axis=1)
        stats = np.asarray(
            [
                np.mean(np.abs(difference)),
                np.sqrt(np.mean(np.square(difference))),
                np.max(np.abs(difference)),
                self._safe_corr(first_day, last_day),
                np.std(last_day),
                np.mean(last_day),
                np.mean(np.abs(recent_ramp)),
                np.max(np.abs(recent_ramp)),
                np.std(day_totals) / max(np.mean(day_totals), 1e-6),
            ],
            dtype=np.float32,
        )
        return stats

    @staticmethod
    def _safe_corr(first_day: np.ndarray, last_day: np.ndarray) -> np.float32:
        first_std = float(np.std(first_day))
        last_std = float(np.std(last_day))
        if first_std < 1e-6 or last_std < 1e-6:
            return np.float32(0.0)
        corr = np.corrcoef(first_day, last_day)[0, 1]
        if not np.isfinite(corr):
            return np.float32(0.0)
        return np.float32(corr)

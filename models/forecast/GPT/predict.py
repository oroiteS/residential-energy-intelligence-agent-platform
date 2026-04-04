"""Patch-based direct Transformer 预测推理入口。"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forecast.GPT.config import DEFAULT_CONFIG_PATH, detect_device, load_experiment_config
from forecast.GPT.constants import ALL_FEATURE_NAMES, INPUT_LENGTH, TARGET_LENGTH
from forecast.GPT.engine import build_model, checkpoint_to_normalization, load_checkpoint, save_json_summary
from forecast.LSTM.dataset import (
    DEFAULT_NORMALIZATION_MODE,
    FEATURE_COLUMN_MAP,
    FEATURE_PAYLOAD_FIELD_MAP,
    LEGACY_NORMALIZATION_MODE,
    ForecastNormalizationStats,
    build_temporal_feature_sequences,
)


class PredictionDataset(Dataset[tuple[torch.Tensor, int]]):
    """用于推理阶段的标准化数据集。"""

    def __init__(
        self,
        features: np.ndarray,
        normalization: ForecastNormalizationStats,
    ) -> None:
        normalized, denorm_mean, denorm_std = _normalize_features(
            features=features,
            normalization=normalization,
        )
        self.features = normalized
        self.denorm_mean = denorm_mean
        self.denorm_std = denorm_std

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.from_numpy(self.features[index]), index


def _normalize_features(
    features: np.ndarray,
    normalization: ForecastNormalizationStats,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_features = features.astype(np.float32)

    if normalization.aggregate_mode == DEFAULT_NORMALIZATION_MODE:
        if normalization.auxiliary_mean is None or normalization.auxiliary_std is None:
            raise ValueError("input_window 模式缺少 auxiliary_mean / auxiliary_std")

        aggregate = raw_features[:, :, 0]
        aggregate_mean = aggregate.mean(axis=1, keepdims=True).astype(np.float32)
        aggregate_std = aggregate.std(axis=1, keepdims=True).astype(np.float32)
        safe_aggregate_std = np.where(
            aggregate_std < normalization.aggregate_eps,
            1.0,
            aggregate_std,
        ).astype(np.float32)
        normalized_aggregate = (
            (aggregate - aggregate_mean) / safe_aggregate_std
        )[:, :, np.newaxis].astype(np.float32)
        raw_auxiliary = raw_features[:, :, 1:]
        if raw_auxiliary.shape[-1] == 0:
            normalized = normalized_aggregate
        else:
            normalized_auxiliary = (
                raw_auxiliary - normalization.auxiliary_mean.astype(np.float32)
            ) / normalization.auxiliary_std.astype(np.float32)
            normalized = np.concatenate(
                [normalized_aggregate, normalized_auxiliary.astype(np.float32)],
                axis=-1,
            )
        return normalized, aggregate_mean, safe_aggregate_std

    if normalization.aggregate_mode == LEGACY_NORMALIZATION_MODE:
        if normalization.feature_mean is None or normalization.feature_std is None:
            raise ValueError("global 模式缺少 feature_mean / feature_std")
        if normalization.target_mean is None or normalization.target_std is None:
            raise ValueError("global 模式缺少 target_mean / target_std")
        normalized = (
            raw_features - normalization.feature_mean.astype(np.float32)
        ) / normalization.feature_std.astype(np.float32)
        denorm_mean = np.full(
            (len(raw_features), 1),
            float(normalization.target_mean),
            dtype=np.float32,
        )
        denorm_std = np.full(
            (len(raw_features), 1),
            float(normalization.target_std),
            dtype=np.float32,
        )
        return normalized.astype(np.float32), denorm_mean, denorm_std

    raise ValueError(f"不支持的 aggregate_mode: {normalization.aggregate_mode}")


def _build_feature_array(
    feature_values: dict[str, list[float] | None],
    feature_names: list[str],
) -> np.ndarray:
    aggregate = feature_values.get("aggregate")
    if aggregate is None or len(aggregate) != INPUT_LENGTH:
        raise ValueError("aggregate 输入序列长度必须为 288")

    feature_arrays: list[np.ndarray] = []
    for feature_name in feature_names:
        values = feature_values.get(feature_name)
        if values is None or len(values) != INPUT_LENGTH:
            raise ValueError(f"{feature_name} 输入序列长度必须为 288")
        feature_arrays.append(np.asarray(values, dtype=np.float32))

    features = np.stack(feature_arrays, axis=-1)
    if features.shape != (INPUT_LENGTH, len(feature_names)):
        raise ValueError(f"输入形状不正确: {features.shape}")
    return features


def _single_feature_record_from_payload(
    payload: dict[str, Any],
    feature_names: list[str],
) -> dict[str, Any]:
    temporal_feature_names = {
        "slot_sin",
        "slot_cos",
        "weekday_sin",
        "weekday_cos",
    }
    feature_values: dict[str, list[float] | None] = {}
    if "series" in payload:
        series = payload["series"]
        if not isinstance(series, list):
            raise ValueError("series 必须是数组")
        for feature_name in feature_names:
            payload_key = FEATURE_PAYLOAD_FIELD_MAP[feature_name]
            if feature_name in temporal_feature_names:
                values = [item.get(payload_key) for item in series]
                if any(value is None for value in values):
                    feature_values[feature_name] = None
                else:
                    feature_values[feature_name] = [float(value) for value in values]
                continue
            feature_values[feature_name] = [float(item[payload_key]) for item in series]
    else:
        for feature_name in feature_names:
            payload_key = FEATURE_PAYLOAD_FIELD_MAP[feature_name]
            raw_value = payload.get(payload_key)
            feature_values[feature_name] = raw_value if raw_value is not None else None

    if feature_values.get("aggregate") is None:
        raise ValueError("单样本 json 至少需要包含 aggregate")

    missing_temporal_features = [
        feature_name
        for feature_name in feature_names
        if feature_name in temporal_feature_names and feature_values.get(feature_name) is None
    ]
    if missing_temporal_features:
        temporal_sequences = build_temporal_feature_sequences(
            str(payload.get("input_start", ""))
        )
        for feature_name in missing_temporal_features:
            feature_values[feature_name] = temporal_sequences[feature_name]

    _build_feature_array(feature_values, feature_names)
    record: dict[str, Any] = {
        "sample_id": payload.get("sample_id", "single_sample"),
        "house_id": payload.get("house_id", ""),
        "input_start": payload.get("input_start", ""),
        "input_end": payload.get("input_end", ""),
    }
    for feature_name, values in feature_values.items():
        if values is None:
            continue
        for index, value in enumerate(values):
            record[FEATURE_COLUMN_MAP[feature_name][index]] = float(value)
    return record


def _load_input_records(input_path: Path, feature_names: list[str]) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        data_frame = pd.read_csv(input_path)
    elif suffix == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            records = [
                _single_feature_record_from_payload(item, feature_names)
                for item in payload
            ]
        elif isinstance(payload, dict):
            records = [_single_feature_record_from_payload(payload, feature_names)]
        else:
            raise ValueError("json 输入必须是对象或对象数组")
        data_frame = pd.DataFrame(records)
    else:
        raise ValueError(f"暂不支持的输入格式: {input_path.suffix}")

    required_columns = {
        "sample_id",
        "house_id",
        "input_start",
        "input_end",
        *{
            column
            for feature_name in feature_names
            for column in FEATURE_COLUMN_MAP[feature_name]
        },
    }
    missing_columns = required_columns.difference(data_frame.columns)
    if missing_columns:
        raise ValueError(f"推理输入缺少必要字段: {sorted(missing_columns)}")
    return data_frame


def _extract_features_and_metadata(
    data_frame: pd.DataFrame,
    feature_names: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    feature_arrays = [
        data_frame[FEATURE_COLUMN_MAP[feature_name]].to_numpy(dtype=np.float32)
        for feature_name in feature_names
    ]
    features = np.stack(feature_arrays, axis=-1)
    metadata = data_frame[
        [column for column in ["sample_id", "house_id", "input_start", "input_end"]]
    ].copy()
    return features, metadata


def _predict_values(
    features: np.ndarray,
    checkpoint_path: Path,
    experiment_config,
    batch_size: int,
) -> np.ndarray:
    device_name = detect_device()
    device = torch.device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, device)
    normalization = checkpoint_to_normalization(checkpoint)

    dataset = PredictionDataset(
        features=features,
        normalization=normalization,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(experiment_config.model).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prediction_batches: list[np.ndarray] = []
    denorm_means: list[np.ndarray] = []
    denorm_stds: list[np.ndarray] = []
    with torch.no_grad():
        for feature_batch, index_batch in data_loader:
            feature_batch = feature_batch.to(device)
            outputs = model(feature_batch)
            prediction_batches.append(outputs.cpu().numpy())
            batch_indices = index_batch.cpu().numpy()
            denorm_means.append(dataset.denorm_mean[batch_indices])
            denorm_stds.append(dataset.denorm_std[batch_indices])

    predictions = np.concatenate(prediction_batches, axis=0)
    denorm_mean = np.concatenate(denorm_means, axis=0)
    denorm_std = np.concatenate(denorm_stds, axis=0)
    return predictions * denorm_std + denorm_mean


def predict_batch_from_path(
    input_path: Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> pd.DataFrame:
    experiment_config = load_experiment_config(config_path=config_path)
    predict_config = experiment_config.predict
    checkpoint_path = predict_config.checkpoint_path or (
        experiment_config.train.output_dir / "best_model.pt"
    )

    data_frame = _load_input_records(
        input_path,
        experiment_config.data.feature_names,
    )
    features, metadata = _extract_features_and_metadata(
        data_frame,
        experiment_config.data.feature_names,
    )
    predictions = _predict_values(
        features=features,
        checkpoint_path=checkpoint_path,
        experiment_config=experiment_config,
        batch_size=predict_config.batch_size,
    )
    prediction_columns = {
        f"y_pred_{index:03d}": predictions[:, index]
        for index in range(TARGET_LENGTH)
    }
    return pd.concat([metadata, pd.DataFrame(prediction_columns)], axis=1)


def save_predictions(
    prediction_frame: pd.DataFrame,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json":
        output_path.write_text(
            prediction_frame.to_json(orient="records", force_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        prediction_frame.to_csv(output_path, index=False)
    return output_path


def main(
    input_path: Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_path: Path | None = None,
) -> Path:
    experiment_config = load_experiment_config(config_path=config_path)
    if experiment_config.data.feature_names[0] not in ALL_FEATURE_NAMES:
        raise ValueError("配置中的 feature_names 非法")

    prediction_frame = predict_batch_from_path(
        input_path=input_path,
        config_path=config_path,
    )
    target_output = output_path or (
        experiment_config.predict.output_dir / "prediction.csv"
    )
    saved_path = save_predictions(prediction_frame, target_output)
    save_json_summary(
        experiment_config.predict.output_dir / "prediction_summary.json",
        {
            "input_path": str(input_path),
            "output_path": str(saved_path),
            "rows": len(prediction_frame),
            "config": experiment_config.to_dict(),
        },
    )
    print(f"推理完成，输出已保存到: {saved_path}")
    return saved_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="执行 decoder-only Transformer 推理")
    parser.add_argument("--input", type=Path, required=True, help="输入 csv 或 json 路径")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="yaml 配置文件路径，默认: %(default)s",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="推理输出路径；不传时使用配置中的默认输出目录",
    )
    args = parser.parse_args()
    main(input_path=args.input, config_path=args.config, output_path=args.output)

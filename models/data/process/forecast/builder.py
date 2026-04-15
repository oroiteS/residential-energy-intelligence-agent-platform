"""预测任务数据集构造。"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data.process.common.base import list_base_files, load_base_file, select_complete_days
from data.process.common.progress import ProgressBar, log_stage
from forecast.LSTM.constants import ALL_FEATURE_NAMES, INPUT_LENGTH, TARGET_LENGTH

INPUT_DAYS = 7
TARGET_DAYS = 1
WINDOW_DAYS = INPUT_DAYS + TARGET_DAYS


def _resolve_array_paths(metadata_path: Path) -> tuple[Path, Path]:
    stem = metadata_path.stem
    return (
        metadata_path.with_name(f"{stem}_features.npy"),
        metadata_path.with_name(f"{stem}_targets.npy"),
    )


def _iter_valid_windows(
    day_groups: list[tuple[pd.Timestamp, pd.DataFrame]],
) -> list[tuple[list[tuple[pd.Timestamp, pd.DataFrame]], tuple[pd.Timestamp, pd.DataFrame]]]:
    valid_windows: list[
        tuple[list[tuple[pd.Timestamp, pd.DataFrame]], tuple[pd.Timestamp, pd.DataFrame]]
    ] = []
    for start_index in range(len(day_groups) - WINDOW_DAYS + 1):
        window = day_groups[start_index : start_index + WINDOW_DAYS]
        dates = [day for day, _ in window]
        if any(
            dates[offset + 1] - dates[offset] != timedelta(days=1)
            for offset in range(WINDOW_DAYS - 1)
        ):
            continue
        valid_windows.append((window[:INPUT_DAYS], window[INPUT_DAYS]))
    return valid_windows


def _build_forecast_sample(
    house_id: str,
    input_days: list[tuple[pd.Timestamp, pd.DataFrame]],
    target_day: tuple[pd.Timestamp, pd.DataFrame],
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    input_dates = [day.isoformat() for day, _ in input_days]
    target_date = target_day[0].isoformat()

    metadata: dict[str, object] = {
        "sample_id": f"{house_id}_{input_dates[0]}_{target_date}",
        "house_id": house_id,
        "input_start": input_dates[0],
        "input_end": input_dates[-1],
        "target_start": target_date,
        "target_end": target_date,
        "input_imputed_points": 0,
        "input_imputed_ratio": 0.0,
        "input_clipped_points": 0,
        "target_imputed_points": 0,
        "target_imputed_ratio": 0.0,
        "target_clipped_points": 0,
    }

    feature_columns: dict[str, list[float]] = {
        "aggregate": [],
        "active_appliance_count": [],
        "burst_event_count": [],
        "slot_sin": [],
        "slot_cos": [],
        "weekday_sin": [],
        "weekday_cos": [],
    }
    input_imputed_points = 0
    input_clipped_points = 0
    for _, day_df in input_days:
        sorted_df = day_df.sort_values("slot_index")
        slot_index = sorted_df["slot_index"].to_numpy(dtype=np.float32)
        weekday_index = sorted_df["timestamp"].dt.dayofweek.to_numpy(dtype=np.float32)
        slot_angle = 2.0 * np.pi * slot_index / 96.0
        weekday_angle = 2.0 * np.pi * weekday_index / 7.0
        feature_columns["aggregate"].extend(
            sorted_df["aggregate"].astype(float).tolist()
        )
        feature_columns["active_appliance_count"].extend(
            sorted_df["active_appliance_count"].astype(float).tolist()
        )
        feature_columns["burst_event_count"].extend(
            sorted_df["burst_event_count"].astype(float).tolist()
        )
        feature_columns["slot_sin"].extend(np.sin(slot_angle).astype(float).tolist())
        feature_columns["slot_cos"].extend(np.cos(slot_angle).astype(float).tolist())
        feature_columns["weekday_sin"].extend(
            np.sin(weekday_angle).astype(float).tolist()
        )
        feature_columns["weekday_cos"].extend(
            np.cos(weekday_angle).astype(float).tolist()
        )
        input_imputed_points += int(sorted_df["is_imputed_point"].sum())
        input_clipped_points += int(sorted_df["is_clipped_point"].sum())

    target_sorted_df = target_day[1].sort_values("slot_index")
    target_values = target_sorted_df["aggregate"].astype(float).tolist()
    target_imputed_points = int(target_sorted_df["is_imputed_point"].sum())
    target_clipped_points = int(target_sorted_df["is_clipped_point"].sum())

    aggregate_values = feature_columns["aggregate"]
    metadata["input_imputed_points"] = input_imputed_points
    metadata["input_imputed_ratio"] = float(input_imputed_points) / float(len(aggregate_values))
    metadata["input_clipped_points"] = input_clipped_points
    metadata["target_imputed_points"] = target_imputed_points
    metadata["target_imputed_ratio"] = float(target_imputed_points) / float(len(target_values))
    metadata["target_clipped_points"] = target_clipped_points

    feature_array = np.stack(
        [
            np.asarray(feature_columns[feature_name], dtype=np.float32)
            for feature_name in ALL_FEATURE_NAMES
        ],
        axis=-1,
    )
    target_array = np.asarray(target_values, dtype=np.float32)
    if feature_array.shape != (INPUT_LENGTH, len(ALL_FEATURE_NAMES)):
        raise ValueError(f"预测特征形状不正确: {feature_array.shape}")
    if target_array.shape != (TARGET_LENGTH,):
        raise ValueError(f"预测目标形状不正确: {target_array.shape}")
    return metadata, feature_array, target_array


def _write_rows(
    rows: list[dict[str, object]],
    output_path: Path,
    write_header: bool,
) -> bool:
    if not rows:
        return write_header

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
        mode="w" if write_header else "a",
        header=write_header,
    )
    return False


def build_forecast_dataset(base_dir: Path, output_dir: Path) -> pd.DataFrame:
    base_files = list_base_files(base_dir)
    if not base_files:
        raise FileNotFoundError(f"未在 {base_dir} 找到基础15分钟数据文件")

    log_stage("按家庭构造预测样本")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "forecast_samples.csv"
    feature_path, target_path = _resolve_array_paths(metadata_path)

    total_samples = 0
    for base_file in base_files:
        base_df = load_base_file(base_file)
        complete_days_df = select_complete_days(base_df)
        if complete_days_df.empty:
            continue
        day_groups = [
            (date_value, day_df.copy())
            for date_value, day_df in complete_days_df.groupby("date", sort=True)
        ]
        total_samples += len(_iter_valid_windows(day_groups))

    if total_samples == 0:
        raise ValueError("基础数据中没有可用于预测任务的完整连续窗口样本")

    feature_store = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_samples, INPUT_LENGTH, len(ALL_FEATURE_NAMES)),
    )
    target_store = np.lib.format.open_memmap(
        target_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_samples, TARGET_LENGTH),
    )

    write_header = True
    row_index = 0
    house_progress = ProgressBar("构造预测样本", total=len(base_files), unit="家庭")
    for base_file in base_files:
        base_df = load_base_file(base_file)
        complete_days_df = select_complete_days(base_df)
        if complete_days_df.empty:
            house_progress.update(detail=f"{base_file.stem}（无完整日）")
            continue

        house_id = str(complete_days_df["house_id"].iloc[0])
        day_groups = [
            (date_value, day_df.copy())
            for date_value, day_df in complete_days_df.groupby("date", sort=True)
        ]
        generated_count = 0
        house_metadata_rows: list[dict[str, object]] = []
        for input_days, target_day in _iter_valid_windows(day_groups):
            sample_metadata, feature_array, target_array = _build_forecast_sample(
                house_id=house_id,
                input_days=input_days,
                target_day=target_day,
            )
            sample_metadata["row_index"] = row_index
            house_metadata_rows.append(sample_metadata)
            feature_store[row_index] = feature_array
            target_store[row_index] = target_array
            row_index += 1
            generated_count += 1

        write_header = _write_rows(house_metadata_rows, metadata_path, write_header)
        house_progress.update(detail=f"{house_id}，样本数 {generated_count}")
    house_progress.finish()

    if row_index != total_samples:
        raise RuntimeError(
            f"预测样本写入数量异常，预估 {total_samples}，实际 {row_index}"
        )

    feature_store.flush()
    target_store.flush()
    forecast_df = pd.read_csv(
        metadata_path,
        usecols=["sample_id", "house_id", "input_start", "target_start"],
    )
    return forecast_df.sort_values(["house_id", "input_start"]).reset_index(drop=True)

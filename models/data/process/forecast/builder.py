"""预测任务数据集构造。"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data.process.common.base import list_base_files, load_base_file, select_complete_days
from data.process.common.progress import ProgressBar, log_stage


def _build_forecast_record(
    house_id: str,
    input_days: list[tuple[pd.Timestamp, pd.DataFrame]],
    target_day: tuple[pd.Timestamp, pd.DataFrame],
) -> dict[str, object]:
    input_dates = [day.isoformat() for day, _ in input_days]
    target_date = target_day[0].isoformat()

    record: dict[str, object] = {
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

    aggregate_values: list[float] = []
    active_values: list[int] = []
    burst_values: list[int] = []
    slot_sin_values: list[float] = []
    slot_cos_values: list[float] = []
    weekday_sin_values: list[float] = []
    weekday_cos_values: list[float] = []
    input_imputed_points = 0
    input_clipped_points = 0
    for _, day_df in input_days:
        sorted_df = day_df.sort_values("slot_index")
        slot_index = sorted_df["slot_index"].to_numpy(dtype=np.float32)
        weekday_index = sorted_df["timestamp"].dt.dayofweek.to_numpy(dtype=np.float32)
        slot_angle = 2.0 * np.pi * slot_index / 96.0
        weekday_angle = 2.0 * np.pi * weekday_index / 7.0
        aggregate_values.extend(sorted_df["aggregate"].astype(float).tolist())
        active_values.extend(sorted_df["active_appliance_count"].astype(int).tolist())
        burst_values.extend(sorted_df["burst_event_count"].astype(int).tolist())
        slot_sin_values.extend(np.sin(slot_angle).astype(float).tolist())
        slot_cos_values.extend(np.cos(slot_angle).astype(float).tolist())
        weekday_sin_values.extend(np.sin(weekday_angle).astype(float).tolist())
        weekday_cos_values.extend(np.cos(weekday_angle).astype(float).tolist())
        input_imputed_points += int(sorted_df["is_imputed_point"].sum())
        input_clipped_points += int(sorted_df["is_clipped_point"].sum())

    target_sorted_df = target_day[1].sort_values("slot_index")
    target_values = target_sorted_df["aggregate"].astype(float).tolist()
    target_imputed_points = int(target_sorted_df["is_imputed_point"].sum())
    target_clipped_points = int(target_sorted_df["is_clipped_point"].sum())

    record["input_imputed_points"] = input_imputed_points
    record["input_imputed_ratio"] = float(input_imputed_points) / float(len(aggregate_values))
    record["input_clipped_points"] = input_clipped_points
    record["target_imputed_points"] = target_imputed_points
    record["target_imputed_ratio"] = float(target_imputed_points) / float(len(target_values))
    record["target_clipped_points"] = target_clipped_points

    for index, value in enumerate(aggregate_values):
        record[f"x_aggregate_{index:03d}"] = float(value)
    for index, value in enumerate(active_values):
        record[f"x_active_count_{index:03d}"] = int(value)
    for index, value in enumerate(burst_values):
        record[f"x_burst_count_{index:03d}"] = int(value)
    for index, value in enumerate(slot_sin_values):
        record[f"x_slot_sin_{index:03d}"] = float(value)
    for index, value in enumerate(slot_cos_values):
        record[f"x_slot_cos_{index:03d}"] = float(value)
    for index, value in enumerate(weekday_sin_values):
        record[f"x_weekday_sin_{index:03d}"] = float(value)
    for index, value in enumerate(weekday_cos_values):
        record[f"x_weekday_cos_{index:03d}"] = float(value)
    for index, value in enumerate(target_values):
        record[f"y_aggregate_{index:03d}"] = float(value)
    return record


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
    forecast_path = output_dir / "forecast_samples.csv"
    summary_records: list[dict[str, object]] = []
    write_header = True
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
        house_sample_records: list[dict[str, object]] = []
        for start_index in range(len(day_groups) - 3):
            window = day_groups[start_index : start_index + 4]
            dates = [day for day, _ in window]
            if any(dates[offset + 1] - dates[offset] != timedelta(days=1) for offset in range(3)):
                continue

            input_days = window[:3]
            target_day = window[3]
            sample_record = _build_forecast_record(
                house_id=house_id,
                input_days=input_days,
                target_day=target_day,
            )
            house_sample_records.append(sample_record)
            summary_records.append(
                {
                    "sample_id": str(sample_record["sample_id"]),
                    "house_id": house_id,
                    "input_start": str(sample_record["input_start"]),
                    "target_start": str(sample_record["target_start"]),
                }
            )
            generated_count += 1

        write_header = _write_rows(house_sample_records, forecast_path, write_header)
        house_progress.update(detail=f"{house_id}，样本数 {generated_count}")
    house_progress.finish()

    if not summary_records:
        raise ValueError("基础数据中没有可用于预测任务的完整连续窗口样本")

    forecast_df = pd.DataFrame(summary_records).sort_values(["house_id", "input_start"]).reset_index(drop=True)
    return forecast_df

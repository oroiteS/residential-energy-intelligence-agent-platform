"""分类任务日级特征构造。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.process.common.base import list_base_files, load_base_file, select_complete_days
from data.process.common.progress import ProgressBar, log_stage


def _build_daily_feature_record(
    house_id: str,
    day: pd.Timestamp,
    day_df: pd.DataFrame,
) -> dict[str, object]:
    sorted_df = day_df.sort_values("slot_index")
    aggregate_values = sorted_df["aggregate"].to_numpy(dtype=float)
    slot_index = sorted_df["slot_index"].to_numpy(dtype=np.float32)
    weekday_index = sorted_df["timestamp"].dt.dayofweek.to_numpy(dtype=np.float32)
    slot_angle = 2.0 * np.pi * slot_index / 96.0
    weekday_angle = 2.0 * np.pi * weekday_index / 7.0
    slot_sin_values = np.sin(slot_angle).astype(float)
    slot_cos_values = np.cos(slot_angle).astype(float)
    weekday_sin_values = np.sin(weekday_angle).astype(float)
    weekday_cos_values = np.cos(weekday_angle).astype(float)

    record: dict[str, object] = {
        "sample_id": f"{house_id}_{day.isoformat()}",
        "house_id": house_id,
        "date": day.isoformat(),
    }

    for index, value in enumerate(aggregate_values):
        record[f"aggregate_{index:03d}"] = float(value)
    for index, value in enumerate(slot_sin_values):
        record[f"slot_sin_{index:03d}"] = float(value)
    for index, value in enumerate(slot_cos_values):
        record[f"slot_cos_{index:03d}"] = float(value)
    for index, value in enumerate(weekday_sin_values):
        record[f"weekday_sin_{index:03d}"] = float(value)
    for index, value in enumerate(weekday_cos_values):
        record[f"weekday_cos_{index:03d}"] = float(value)
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


def build_classification_features(base_dir: Path, output_dir: Path) -> pd.DataFrame:
    """从基础 15 分钟时序生成日级分类特征文件，不再在此阶段生成规则标签。"""

    base_files = list_base_files(base_dir)
    if not base_files:
        raise FileNotFoundError(f"未在 {base_dir} 找到基础15分钟数据文件")

    log_stage("写出分类日级特征文件")
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "classification_day_features.csv"

    write_progress = ProgressBar("写出分类样本", total=len(base_files), unit="家庭")
    feature_write_header = True
    feature_index_records: list[dict[str, object]] = []

    for base_file in base_files:
        base_df = load_base_file(base_file)
        complete_days_df = select_complete_days(base_df)
        if complete_days_df.empty:
            write_progress.update(detail=f"{base_file.stem}（无完整日）")
            continue

        house_id = str(complete_days_df["house_id"].iloc[0])
        feature_records: list[dict[str, object]] = []
        for date_value, day_df in complete_days_df.groupby("date", sort=True):
            feature_record = _build_daily_feature_record(
                house_id=house_id,
                day=date_value,
                day_df=day_df,
            )
            feature_records.append(feature_record)
            feature_index_records.append(
                {
                    "sample_id": feature_record["sample_id"],
                    "house_id": house_id,
                    "date": date_value.isoformat(),
                }
            )

        feature_write_header = _write_rows(feature_records, features_path, feature_write_header)
        write_progress.update(detail=house_id)

    write_progress.finish()
    if not feature_index_records:
        raise ValueError("基础数据中没有可用于分类任务的完整日样本")

    return (
        pd.DataFrame(feature_index_records)
        .sort_values(["house_id", "date"])
        .reset_index(drop=True)
    )


__all__ = ["build_classification_features"]

"""分类任务数据集构造。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.process.common.base import list_base_files, load_base_file, select_complete_days
from data.process.common.constants import DAY_END_SLOT, DAY_START_SLOT
from data.process.common.progress import ProgressBar, log_stage


def _build_daily_feature_record(house_id: str, day: pd.Timestamp, day_df: pd.DataFrame) -> dict[str, object]:
    sorted_df = day_df.sort_values("slot_index")
    aggregate_values = sorted_df["aggregate"].to_numpy(dtype=float)
    active_values = sorted_df["active_appliance_count"].to_numpy(dtype=int)
    burst_values = sorted_df["burst_event_count"].to_numpy(dtype=int)

    record: dict[str, object] = {
        "sample_id": f"{house_id}_{day.isoformat()}",
        "house_id": house_id,
        "date": day.isoformat(),
    }

    for index, value in enumerate(aggregate_values):
        record[f"aggregate_{index:03d}"] = float(value)
    for index, value in enumerate(active_values):
        record[f"active_count_{index:03d}"] = int(value)
    for index, value in enumerate(burst_values):
        record[f"burst_count_{index:03d}"] = int(value)
    return record


def _compute_label_stats(day_df: pd.DataFrame) -> dict[str, float]:
    sorted_df = day_df.sort_values("slot_index")
    aggregate_values = sorted_df["aggregate"].to_numpy(dtype=float)
    day_values = aggregate_values[DAY_START_SLOT:DAY_END_SLOT]
    night_values = pd.Series(aggregate_values).drop(range(DAY_START_SLOT, DAY_END_SLOT)).to_numpy(dtype=float)
    return {
        "day_mean": float(day_values.mean()),
        "night_mean": float(night_values.mean()),
        "full_mean": float(aggregate_values.mean()),
    }


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


def _assign_rule_label(
    day_mean: float,
    night_mean: float,
    full_mean: float,
    high_threshold: float,
    low_threshold: float,
    ratio_threshold: float,
) -> str:
    epsilon = 1e-6
    if day_mean >= high_threshold and night_mean >= high_threshold:
        return "all_day_high"
    if day_mean <= low_threshold and night_mean <= low_threshold:
        return "all_day_low"
    if day_mean / max(night_mean, epsilon) >= ratio_threshold:
        return "day_high_night_low"
    if night_mean / max(day_mean, epsilon) >= ratio_threshold:
        return "day_low_night_high"

    scores = {
        "all_day_high": min(day_mean, night_mean) / max(high_threshold, epsilon),
        "all_day_low": min(low_threshold / max(day_mean, epsilon), low_threshold / max(night_mean, epsilon)),
        "day_high_night_low": (day_mean / max(night_mean, epsilon)) / ratio_threshold,
        "day_low_night_high": (night_mean / max(day_mean, epsilon)) / ratio_threshold,
    }
    return max(scores, key=scores.get)


def build_classification_dataset(base_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_files = list_base_files(base_dir)
    if not base_files:
        raise FileNotFoundError(f"未在 {base_dir} 找到基础15分钟数据文件")

    log_stage("按家庭统计分类日样本")
    label_stats_records: list[dict[str, object]] = []
    stats_progress = ProgressBar("统计分类样本", total=len(base_files), unit="家庭")
    for base_file in base_files:
        base_df = load_base_file(base_file)
        complete_days_df = select_complete_days(base_df)
        if complete_days_df.empty:
            stats_progress.update(detail=f"{base_file.stem}（无完整日）")
            continue

        house_id = str(complete_days_df["house_id"].iloc[0])
        for date_value, day_df in complete_days_df.groupby("date", sort=True):
            label_stats_records.append(
                {
                    "sample_id": f"{house_id}_{date_value.isoformat()}",
                    "house_id": house_id,
                    "date": date_value.isoformat(),
                    **_compute_label_stats(day_df),
                }
            )
        stats_progress.update(detail=house_id)
    stats_progress.finish()

    if not label_stats_records:
        raise ValueError("基础数据中没有可用于分类任务的完整日样本")

    stats_df = pd.DataFrame(label_stats_records).sort_values(["house_id", "date"]).reset_index(drop=True)

    high_threshold = float(stats_df["full_mean"].quantile(0.75))
    low_threshold = float(stats_df["full_mean"].quantile(0.25))
    ratio_threshold = 1.2

    stats_df["high_threshold"] = high_threshold
    stats_df["low_threshold"] = low_threshold
    stats_df["ratio_threshold"] = ratio_threshold
    stats_df["label_name"] = stats_df.apply(
        lambda row: _assign_rule_label(
            day_mean=float(row["day_mean"]),
            night_mean=float(row["night_mean"]),
            full_mean=float(row["full_mean"]),
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            ratio_threshold=ratio_threshold,
        ),
        axis=1,
    )

    log_stage("写出分类数据文件")
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "classification_day_features.csv"
    labels_path = output_dir / "classification_day_labels.csv"
    house_stats_map = {
        house_id: house_stats.set_index("sample_id")
        for house_id, house_stats in stats_df.groupby("house_id", sort=True)
    }

    write_progress = ProgressBar("写出分类样本", total=len(base_files), unit="家庭")
    feature_write_header = True
    label_write_header = True
    for base_file in base_files:
        base_df = load_base_file(base_file)
        complete_days_df = select_complete_days(base_df)
        if complete_days_df.empty:
            write_progress.update(detail=f"{base_file.stem}（无完整日）")
            continue

        house_id = str(complete_days_df["house_id"].iloc[0])
        house_stats = house_stats_map.get(house_id)
        if house_stats is None or house_stats.empty:
            write_progress.update(detail=f"{house_id}（无标签）")
            continue

        feature_records: list[dict[str, object]] = []
        label_records: list[dict[str, object]] = []
        for date_value, day_df in complete_days_df.groupby("date", sort=True):
            sample_id = f"{house_id}_{date_value.isoformat()}"
            feature_record = _build_daily_feature_record(
                house_id=house_id,
                day=date_value,
                day_df=day_df,
            )
            feature_records.append(feature_record)
            stats_row = house_stats.loc[sample_id].to_dict()
            label_records.append(feature_record | stats_row)

        feature_write_header = _write_rows(feature_records, features_path, feature_write_header)
        label_write_header = _write_rows(label_records, labels_path, label_write_header)
        write_progress.update(detail=house_id)
    write_progress.finish()

    features_df = stats_df.loc[:, ["sample_id", "house_id", "date"]].copy()
    labels_df = stats_df.loc[:, ["sample_id", "house_id", "date", "label_name"]].copy()
    return features_df, labels_df

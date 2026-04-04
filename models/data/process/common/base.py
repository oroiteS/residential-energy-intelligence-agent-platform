"""基础15分钟时序预处理。"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from data.process.common.constants import (
    AGGREGATE_CLIP_UPPER_QUANTILE,
    ACTIVE_POWER_THRESHOLD_W,
    BASE_FREQ,
    BASE_OUTPUT_GLOB,
    BURST_DELTA_THRESHOLD_W,
    BURST_MAX_DURATION_SECONDS,
    MAX_DAILY_IMPUTED_RATIO,
    MAX_INTERPOLATION_GAP_SLOTS,
    OPENSYNTH_SUBDIR_NAME,
    REFIT_SUBDIR_NAME,
    RAW_FILE_GLOB,
    SLOVAKIA_SUBDIR_NAME,
    SLOTS_PER_DAY,
    UKDALE_SUBDIR_NAME,
)
from data.process.common.progress import ProgressBar, log_stage

REQUIRED_RAW_COLUMNS = {"Time", "Aggregate", "Issues"}
HOUSE_ID_PATTERN = re.compile(r"House(\d+)", re.IGNORECASE)
METER_ID_PATTERN = re.compile(r"meters_(\d+)_measurement", re.IGNORECASE)
CATEGORY_MINUTES_PATTERN = re.compile(r"(\d+)\s*m", re.IGNORECASE)


def list_raw_files(input_dir: Path) -> list[Path]:
    raw_files = sorted(input_dir.glob(RAW_FILE_GLOB))
    if raw_files:
        return raw_files

    refit_dir = input_dir / REFIT_SUBDIR_NAME
    if refit_dir.is_dir():
        return sorted(refit_dir.glob(RAW_FILE_GLOB))

    return []


def list_base_files(base_dir: Path) -> list[Path]:
    return sorted(base_dir.glob(BASE_OUTPUT_GLOB))


def extract_house_id(file_path: Path) -> str:
    match = HOUSE_ID_PATTERN.search(file_path.stem)
    if not match:
        raise ValueError(f"无法从文件名提取 house_id: {file_path.name}")
    return match.group(1)


def extract_meter_id(file_path: Path) -> str:
    match = METER_ID_PATTERN.search(file_path.stem)
    if not match:
        raise ValueError(f"无法从文件名提取 meter_id: {file_path.name}")
    return match.group(1)


def sanitize_identifier(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_")
    if not normalized:
        raise ValueError(f"无法生成合法标识符: {value!r}")
    return normalized.lower()


def resolve_source_dir(input_dir: Path, subdir_name: str) -> Path | None:
    if input_dir.name == subdir_name and input_dir.is_dir():
        return input_dir

    candidate = input_dir / subdir_name
    if candidate.is_dir():
        return candidate

    return None


def build_standard_raw_df(
    timestamps: pd.Series | pd.DatetimeIndex,
    aggregate_values_w: pd.Series | np.ndarray,
) -> pd.DataFrame:
    timestamp_series = normalize_timestamp_series(pd.Series(timestamps))
    aggregate_series = pd.to_numeric(pd.Series(aggregate_values_w), errors="coerce")
    raw_df = pd.DataFrame(
        {
            "Time": timestamp_series.astype("string"),
            "Aggregate": aggregate_series.astype(float),
            "Issues": 0,
            "timestamp": timestamp_series,
        }
    )
    return raw_df


def normalize_timestamp_series(
    timestamps: pd.Series | pd.DatetimeIndex,
) -> pd.Series:
    timestamp_series = pd.to_datetime(pd.Series(timestamps), errors="coerce")
    if getattr(timestamp_series.dt, "tz", None) is not None:
        timestamp_series = timestamp_series.dt.tz_convert("UTC").dt.tz_localize(None)
    return timestamp_series


def category_to_minutes(category: str) -> int:
    match = CATEGORY_MINUTES_PATTERN.search(category)
    if not match:
        raise ValueError(f"无法解析 OpenSynth 分辨率: {category}")
    return int(match.group(1))


def expand_interval_energy_to_15min_power(
    timestamps: pd.Series,
    energy_kwh: pd.Series,
    interval_minutes: int,
) -> tuple[pd.Series, np.ndarray]:
    if interval_minutes % 15 != 0:
        raise ValueError(f"暂不支持无法整除 15 分钟的分辨率: {interval_minutes}")

    factor = interval_minutes // 15
    timestamps = pd.to_datetime(timestamps, errors="coerce")
    energy_values = pd.to_numeric(energy_kwh, errors="coerce").astype(float)
    valid_mask = timestamps.notna() & energy_values.notna()
    timestamps = timestamps.loc[valid_mask].reset_index(drop=True)
    energy_values = energy_values.loc[valid_mask].reset_index(drop=True)
    if timestamps.empty:
        return pd.Series(dtype="datetime64[ns]"), np.array([], dtype=float)

    average_power_w = energy_values.to_numpy(dtype=float) * (60000.0 / float(interval_minutes))
    offsets = np.tile(np.arange(0, interval_minutes, 15, dtype=int), len(timestamps))
    expanded_timestamps = timestamps.repeat(factor).reset_index(drop=True) + pd.to_timedelta(
        offsets,
        unit="m",
    )
    expanded_values_w = np.repeat(average_power_w, factor)
    return expanded_timestamps, expanded_values_w


def load_refit_sources(input_dir: Path) -> Iterator[tuple[str, str, str, pd.DataFrame]]:
    refit_dir = resolve_source_dir(input_dir, REFIT_SUBDIR_NAME) or input_dir
    raw_files = sorted(refit_dir.glob(RAW_FILE_GLOB))
    for raw_file in raw_files:
        house_id = f"refit_{extract_house_id(raw_file)}"
        raw_df = load_raw_house_data(raw_file)
        yield ("refit", house_id, "W", raw_df)


def load_slovakia_sources(input_dir: Path) -> Iterator[tuple[str, str, str, pd.DataFrame]]:
    slovakia_dir = resolve_source_dir(input_dir, SLOVAKIA_SUBDIR_NAME)
    if slovakia_dir is None:
        return

    for json_path in sorted(slovakia_dir.glob("meters_*_measurement.json")):
        meter_id = extract_meter_id(json_path)
        with json_path.open("r", encoding="utf-8") as file:
            daily_records = json.load(file)

        timestamps: list[pd.Timestamp] = []
        aggregate_values_kw: list[float] = []
        for daily_record in daily_records:
            day_start = pd.Timestamp(
                year=int(daily_record["year"]),
                month=int(daily_record["month"]),
                day=int(daily_record["day"]),
            )
            consumption_values = daily_record.get("consumption", [])
            for slot_index, value in enumerate(consumption_values):
                timestamps.append(day_start + pd.Timedelta(minutes=slot_index * 15))
                aggregate_values_kw.append(float(value))

        if not timestamps:
            continue

        raw_df = build_standard_raw_df(
            timestamps=pd.Series(timestamps),
            aggregate_values_w=np.asarray(aggregate_values_kw, dtype=float) * 1000.0,
        )
        yield ("slovakia_households_1000", f"slovakia_{meter_id}", "kW", raw_df)


def load_opensynth_sources(input_dir: Path) -> Iterator[tuple[str, str, str, pd.DataFrame]]:
    opensynth_dir = resolve_source_dir(input_dir, OPENSYNTH_SUBDIR_NAME)
    if opensynth_dir is None:
        return

    parquet_files = sorted(opensynth_dir.rglob("*.parquet"))
    if not parquet_files:
        return

    for parquet_path in parquet_files:
        try:
            parquet_df = pd.read_parquet(
                parquet_path,
                columns=["id", "datetime", "target", "category"],
            )
        except ImportError as exc:
            raise RuntimeError(
                "读取 OpenSynth parquet 需要安装 pyarrow 或 fastparquet"
            ) from exc

        if parquet_df.empty:
            continue

        parquet_df["id"] = parquet_df["id"].astype(str)
        parquet_df["category"] = parquet_df["category"].astype(str)
        parquet_df["datetime"] = pd.to_datetime(parquet_df["datetime"], errors="coerce")
        parquet_df["target"] = pd.to_numeric(parquet_df["target"], errors="coerce")
        parquet_df = parquet_df.dropna(subset=["id", "category", "datetime", "target"])

        group_items = list(parquet_df.groupby(["id", "category"], sort=True))
        group_progress = ProgressBar(
            label=f"OpenSynth 子序列 {parquet_path.name}",
            total=len(group_items),
            unit="序列",
        )
        for (series_id, category), series_df in group_items:
            interval_minutes = category_to_minutes(category)
            expanded_timestamps, expanded_values_w = expand_interval_energy_to_15min_power(
                timestamps=series_df["datetime"],
                energy_kwh=series_df["target"],
                interval_minutes=interval_minutes,
            )
            if len(expanded_values_w) == 0:
                continue

            house_id = f"opensynth_{sanitize_identifier(series_id)}_{sanitize_identifier(category)}"
            raw_df = build_standard_raw_df(
                timestamps=expanded_timestamps,
                aggregate_values_w=expanded_values_w,
            )
            yield (
                "opensynth_tudelft_electricity_consumption_1_0",
                house_id,
                "kWh_per_interval",
                raw_df,
            )
            group_progress.update(detail=house_id)
        group_progress.finish(detail=parquet_path.name)


def _find_ukdale_power_column(frame: pd.DataFrame) -> tuple[object, str] | None:
    if frame.empty:
        return None

    if isinstance(frame.columns, pd.MultiIndex):
        normalized_columns = {
            column: tuple(str(level).lower() for level in column)
            for column in frame.columns
        }
        for preferred in [("power", "active"), ("power", "apparent")]:
            for column, normalized in normalized_columns.items():
                if normalized[: len(preferred)] == preferred:
                    return column, preferred[-1]
        for column, normalized in normalized_columns.items():
            if normalized and normalized[0] == "power":
                measurement = normalized[1] if len(normalized) > 1 else "unknown"
                return column, measurement
        return None

    lowered_columns = {column: str(column).lower() for column in frame.columns}
    for keyword, measurement in [("active", "active"), ("apparent", "apparent"), ("power", "unknown")]:
        for column, lowered in lowered_columns.items():
            if keyword in lowered:
                return column, measurement
    return None


def load_ukdale_sources(input_dir: Path) -> Iterator[tuple[str, str, str, pd.DataFrame]]:
    ukdale_dir = resolve_source_dir(input_dir, UKDALE_SUBDIR_NAME)
    if ukdale_dir is None:
        return

    h5_path = ukdale_dir / "ukdale.h5"
    if not h5_path.exists():
        return

    try:
        store = pd.HDFStore(h5_path, mode="r")
    except ImportError as exc:
        raise RuntimeError(
            "读取 UKDALE HDF5 需要安装 PyTables（tables）依赖"
        ) from exc

    with store:
        meter_keys = sorted(
            key
            for key in store.keys()
            if re.fullmatch(r"/building\d+/elec/meter1", key)
        )

        for meter_key in meter_keys:
            meter_frame = store[meter_key]
            column_info = _find_ukdale_power_column(meter_frame)
            if column_info is None:
                continue

            power_column, measurement = column_info
            aggregate_series = pd.to_numeric(meter_frame[power_column], errors="coerce")
            timestamps = pd.Series(pd.to_datetime(meter_frame.index, errors="coerce"))
            raw_df = build_standard_raw_df(
                timestamps=timestamps,
                aggregate_values_w=aggregate_series.to_numpy(dtype=float),
            )
            building_id = meter_key.split("/")[1].removeprefix("building")
            yield (f"ukdale_{measurement}", f"ukdale_{building_id}", "W", raw_df)


def collect_raw_sources(input_dir: Path) -> Iterator[tuple[str, str, str, pd.DataFrame]]:
    yield from load_refit_sources(input_dir)
    yield from load_ukdale_sources(input_dir)
    yield from load_slovakia_sources(input_dir)
    yield from load_opensynth_sources(input_dir)


def detect_appliance_columns(columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column.startswith("Appliance")]


def load_raw_house_data(file_path: Path) -> pd.DataFrame:
    raw_df = pd.read_csv(file_path)
    missing_columns = REQUIRED_RAW_COLUMNS.difference(raw_df.columns)
    if missing_columns:
        raise ValueError(f"{file_path.name} 缺少必要字段: {sorted(missing_columns)}")

    appliance_columns = detect_appliance_columns(raw_df.columns)
    numeric_columns = ["Aggregate", *appliance_columns]
    for column in numeric_columns:
        raw_df[column] = pd.to_numeric(raw_df[column], errors="coerce")

    raw_df["timestamp"] = normalize_timestamp_series(raw_df["Time"])
    raw_df["Issues"] = pd.to_numeric(raw_df["Issues"], errors="coerce").fillna(1).astype(int)
    return raw_df


def infer_sampling_seconds(timestamp_series: pd.Series) -> float:
    deltas = timestamp_series.sort_values().diff().dropna().dt.total_seconds()
    positive_deltas = deltas[deltas > 0]
    if positive_deltas.empty:
        return 0.0
    return float(positive_deltas.median())


def _detect_burst_counts(raw_df: pd.DataFrame, appliance_columns: list[str]) -> pd.Series:
    if not appliance_columns:
        return pd.Series(dtype="int64", name="burst_event_count")

    timestamps = raw_df["timestamp"].reset_index(drop=True)
    burst_counter: defaultdict[pd.Timestamp, int] = defaultdict(int)

    for column in appliance_columns:
        values = raw_df[column].fillna(0.0).to_numpy(dtype=float)
        if len(values) < 2:
            continue

        deltas = np.diff(values, prepend=values[0])
        candidate_indices = np.flatnonzero(deltas > BURST_DELTA_THRESHOLD_W)
        if len(candidate_indices) == 0:
            continue

        candidate_position = 0
        total_length = len(values)
        while candidate_position < len(candidate_indices):
            start_index = int(candidate_indices[candidate_position])
            if start_index == 0:
                candidate_position += 1
                continue

            baseline = values[start_index - 1]
            end_index = start_index
            returned_to_baseline = False

            while end_index + 1 < total_length:
                next_index = end_index + 1
                elapsed_seconds = (timestamps.iat[next_index] - timestamps.iat[start_index]).total_seconds()
                if elapsed_seconds > BURST_MAX_DURATION_SECONDS:
                    break

                end_index = next_index
                if values[end_index] <= baseline:
                    returned_to_baseline = True
                    break

            if returned_to_baseline:
                burst_slot = timestamps.iat[start_index].floor(BASE_FREQ)
                burst_counter[burst_slot] += 1

            while (
                candidate_position + 1 < len(candidate_indices)
                and int(candidate_indices[candidate_position + 1]) <= end_index
            ):
                candidate_position += 1
            candidate_position += 1

    if not burst_counter:
        return pd.Series(dtype="int64", name="burst_event_count")

    burst_series = pd.Series(burst_counter, dtype="int64")
    burst_series.index.name = "timestamp"
    burst_series.name = "burst_event_count"
    return burst_series.sort_index()


def _find_long_gap_mask(
    missing_mask: pd.Series,
    max_gap_slots: int,
) -> pd.Series:
    values = missing_mask.to_numpy(dtype=bool)
    long_gap_values = np.zeros(len(values), dtype=bool)
    gap_start: int | None = None

    for index, is_missing in enumerate(values):
        if is_missing and gap_start is None:
            gap_start = index
        if not is_missing and gap_start is not None:
            if index - gap_start > max_gap_slots:
                long_gap_values[gap_start:index] = True
            gap_start = None

    if gap_start is not None and len(values) - gap_start > max_gap_slots:
        long_gap_values[gap_start:] = True

    return pd.Series(long_gap_values, index=missing_mask.index)


def _clip_aggregate_outliers(
    aggregate_series: pd.Series,
) -> tuple[pd.Series, float, pd.Series]:
    observed_values = aggregate_series.dropna()
    if observed_values.empty:
        return aggregate_series, 0.0, pd.Series(False, index=aggregate_series.index)

    clip_threshold = float(
        observed_values.quantile(AGGREGATE_CLIP_UPPER_QUANTILE)
    )
    if clip_threshold <= 0.0:
        return aggregate_series, 0.0, pd.Series(False, index=aggregate_series.index)

    clipped_mask = aggregate_series > clip_threshold
    return aggregate_series.clip(upper=clip_threshold), clip_threshold, clipped_mask


def build_base_timeseries(
    raw_df: pd.DataFrame,
    house_id: str,
    source_dataset: str,
    raw_aggregate_unit: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    appliance_columns = detect_appliance_columns(raw_df.columns)
    summary: dict[str, object] = {
        "house_id": house_id,
        "source_dataset": source_dataset,
        "raw_rows": int(len(raw_df)),
        "raw_aggregate_unit": raw_aggregate_unit,
        "aggregate_unit": "W",
        "appliance_column_count": len(appliance_columns),
        "interpolation_strategy": "time",
    }

    summary["issue_rows_removed"] = int((raw_df["Issues"] == 1).sum())
    summary["invalid_time_rows_removed"] = int(raw_df["timestamp"].isna().sum())

    cleaned_df = raw_df.loc[(raw_df["Issues"] != 1) & raw_df["timestamp"].notna()].copy()
    cleaned_df["timestamp"] = normalize_timestamp_series(cleaned_df["timestamp"])
    cleaned_df = cleaned_df.sort_values("timestamp").reset_index(drop=True)

    summary["inferred_sampling_seconds"] = infer_sampling_seconds(cleaned_df["timestamp"])
    summary["duplicate_timestamps_removed"] = int(cleaned_df.duplicated("timestamp").sum())
    cleaned_df = cleaned_df.drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    numeric_columns = ["Aggregate", *appliance_columns]
    cleaned_df[numeric_columns] = cleaned_df[numeric_columns].fillna(0.0).clip(lower=0.0)

    if cleaned_df.empty:
        raise ValueError(f"house {house_id} 在清洗后没有可用记录")

    burst_series = _detect_burst_counts(cleaned_df, appliance_columns)

    indexed_df = cleaned_df.set_index("timestamp")[numeric_columns]
    resampled_df = indexed_df.resample(BASE_FREQ).mean()

    full_index = pd.date_range(
        start=resampled_df.index.min(),
        end=resampled_df.index.max(),
        freq=BASE_FREQ,
    )
    observed_slot_count = int(cleaned_df["timestamp"].dt.floor(BASE_FREQ).nunique())
    summary["actual_15min_points_before_fill"] = observed_slot_count
    summary["expected_15min_points"] = int(len(full_index))
    summary["missing_15min_points"] = int(len(full_index) - observed_slot_count)
    summary["missing_rate"] = (
        float(summary["missing_15min_points"]) / float(summary["expected_15min_points"])
        if summary["expected_15min_points"]
        else 0.0
    )

    resampled_df = resampled_df.reindex(full_index)
    missing_mask = resampled_df["Aggregate"].isna()
    long_gap_mask = _find_long_gap_mask(
        missing_mask=missing_mask,
        max_gap_slots=MAX_INTERPOLATION_GAP_SLOTS,
    )
    summary["long_gap_15min_points"] = int(long_gap_mask.sum())

    resampled_df = resampled_df.interpolate(
        method="time",
        limit=MAX_INTERPOLATION_GAP_SLOTS,
        limit_area="inside",
    )
    if long_gap_mask.any():
        resampled_df.loc[long_gap_mask, numeric_columns] = np.nan

    unresolved_mask = resampled_df["Aggregate"].isna()
    imputed_mask = missing_mask & ~unresolved_mask
    summary["imputed_15min_points"] = int(imputed_mask.sum())
    summary["unresolved_15min_points"] = int(unresolved_mask.sum())
    summary["imputed_rate"] = (
        float(summary["imputed_15min_points"]) / float(summary["expected_15min_points"])
        if summary["expected_15min_points"]
        else 0.0
    )
    summary["unresolved_rate"] = (
        float(summary["unresolved_15min_points"]) / float(summary["expected_15min_points"])
        if summary["expected_15min_points"]
        else 0.0
    )

    (
        clipped_aggregate,
        aggregate_clip_threshold_w,
        clipped_mask,
    ) = _clip_aggregate_outliers(resampled_df["Aggregate"])
    resampled_df["Aggregate"] = clipped_aggregate
    summary["aggregate_clip_upper_quantile"] = AGGREGATE_CLIP_UPPER_QUANTILE
    summary["aggregate_clip_threshold_w"] = aggregate_clip_threshold_w
    summary["aggregate_clipped_15min_points"] = int(clipped_mask.sum())

    if appliance_columns:
        active_count_series = (
            resampled_df[appliance_columns] > ACTIVE_POWER_THRESHOLD_W
        ).sum(axis=1).astype(float)
        appliance_nan_mask = resampled_df[appliance_columns].isna().any(axis=1)
        active_count_series.loc[appliance_nan_mask] = np.nan
    else:
        active_count_series = pd.Series(0.0, index=resampled_df.index, dtype=float)

    burst_series = burst_series.reindex(full_index, fill_value=0).astype(float)
    burst_series.loc[unresolved_mask] = np.nan

    base_df = pd.DataFrame(
        {
            "house_id": house_id,
            "source_dataset": source_dataset,
            "timestamp": full_index,
            "date": full_index.date.astype(str),
            "slot_index": (full_index.hour * 4 + full_index.minute // 15).astype(int),
            "aggregate": resampled_df["Aggregate"].astype(float),
            "active_appliance_count": active_count_series.to_numpy(),
            "burst_event_count": burst_series.to_numpy(),
            "is_weekend": (full_index.dayofweek >= 5).astype(int),
            "is_observed_point": (~missing_mask).astype(int).to_numpy(),
            "is_imputed_point": imputed_mask.astype(int).to_numpy(),
            "is_unresolved_point": unresolved_mask.astype(int).to_numpy(),
            "is_clipped_point": clipped_mask.fillna(False).astype(int).to_numpy(),
        }
    )

    base_df["active_appliance_count"] = base_df["active_appliance_count"].round()
    base_df["burst_event_count"] = base_df["burst_event_count"].round()

    summary["cleaned_rows"] = int(len(cleaned_df))
    summary["base_rows"] = int(len(base_df))
    summary["start_timestamp"] = base_df["timestamp"].min().isoformat()
    summary["end_timestamp"] = base_df["timestamp"].max().isoformat()
    return base_df, summary


def build_base_dataset(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    processed_source_count = 0

    refit_dir = resolve_source_dir(input_dir, REFIT_SUBDIR_NAME) or input_dir
    refit_files = sorted(refit_dir.glob(RAW_FILE_GLOB))
    if refit_files:
        log_stage("处理 REFIT 原始数据")
        refit_progress = ProgressBar("REFIT 基础预处理", total=len(refit_files), unit="家庭")
        for raw_file in refit_files:
            house_id = f"refit_{extract_house_id(raw_file)}"
            raw_df = load_raw_house_data(raw_file)
            base_df, summary = build_base_timeseries(
                raw_df=raw_df,
                house_id=house_id,
                source_dataset="refit",
                raw_aggregate_unit="W",
            )
            output_file = output_dir / f"house_{house_id}_base_15min.csv"
            base_df.to_csv(output_file, index=False)
            summaries.append(summary)
            processed_source_count += 1
            refit_progress.update(detail=house_id)
        refit_progress.finish()

    ukdale_dir = resolve_source_dir(input_dir, UKDALE_SUBDIR_NAME)
    if ukdale_dir is not None and (ukdale_dir / "ukdale.h5").exists():
        log_stage("处理 UKDALE 原始数据")
        try:
            store = pd.HDFStore(ukdale_dir / "ukdale.h5", mode="r")
        except ImportError as exc:
            raise RuntimeError(
                "读取 UKDALE HDF5 需要安装 PyTables（tables）依赖"
            ) from exc
        with store:
            meter_keys = sorted(
                key
                for key in store.keys()
                if re.fullmatch(r"/building\d+/elec/meter1", key)
            )
        ukdale_progress = ProgressBar("UKDALE 基础预处理", total=len(meter_keys), unit="家庭")
        for source_dataset, house_id, raw_aggregate_unit, raw_df in load_ukdale_sources(input_dir):
            base_df, summary = build_base_timeseries(
                raw_df=raw_df,
                house_id=house_id,
                source_dataset=source_dataset,
                raw_aggregate_unit=raw_aggregate_unit,
            )
            output_file = output_dir / f"house_{house_id}_base_15min.csv"
            base_df.to_csv(output_file, index=False)
            summaries.append(summary)
            processed_source_count += 1
            ukdale_progress.update(detail=house_id)
        ukdale_progress.finish()

    slovakia_dir = resolve_source_dir(input_dir, SLOVAKIA_SUBDIR_NAME)
    if slovakia_dir is not None:
        slovakia_files = sorted(slovakia_dir.glob("meters_*_measurement.json"))
        if slovakia_files:
            log_stage("处理斯洛伐克 1000 户数据")
            slovakia_progress = ProgressBar(
                "Slovakia 基础预处理",
                total=len(slovakia_files),
                unit="家庭",
            )
            for source_dataset, house_id, raw_aggregate_unit, raw_df in load_slovakia_sources(input_dir):
                base_df, summary = build_base_timeseries(
                    raw_df=raw_df,
                    house_id=house_id,
                    source_dataset=source_dataset,
                    raw_aggregate_unit=raw_aggregate_unit,
                )
                output_file = output_dir / f"house_{house_id}_base_15min.csv"
                base_df.to_csv(output_file, index=False)
                summaries.append(summary)
                processed_source_count += 1
                slovakia_progress.update(detail=house_id)
            slovakia_progress.finish()

    opensynth_dir = resolve_source_dir(input_dir, OPENSYNTH_SUBDIR_NAME)
    if opensynth_dir is not None:
        parquet_files = sorted(opensynth_dir.rglob("*.parquet"))
        if parquet_files:
            log_stage("处理 OpenSynth 多源整合数据")
            opensynth_progress = ProgressBar(
                "OpenSynth 基础预处理",
                total=len(parquet_files),
                unit="文件",
            )
            processed_before = processed_source_count
            for parquet_path in parquet_files:
                parquet_input_dir = parquet_path.parent
                # 仅处理当前 parquet 所在目录，避免重复扫描整个数据源目录
                try:
                    parquet_df = pd.read_parquet(
                        parquet_path,
                        columns=["id", "datetime", "target", "category"],
                    )
                except ImportError as exc:
                    raise RuntimeError(
                        "读取 OpenSynth parquet 需要安装 pyarrow 或 fastparquet"
                    ) from exc
                if parquet_df.empty:
                    opensynth_progress.update(detail=parquet_path.name)
                    continue
                parquet_df["id"] = parquet_df["id"].astype(str)
                parquet_df["category"] = parquet_df["category"].astype(str)
                parquet_df["datetime"] = pd.to_datetime(parquet_df["datetime"], errors="coerce")
                parquet_df["target"] = pd.to_numeric(parquet_df["target"], errors="coerce")
                parquet_df = parquet_df.dropna(subset=["id", "category", "datetime", "target"])
                group_items = list(parquet_df.groupby(["id", "category"], sort=True))
                group_progress = ProgressBar(
                    label=f"OpenSynth 子序列 {parquet_path.name}",
                    total=len(group_items),
                    unit="序列",
                )
                for (series_id, category), series_df in group_items:
                    interval_minutes = category_to_minutes(category)
                    expanded_timestamps, expanded_values_w = expand_interval_energy_to_15min_power(
                        timestamps=series_df["datetime"],
                        energy_kwh=series_df["target"],
                        interval_minutes=interval_minutes,
                    )
                    if len(expanded_values_w) == 0:
                        group_progress.update(detail=f"{series_id}_{category}")
                        continue
                    house_id = f"opensynth_{sanitize_identifier(series_id)}_{sanitize_identifier(category)}"
                    raw_df = build_standard_raw_df(
                        timestamps=expanded_timestamps,
                        aggregate_values_w=expanded_values_w,
                    )
                    base_df, summary = build_base_timeseries(
                        raw_df=raw_df,
                        house_id=house_id,
                        source_dataset="opensynth_tudelft_electricity_consumption_1_0",
                        raw_aggregate_unit="kWh_per_interval",
                    )
                    output_file = output_dir / f"house_{house_id}_base_15min.csv"
                    base_df.to_csv(output_file, index=False)
                    summaries.append(summary)
                    processed_source_count += 1
                    group_progress.update(detail=house_id)
                group_progress.finish(detail=parquet_path.name)
                opensynth_progress.update(detail=parquet_path.name)
            if processed_source_count > processed_before or parquet_files:
                opensynth_progress.finish()

    if processed_source_count == 0:
        raise FileNotFoundError(
            f"未在 {input_dir} 找到可用原始数据。"
            "当前支持的数据源目录包括 refit、ukdale、"
            "slovakia_households_1000、opensynth_tudelft_electricity_consumption_1_0"
        )

    summary_df = pd.DataFrame(summaries).sort_values("house_id").reset_index(drop=True)
    summary_df.to_csv(output_dir / "quality_summary.csv", index=False)
    return summary_df


def load_base_dataset(base_dir: Path) -> pd.DataFrame:
    base_files = list_base_files(base_dir)
    if not base_files:
        raise FileNotFoundError(f"未在 {base_dir} 找到基础15分钟数据文件")

    progress = ProgressBar("加载基础时序数据", total=len(base_files), unit="文件")
    dataframes: list[pd.DataFrame] = []
    for base_file in base_files:
        dataframes.append(pd.read_csv(base_file, parse_dates=["timestamp"]))
        progress.update(detail=base_file.name)
    progress.finish()
    base_df = pd.concat(dataframes, ignore_index=True)
    base_df["date"] = pd.to_datetime(base_df["date"]).dt.date
    base_df["house_id"] = base_df["house_id"].astype(str)
    return base_df.sort_values(["house_id", "timestamp"]).reset_index(drop=True)


def select_complete_days(base_df: pd.DataFrame) -> pd.DataFrame:
    required_quality_columns = {
        "is_imputed_point",
        "is_unresolved_point",
        "is_clipped_point",
    }
    missing_quality_columns = required_quality_columns.difference(base_df.columns)
    if missing_quality_columns:
        raise ValueError(
            "基础时序缺少质量字段，请先重新执行 preprocess-base："
            f"{sorted(missing_quality_columns)}"
        )

    day_stats = (
        base_df.groupby(["house_id", "date"])
        .agg(
            row_count=("slot_index", "size"),
            slot_count=("slot_index", "nunique"),
            min_slot=("slot_index", "min"),
            max_slot=("slot_index", "max"),
            imputed_points=("is_imputed_point", "sum"),
            unresolved_points=("is_unresolved_point", "sum"),
            clipped_points=("is_clipped_point", "sum"),
        )
        .reset_index()
    )
    day_stats["imputed_ratio"] = (
        day_stats["imputed_points"].astype(float) / float(SLOTS_PER_DAY)
    )
    valid_days = day_stats.loc[
        (day_stats["row_count"] == SLOTS_PER_DAY)
        & (day_stats["slot_count"] == SLOTS_PER_DAY)
        & (day_stats["min_slot"] == 0)
        & (day_stats["max_slot"] == SLOTS_PER_DAY - 1)
        & (day_stats["unresolved_points"] == 0),
        ["house_id", "date", "imputed_ratio", "clipped_points"],
    ]
    valid_days = valid_days.loc[
        (valid_days["imputed_ratio"] <= MAX_DAILY_IMPUTED_RATIO),
    ]
    complete_df = base_df.merge(valid_days, on=["house_id", "date"], how="inner")
    return complete_df.sort_values(["house_id", "date", "slot_index"]).reset_index(drop=True)

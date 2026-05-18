from __future__ import annotations

import json
import math
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import current_app
from sqlalchemy import or_
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from app.errors import NotFoundError, ValidationError
from app.extensions import db
from app.models import Dataset
from app.services.analysis_service import build_analysis_payload, upsert_analysis_result
from app.services.common import slugify, to_iso


STANDARD_TIMESTAMP_COLUMN = "timestamp"
STANDARD_VALUE_COLUMN = "aggregate_w"


def list_datasets(*, page: int, page_size: int, status: str | None, keyword: str | None) -> dict:
    query = Dataset.query
    if status:
        query = query.filter(Dataset.status == status)
    if keyword:
        pattern = f"%{keyword.strip()}%"
        query = query.filter(
            or_(
                Dataset.name.like(pattern),
                Dataset.description.like(pattern),
                Dataset.household_id.like(pattern),
            )
        )

    safe_page = max(page, 1)
    safe_page_size = min(max(page_size, 1), 100)
    pagination = query.order_by(Dataset.created_at.desc()).paginate(page=safe_page, per_page=safe_page_size, error_out=False)
    return {
        "items": [dataset_summary_dto(item) for item in pagination.items],
        "pagination": {
            "page": safe_page,
            "page_size": safe_page_size,
            "total": pagination.total,
        },
    }


def get_dataset_detail(dataset_id: int) -> dict:
    dataset = Dataset.query.get(dataset_id)
    if dataset is None:
        raise NotFoundError("数据集不存在", code="DATASET_NOT_FOUND")
    return {
        "dataset": dataset_detail_dto(dataset),
        "quality_summary": dataset.quality_summary,
    }


def import_dataset(
    *,
    file: FileStorage | None,
    name: str,
    description: str | None,
    household_id: str | None,
    unit: str,
    column_mapping: dict | None,
) -> dict:
    if file is None or not file.filename:
        raise ValidationError("请上传文件", code="FILE_REQUIRED")
    if not name.strip():
        raise ValidationError("数据集名称不能为空", code="NAME_REQUIRED")

    ext = Path(file.filename).suffix.lower()
    if ext != ".csv":
        raise ValidationError("当前仅支持 CSV 文件导入", code="UNSUPPORTED_FILE_TYPE")

    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{secure_filename(file.filename)}"
    raw_path = current_app.config["UPLOAD_DIR"] / filename
    file.save(raw_path)

    dataset = Dataset(
        name=name.strip(),
        description=description.strip() if description else None,
        household_id=household_id.strip() if household_id else None,
        source_file_name=file.filename,
        raw_file_path=str(raw_path),
        status="processing",
        column_mapping=column_mapping or {},
    )
    db.session.add(dataset)
    db.session.commit()

    try:
        frame = pd.read_csv(raw_path)
        timestamp_column, value_column = _detect_columns(frame, column_mapping or {})
        frame = frame[[timestamp_column, value_column]].rename(
            columns={timestamp_column: "timestamp", value_column: "raw_value"}
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame["raw_value"] = pd.to_numeric(frame["raw_value"], errors="coerce")
        frame = frame.dropna(subset=["timestamp", "raw_value"]).sort_values("timestamp").reset_index(drop=True)
        if frame.empty:
            raise ValidationError("文件中没有可用的时间序列数据", code="EMPTY_DATASET")

        duplicate_count = int(frame.duplicated(subset=["timestamp"]).sum())
        frame = frame.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

        diff_seconds = _diff_seconds(frame["timestamp"])
        if not diff_seconds:
            raise ValidationError("无法识别时间粒度，请至少提供两条有效记录", code="INVALID_GRANULARITY")

        if len(set(diff_seconds)) != 1:
            raise ValidationError(
                "上传数据格式错误：采样间隔必须固定一致，请使用前端提供的标准 CSV 模板",
                code="INCONSISTENT_GRANULARITY",
            )

        source_granularity_seconds = diff_seconds[0]
        if source_granularity_seconds % 60 != 0:
            raise ValidationError(
                "上传数据格式错误：采样间隔必须按整分钟组织，且位于 1 到 60 分钟之间",
                code="INVALID_MINUTE_GRANULARITY",
            )

        source_granularity = source_granularity_seconds // 60
        max_granularity = max(diff_seconds) // 60
        allowed_min = current_app.config["ACCEPTED_MIN_GRANULARITY_MINUTES"]
        allowed_max = current_app.config["ACCEPTED_MAX_GRANULARITY_MINUTES"]
        if source_granularity < allowed_min:
            raise ValidationError(
                f"上传数据粒度过细，标准上传格式要求最小粒度为 {allowed_min} 分钟",
                code="GRANULARITY_TOO_FINE",
            )
        if source_granularity > allowed_max:
            raise ValidationError(
                f"上传数据粒度过粗，系统要求至少小时级（<= {allowed_max} 分钟）",
                code="GRANULARITY_TOO_COARSE",
            )

        normalized_df = _normalize_series(frame, unit=unit, granularity_minutes=source_granularity)
        daily_df = _build_daily_aggregate(normalized_df, granularity_minutes=source_granularity)
        forecast_history_days = current_app.config["FORECAST_HISTORY_DAYS"]
        available_days = int(daily_df["date"].nunique())
        if available_days < forecast_history_days:
            raise ValidationError(
                f"上传数据时长不足，当前预测模型要求上传数据覆盖至少 {forecast_history_days} 天",
                code="INSUFFICIENT_HISTORY_DAYS",
            )

        dataset_name_slug = slugify(name)
        normalized_path = current_app.config["NORMALIZED_DIR"] / f"dataset_{dataset.id}_{dataset_name_slug}.csv"
        daily_path = current_app.config["DAILY_DIR"] / f"dataset_{dataset.id}_{dataset_name_slug}.csv"
        quality_path = current_app.config["QUALITY_DIR"] / f"dataset_{dataset.id}_quality.json"
        analysis_path = current_app.config["ANALYSIS_DIR"] / f"dataset_{dataset.id}_analysis.json"

        normalized_df.to_csv(normalized_path, index=False)
        daily_df.to_csv(daily_path, index=False)

        quality_summary = {
            "missing_rate": 0.0,
            "duplicate_count": duplicate_count,
            "sampling_interval": f"{source_granularity}min",
            "cleaning_strategies": ["时间戳排序", "重复时间戳去重", "统一字段标准化", "按日聚合"],
            "min_granularity_minutes": source_granularity,
            "max_granularity_minutes": max_granularity,
            "accepted_min_granularity_minutes": current_app.config["ACCEPTED_MIN_GRANULARITY_MINUTES"],
            "accepted_max_granularity_minutes": allowed_max,
        }
        quality_path.write_text(json.dumps(quality_summary, ensure_ascii=False, indent=2), encoding="utf-8")

        analysis_payload = build_analysis_payload(
            normalized_df=normalized_df,
            daily_df=daily_df,
            peak_periods=current_app.config["PEAK_PERIODS"],
            valley_periods=current_app.config["VALLEY_PERIODS"],
            detail_path=analysis_path,
        )
        analysis = upsert_analysis_result(dataset, analysis_payload, analysis_path)

        dataset.normalized_file_path = str(normalized_path)
        dataset.daily_aggregate_file_path = str(daily_path)
        dataset.row_count = int(len(frame))
        dataset.time_start = normalized_df["timestamp"].min().to_pydatetime()
        dataset.time_end = normalized_df["timestamp"].max().to_pydatetime()
        dataset.source_granularity_minutes = int(source_granularity)
        dataset.quality_summary = quality_summary
        dataset.quality_report_path = str(quality_path)
        dataset.status = "ready"

        db.session.add(analysis)
        db.session.commit()
        return dataset_summary_dto(dataset)
    except ValidationError as exc:
        db.session.rollback()
        dataset.status = "error"
        dataset.error_message = exc.message
        db.session.add(dataset)
        db.session.commit()
        raise
    except Exception as exc:
        db.session.rollback()
        dataset.status = "error"
        dataset.error_message = str(exc)
        db.session.add(dataset)
        db.session.commit()
        raise ValidationError(f"导入失败：{exc}", code="IMPORT_FAILED") from exc


def get_daily_rows(dataset: Dataset) -> list[dict]:
    daily_path = Path(dataset.daily_aggregate_file_path or "")
    if not daily_path.exists():
        return []
    frame = pd.read_csv(daily_path, parse_dates=["date"])
    rows = frame.to_dict("records")
    for row in rows:
        row["date"] = pd.to_datetime(row["date"]).date()
    return rows


def get_normalized_frame(dataset: Dataset) -> pd.DataFrame:
    path = Path(dataset.normalized_file_path or "")
    if not path.exists():
        raise NotFoundError("标准化数据不存在", code="NORMALIZED_DATA_NOT_FOUND")
    return pd.read_csv(path, parse_dates=["timestamp"])


def get_dataset_or_404(dataset_id: int) -> Dataset:
    dataset = Dataset.query.get(dataset_id)
    if dataset is None:
        raise NotFoundError("数据集不存在", code="DATASET_NOT_FOUND")
    return dataset


def dataset_summary_dto(dataset: Dataset) -> dict:
    return {
        "id": dataset.id,
        "name": dataset.name,
        "description": dataset.description,
        "household_id": dataset.household_id,
        "row_count": dataset.row_count,
        "time_start": to_iso(dataset.time_start),
        "time_end": to_iso(dataset.time_end),
        "status": dataset.status,
        "created_at": to_iso(dataset.created_at),
        "updated_at": to_iso(dataset.updated_at),
    }


def dataset_detail_dto(dataset: Dataset) -> dict:
    return {
        **dataset_summary_dto(dataset),
        "raw_file_path": dataset.raw_file_path,
        "processed_file_path": dataset.normalized_file_path,
        "feature_cols": ["timestamp", "aggregate_w"],
        "column_mapping": dataset.column_mapping or {},
        "quality_report_path": dataset.quality_report_path,
        "error_message": dataset.error_message,
    }


def _detect_columns(frame: pd.DataFrame, column_mapping: dict) -> tuple[str, str]:
    if column_mapping:
        timestamp_column = column_mapping.get("timestamp")
        value_column = column_mapping.get("value") or column_mapping.get("aggregate")
        if timestamp_column and value_column:
            return timestamp_column, value_column

    lowercase_map = {column.lower(): column for column in frame.columns}
    timestamp_column = lowercase_map.get(STANDARD_TIMESTAMP_COLUMN)
    value_column = lowercase_map.get(STANDARD_VALUE_COLUMN)
    if timestamp_column and value_column:
        return timestamp_column, value_column

    raise ValidationError(
        f"上传文件格式错误：CSV 表头必须严格为 {STANDARD_TIMESTAMP_COLUMN},{STANDARD_VALUE_COLUMN}",
        code="INVALID_UPLOAD_HEADERS",
    )


def _diff_seconds(series: pd.Series) -> list[int]:
    diffs = (
        series.sort_values()
        .diff()
        .dropna()
        .dt.total_seconds()
    )
    values = [int(round(item)) for item in diffs.tolist() if item > 0]
    return values


def _normalize_series(frame: pd.DataFrame, *, unit: str, granularity_minutes: int) -> pd.DataFrame:
    hours = granularity_minutes / 60
    if unit == "w":
        aggregate_w = frame["raw_value"]
    elif unit == "wh":
        aggregate_w = frame["raw_value"] / max(hours, 1e-6)
    elif unit == "kwh":
        aggregate_w = frame["raw_value"] * 1000 / max(hours, 1e-6)
    else:
        raise ValidationError(f"不支持的单位：{unit}", code="UNSUPPORTED_UNIT")

    normalized_df = pd.DataFrame(
        {
            "timestamp": frame["timestamp"],
            "aggregate_w": aggregate_w.astype(float),
        }
    )
    return normalized_df


def _build_daily_aggregate(normalized_df: pd.DataFrame, *, granularity_minutes: int) -> pd.DataFrame:
    hours = granularity_minutes / 60
    frame = normalized_df.copy()
    frame["date"] = frame["timestamp"].dt.date
    frame["time_text"] = frame["timestamp"].dt.strftime("%H:%M")
    frame["energy_kwh"] = frame["aggregate_w"] * hours / 1000

    def time_bucket(value: str) -> str:
        if _match_period(value, current_app.config["PEAK_PERIODS"]):
            return "peak"
        if _match_period(value, current_app.config["VALLEY_PERIODS"]):
            return "valley"
        return "valley"

    frame["bucket"] = frame["time_text"].map(time_bucket)

    daily = (
        frame.groupby(["date", "bucket"])["energy_kwh"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    for column in ["peak", "valley"]:
        if column not in daily.columns:
            daily[column] = 0.0
    daily["total_kwh"] = daily["peak"] + daily["valley"]
    daily = daily.rename(columns={"peak": "peak_kwh", "valley": "valley_kwh"})
    daily["date"] = pd.to_datetime(daily["date"])
    return daily[["date", "total_kwh", "peak_kwh", "valley_kwh"]].sort_values("date")


def _match_period(time_text: str, periods: Iterable[str]) -> bool:
    current_minutes = _clock_to_minutes(time_text)
    for period in periods:
        start_text, end_text = [part.strip() for part in period.split("-", 1)]
        start_minutes = _clock_to_minutes(start_text)
        end_minutes = _clock_to_minutes(end_text)
        if start_minutes < end_minutes:
            if start_minutes <= current_minutes < end_minutes:
                return True
        else:
            if current_minutes >= start_minutes or current_minutes < end_minutes:
                return True
    return False


def _clock_to_minutes(text: str) -> int:
    hour, minute = text.split(":")
    return int(hour) * 60 + int(minute)

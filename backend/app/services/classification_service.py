from __future__ import annotations

from datetime import date, datetime, timedelta

from flask import current_app

from app.extensions import db
from app.models import ClassificationResult
from app.services.dataset_service import get_daily_rows, get_dataset_or_404
from app.services.common import to_iso
from models.classification import classify_daily_window


def predict_classification(dataset_id: int, *, window_role: str = "current", forecast_id: int | None = None) -> dict:
    dataset = get_dataset_or_404(dataset_id)
    daily_rows = get_daily_rows(dataset)
    if window_role == "future":
        window_days = current_app.config["CLASSIFICATION_DAYS"]
        window = daily_rows[-window_days:] if len(daily_rows) >= window_days else daily_rows
        return classification_dto(
            _create_classification_record(
                dataset_id=dataset.id,
                forecast_id=forecast_id,
                window_role=window_role,
                window=window,
                sample_suffix="future",
            )
        )

    created_records = _refresh_current_classifications(dataset.id, daily_rows)
    latest_record = max(created_records, key=lambda item: item.window_end or datetime.min)
    return classification_dto(latest_record)


def list_classifications(dataset_id: int) -> list[dict]:
    _ensure_current_classifications(dataset_id)
    records = (
        ClassificationResult.query.filter_by(dataset_id=dataset_id, window_role="current")
        .order_by(ClassificationResult.window_start.desc(), ClassificationResult.created_at.desc())
        .all()
    )
    return [classification_dto(item) for item in records]


def get_latest_classification(dataset_id: int) -> dict | None:
    _ensure_current_classifications(dataset_id)
    record = (
        ClassificationResult.query.filter_by(dataset_id=dataset_id, window_role="current")
        .order_by(ClassificationResult.window_end.desc(), ClassificationResult.created_at.desc())
        .first()
    )
    return classification_dto(record) if record else None


def _create_classification_record(
    *,
    dataset_id: int,
    forecast_id: int | None,
    window_role: str,
    window: list[dict],
    sample_suffix: str,
    commit: bool = True,
) -> ClassificationResult:
    result = classify_daily_window(window)
    window_start = datetime.combine(window[0]["date"], datetime.min.time()) if window else None
    window_end = datetime.combine(window[-1]["date"], datetime.min.time()) if window else None
    record = ClassificationResult(
        dataset_id=dataset_id,
        forecast_id=forecast_id,
        model_type="xgboost",
        window_role=window_role,
        predicted_label=result["predicted_label"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        explanation=result["explanation"],
        sample_id=f"{dataset_id}-{window_role}-{sample_suffix}",
        runtime_library="backend-heuristic",
        window_start=window_start,
        window_end=window_end,
    )
    db.session.add(record)
    if commit:
        db.session.commit()
    return record


def _ensure_current_classifications(dataset_id: int) -> None:
    dataset = get_dataset_or_404(dataset_id)
    daily_rows = get_daily_rows(dataset)
    expected_window_count = len(_build_weekly_windows(daily_rows))
    existing_count = (
        ClassificationResult.query.filter_by(
            dataset_id=dataset_id,
            window_role="current",
            forecast_id=None,
        ).count()
    )
    if expected_window_count and existing_count != expected_window_count:
        _refresh_current_classifications(dataset_id, daily_rows)


def _refresh_current_classifications(dataset_id: int, daily_rows: list[dict]) -> list[ClassificationResult]:
    windows = _build_weekly_windows(daily_rows)
    if not windows:
        raise ValueError("分类窗口不能为空")

    ClassificationResult.query.filter_by(
        dataset_id=dataset_id,
        window_role="current",
        forecast_id=None,
    ).delete()

    created_records: list[ClassificationResult] = []
    for index, window in enumerate(windows):
        record = _create_classification_record(
            dataset_id=dataset_id,
            forecast_id=None,
            window_role="current",
            window=window,
            sample_suffix=f"week{index + 1:02d}",
            commit=False,
        )
        created_records.append(record)

    db.session.commit()
    return created_records


def _build_weekly_windows(daily_rows: list[dict]) -> list[list[dict]]:
    if not daily_rows:
        return []

    rows_by_date = {item["date"]: item for item in daily_rows}
    start_date = min(rows_by_date)
    end_date = max(rows_by_date)
    current_week_start = start_date - timedelta(days=start_date.weekday())
    final_week_start = end_date - timedelta(days=end_date.weekday())
    windows: list[list[dict]] = []

    while current_week_start <= final_week_start:
        week_dates = [current_week_start + timedelta(days=index) for index in range(7)]
        available_rows = [rows_by_date[day] for day in week_dates if day in rows_by_date]
        if available_rows:
            week_mean_total = sum(float(item["total_kwh"]) for item in available_rows) / len(available_rows)
            week_mean_peak = sum(float(item["peak_kwh"]) for item in available_rows) / len(available_rows)
            week_mean_valley = sum(float(item["valley_kwh"]) for item in available_rows) / len(available_rows)
            window: list[dict] = []
            for day in week_dates:
                if day in rows_by_date:
                    window.append(rows_by_date[day])
                else:
                    window.append(
                        {
                            "date": day,
                            "total_kwh": week_mean_total,
                            "peak_kwh": week_mean_peak,
                            "valley_kwh": week_mean_valley,
                        }
                    )
            windows.append(window)
        current_week_start += timedelta(days=7)

    return windows


def classification_dto(record: ClassificationResult | None) -> dict | None:
    if record is None:
        return None
    return {
        "id": record.id,
        "dataset_id": record.dataset_id,
        "model_type": record.model_type,
        "schema_version": record.schema_version,
        "predicted_label": record.predicted_label,
        "label_display_name": record.predicted_label,
        "confidence": record.confidence or 0,
        "probabilities": record.probabilities or {},
        "explanation": record.explanation,
        "sample_id": record.sample_id,
        "runtime_library": record.runtime_library,
        "window_start": to_iso(record.window_start),
        "window_end": to_iso(record.window_end),
        "created_at": to_iso(record.created_at),
    }

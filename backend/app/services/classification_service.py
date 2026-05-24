from __future__ import annotations

from datetime import date, datetime, timedelta

from flask import current_app

from app.extensions import db
from app.models import ClassificationResult
from app.services.dataset_service import get_daily_rows, get_dataset_or_404
from app.services.common import to_iso
from models.classification import classify_daily_window


def predict_classification(dataset_id: int, *, window_role: str = "current", forecast_id: int | None = None) -> dict:
    """执行用电类型分类。

    current 表示基于历史日聚合数据生成当前分类；
    future 表示基于预测窗口生成未来分类，并可关联 forecast_id。
    """

    dataset = get_dataset_or_404(dataset_id)
    daily_rows = get_daily_rows(dataset)
    if window_role == "future":
        # 未来窗口使用配置中的分类天数。
        # 默认 7 天，对应“按周观察用户用电类型”的业务口径。
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

    # 当前分类会刷新所有历史周窗口，并返回结束日期最新的一条记录。
    created_records = _refresh_current_classifications(dataset.id, daily_rows)
    latest_record = max(created_records, key=lambda item: item.window_end or datetime.min)
    return classification_dto(latest_record)


def list_classifications(dataset_id: int) -> list[dict]:
    """列出某个数据集的当前历史分类结果。"""

    _ensure_current_classifications(dataset_id)
    records = (
        ClassificationResult.query.filter_by(dataset_id=dataset_id, window_role="current")
        .order_by(ClassificationResult.window_start.desc(), ClassificationResult.created_at.desc())
        .all()
    )
    return [classification_dto(item) for item in records]


def get_latest_classification(dataset_id: int) -> dict | None:
    """获取结束日期最新的当前分类结果。"""

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
    """根据一个日级窗口创建分类结果记录。"""

    # classify_daily_window 返回模型/规则推理结果；
    # 服务层负责补齐 dataset_id、窗口范围和入库字段。
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
    """确保数据库中的当前分类结果与日级窗口数量一致。"""

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
    """重建某个数据集的全部历史周分类结果。"""

    windows = _build_weekly_windows(daily_rows)
    if not windows:
        raise ValueError("分类窗口不能为空")

    # 当前窗口分类是可重算结果。
    # 重建前先删除旧记录，避免上传数据更新后出现重复或过期窗口。
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
    """将日级数据切分为自然周窗口。

    每个窗口固定覆盖周一到周日 7 天；
    缺失日期会用该周已有数据均值补齐，保证分类输入长度稳定。
    """

    if not daily_rows:
        return []

    rows_by_date = {item["date"]: item for item in daily_rows}
    start_date = min(rows_by_date)
    end_date = max(rows_by_date)
    current_week_start = start_date - timedelta(days=start_date.weekday())
    final_week_start = end_date - timedelta(days=end_date.weekday())
    windows: list[list[dict]] = []

    while current_week_start <= final_week_start:
        # 构造自然周日期序列。
        # 如果该周部分日期缺失，则使用该周已有天数的平均峰谷指标补齐。
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


def classification_dto(record: ClassificationResult) -> dict:
    """转换分类结果为接口返回结构。"""

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

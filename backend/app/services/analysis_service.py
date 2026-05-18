from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import pandas as pd

from app.models import AnalysisResult, Dataset
from app.services.common import read_json, to_iso, write_json


def build_analysis_payload(
    normalized_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    *,
    peak_periods: list[str],
    valley_periods: list[str],
    detail_path: Path,
) -> dict:
    if normalized_df.empty or daily_df.empty:
        payload = {
            "summary": {
                "total_kwh": 0,
                "daily_avg_kwh": 0,
                "max_load_w": 0,
                "max_load_time": "",
                "min_load_w": 0,
                "min_load_time": "",
                "peak_kwh": 0,
                "valley_kwh": 0,
                "peak_ratio": 0,
                "valley_ratio": 0,
            },
            "peak_valley_config": {
                "peak": peak_periods,
                "valley": valley_periods,
            },
            "charts": {
                "daily_trend": [],
                "weekly_trend": [],
                "typical_day_curve": [],
                "peak_valley_pie": [],
            },
            "detail_path": str(detail_path),
            "updated_at": "",
        }
        write_json(detail_path, payload)
        return payload

    total_kwh = float(daily_df["total_kwh"].sum())
    peak_kwh = float(daily_df["peak_kwh"].sum())
    valley_kwh = float(daily_df["valley_kwh"].sum())

    max_row = normalized_df.loc[normalized_df["aggregate_w"].idxmax()]
    min_row = normalized_df.loc[normalized_df["aggregate_w"].idxmin()]

    daily_trend = [
        {"date": item["date"].strftime("%Y-%m-%d"), "kwh": round(float(item["total_kwh"]), 4)}
        for item in daily_df[["date", "total_kwh"]].to_dict("records")
    ]

    weekly_buckets: dict[str, dict] = defaultdict(lambda: {"total": 0.0, "end": None})
    for item in daily_df.to_dict("records"):
        day = item["date"]
        week_start = day - timedelta(days=day.weekday())
        key = week_start.strftime("%Y-%m-%d")
        weekly_buckets[key]["total"] += float(item["total_kwh"])
        weekly_buckets[key]["end"] = week_start + timedelta(days=6)

    weekly_trend = [
        {
            "week_start": key,
            "week_end": value["end"].strftime("%Y-%m-%d"),
            "kwh": round(value["total"], 4),
        }
        for key, value in sorted(weekly_buckets.items())
    ]

    typical_day = (
        normalized_df.assign(hour=normalized_df["timestamp"].dt.hour)
        .groupby("hour")["aggregate_w"]
        .mean()
        .reset_index()
    )
    typical_day_curve = [
        {"hour": int(item["hour"]), "avg_load_w": round(float(item["aggregate_w"]), 2)}
        for item in typical_day.to_dict("records")
    ]

    summary = {
        "total_kwh": round(total_kwh, 4),
        "daily_avg_kwh": round(total_kwh / max(len(daily_df), 1), 4),
        "max_load_w": round(float(max_row["aggregate_w"]), 2),
        "max_load_time": to_iso(max_row["timestamp"]),
        "min_load_w": round(float(min_row["aggregate_w"]), 2),
        "min_load_time": to_iso(min_row["timestamp"]),
        "peak_kwh": round(peak_kwh, 4),
        "valley_kwh": round(valley_kwh, 4),
        "peak_ratio": round(peak_kwh / max(total_kwh, 1e-6), 4),
        "valley_ratio": round(valley_kwh / max(total_kwh, 1e-6), 4),
    }

    payload = {
        "summary": summary,
        "peak_valley_config": {
            "peak": peak_periods,
            "valley": valley_periods,
        },
        "charts": {
            "daily_trend": daily_trend,
            "weekly_trend": weekly_trend,
            "typical_day_curve": typical_day_curve,
            "peak_valley_pie": [
                {"name": "峰时", "ratio": summary["peak_ratio"], "kwh": summary["peak_kwh"]},
                {"name": "谷时", "ratio": summary["valley_ratio"], "kwh": summary["valley_kwh"]},
            ],
        },
        "detail_path": str(detail_path),
        "updated_at": daily_df["date"].max().strftime("%Y-%m-%dT00:00:00"),
    }
    write_json(detail_path, payload)
    return payload


def upsert_analysis_result(dataset: Dataset, payload: dict, detail_path: Path) -> AnalysisResult:
    analysis = AnalysisResult.query.filter_by(dataset_id=dataset.id).first()
    if analysis is None:
        analysis = AnalysisResult(dataset_id=dataset.id)

    summary = payload["summary"]
    analysis.total_kwh = summary["total_kwh"]
    analysis.daily_avg_kwh = summary["daily_avg_kwh"]
    analysis.max_load_w = summary["max_load_w"]
    analysis.max_load_time = pd.to_datetime(summary["max_load_time"]).to_pydatetime() if summary["max_load_time"] else None
    analysis.min_load_w = summary["min_load_w"]
    analysis.min_load_time = pd.to_datetime(summary["min_load_time"]).to_pydatetime() if summary["min_load_time"] else None
    analysis.peak_kwh = summary["peak_kwh"]
    analysis.valley_kwh = summary["valley_kwh"]
    analysis.peak_ratio = summary["peak_ratio"]
    analysis.valley_ratio = summary["valley_ratio"]
    analysis.summary_json = payload
    analysis.detail_path = str(detail_path)
    return analysis


def get_analysis_payload(dataset_id: int) -> dict:
    analysis = AnalysisResult.query.filter_by(dataset_id=dataset_id).first()
    if analysis is None:
        raise ValueError("分析结果不存在")
    return read_json(analysis.detail_path, default=analysis.summary_json)

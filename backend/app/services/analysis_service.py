from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from pathlib import Path

from typing import cast

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
    """构建用电分析结果。

    输入包括标准化明细数据和日级聚合数据；
    输出包括汇总指标、峰谷配置、图表数据和明细文件路径。
    """

    if normalized_df.empty or daily_df.empty:
        # 空数据集仍返回完整结构，避免前端因为字段缺失而报错。
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

    # 汇总核心能耗指标。
    # total_kwh 是总用电量，peak_kwh/valley_kwh 用于计算峰谷占比。
    total_kwh = float(daily_df["total_kwh"].sum())
    peak_kwh = float(daily_df["peak_kwh"].sum())
    valley_kwh = float(daily_df["valley_kwh"].sum())

    # 从标准化明细中找最大/最小负荷点。
    # 这两个时间点通常是答辩时解释用户用电峰值和低谷的依据。
    max_row = normalized_df.loc[normalized_df["aggregate_w"].idxmax()]
    min_row = normalized_df.loc[normalized_df["aggregate_w"].idxmin()]

    # 日趋势图：每一天的总用电量。
    daily_subset = cast(pd.DataFrame, daily_df[["date", "total_kwh"]])
    daily_trend = [
        {"date": item["date"].strftime("%Y-%m-%d"), "kwh": round(float(item["total_kwh"]), 4)}
        for item in daily_subset.to_dict("records")
    ]

    # 周趋势图：按自然周聚合每日用电量。
    # week_start 使用周一，week_end 使用周日，便于前端横轴展示。
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

    # 典型日曲线：按小时求平均负荷。
    # 它用于展示用户一天内的典型用电节奏。
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

    # payload 是分析结果的完整前端数据结构。
    # 数据库保存 summary 和 detail_path，完整图表明细写入 JSON 文件。
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
    """创建或更新某个数据集的分析结果记录。"""

    analysis = AnalysisResult.query.filter_by(dataset_id=dataset.id).first()
    if analysis is None:
        analysis = AnalysisResult(dataset_id=dataset.id)

    # 将 JSON 摘要同步到关系型字段，便于后续列表、报告或条件查询直接使用。
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
    """读取分析结果完整 payload。"""

    analysis = AnalysisResult.query.filter_by(dataset_id=dataset_id).first()
    if analysis is None:
        raise ValueError("分析结果不存在")
    return read_json(analysis.detail_path, default=analysis.summary_json)

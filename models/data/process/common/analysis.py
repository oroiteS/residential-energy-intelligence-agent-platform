"""基础时序日级阈值分析工具。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.process.common.base import load_base_dataset, matches_low_signal_day_rule
from data.process.common.constants import (
    LOW_SIGNAL_DAY_MAX_THRESHOLD_W,
    LOW_SIGNAL_DAY_NON_ZERO_RATIO_THRESHOLD,
    LOW_SIGNAL_DAY_RANGE_THRESHOLD_W,
    LOW_SIGNAL_DAY_STD_THRESHOLD_W,
    LOW_SIGNAL_DAY_UNIQUE_VALUE_COUNT_THRESHOLD,
    SLOTS_PER_DAY,
)
from data.process.common.progress import log_stage

NEAR_FLAT_DELTA_W = 1.0
DEFAULT_PERCENTILES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
DEFAULT_ANALYSIS_METRICS = (
    "day_mean_w",
    "day_std_w",
    "day_min_w",
    "day_max_w",
    "day_range_w",
    "non_zero_ratio",
    "unique_value_count",
    "near_flat_ratio",
    "imputed_ratio",
)


def _build_daily_records(base_df: pd.DataFrame) -> list[dict[str, object]]:
    daily_records: list[dict[str, object]] = []
    grouped = base_df.groupby(["house_id", "date"], sort=True)
    for (house_id, date_value), day_df in grouped:
        sorted_day_df = day_df.sort_values("slot_index")
        aggregate_values = sorted_day_df["aggregate"].to_numpy(dtype=np.float32)
        row_count = int(len(sorted_day_df))
        finite_mask = np.isfinite(aggregate_values)
        finite_values = aggregate_values[finite_mask]
        non_zero_ratio = (
            float(np.mean(finite_values > 0.0)) if len(finite_values) else 0.0
        )
        unique_value_count = int(pd.Series(aggregate_values).nunique(dropna=True))
        if len(finite_values) <= 1:
            near_flat_ratio = 1.0
        else:
            diffs = np.abs(np.diff(finite_values))
            near_flat_ratio = float(np.mean(diffs <= NEAR_FLAT_DELTA_W))

        imputed_points = int(sorted_day_df["is_imputed_point"].sum())
        unresolved_points = int(sorted_day_df["is_unresolved_point"].sum())
        clipped_points = int(sorted_day_df["is_clipped_point"].sum())
        day_max_w = float(np.nanmax(aggregate_values)) if len(finite_values) else 0.0
        day_min_w = float(np.nanmin(aggregate_values)) if len(finite_values) else 0.0
        day_range_w = float(day_max_w - day_min_w)

        daily_records.append(
            {
                "house_id": str(house_id),
                "source_dataset": str(sorted_day_df["source_dataset"].iloc[0]),
                "date": str(date_value),
                "row_count": row_count,
                "slot_count": int(sorted_day_df["slot_index"].nunique()),
                "day_mean_w": float(np.nanmean(aggregate_values)) if len(finite_values) else 0.0,
                "day_std_w": float(np.nanstd(aggregate_values)) if len(finite_values) else 0.0,
                "day_min_w": day_min_w,
                "day_max_w": day_max_w,
                "day_range_w": day_range_w,
                "non_zero_ratio": non_zero_ratio,
                "unique_value_count": unique_value_count,
                "near_flat_ratio": near_flat_ratio,
                "imputed_points": imputed_points,
                "imputed_ratio": float(imputed_points) / float(SLOTS_PER_DAY),
                "unresolved_points": unresolved_points,
                "clipped_points": clipped_points,
                "is_complete_day": bool(
                    row_count == SLOTS_PER_DAY
                    and int(sorted_day_df["slot_index"].nunique()) == SLOTS_PER_DAY
                    and int(sorted_day_df["slot_index"].min()) == 0
                    and int(sorted_day_df["slot_index"].max()) == SLOTS_PER_DAY - 1
                ),
                "is_full_zero_day": bool(
                    len(finite_values) > 0
                    and np.all(np.isclose(finite_values, 0.0))
                ),
                "matches_low_signal_rule": bool(
                    matches_low_signal_day_rule(
                        day_max_w=day_max_w,
                        day_range_w=day_range_w,
                        day_std_w=float(np.nanstd(aggregate_values)) if len(finite_values) else 0.0,
                        non_zero_ratio=non_zero_ratio,
                        unique_value_count=unique_value_count,
                    )
                ),
            }
        )
    return daily_records


def build_daily_threshold_frame(base_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "house_id",
        "source_dataset",
        "date",
        "slot_index",
        "aggregate",
        "is_imputed_point",
        "is_unresolved_point",
        "is_clipped_point",
    }
    missing_columns = required_columns.difference(base_df.columns)
    if missing_columns:
        raise ValueError(
            "基础时序缺少阈值分析所需字段: "
            f"{sorted(missing_columns)}"
        )

    daily_df = pd.DataFrame(_build_daily_records(base_df))
    if daily_df.empty:
        raise ValueError("基础时序为空，无法生成日级阈值分析")
    return daily_df.sort_values(["source_dataset", "house_id", "date"]).reset_index(drop=True)


def _summarize_scope(
    daily_df: pd.DataFrame,
    scope_type: str,
    scope_value: str,
    percentiles: tuple[float, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric in DEFAULT_ANALYSIS_METRICS:
        raw_metric_values = daily_df[metric].to_numpy(dtype=np.float64)
        metric_values = raw_metric_values[np.isfinite(raw_metric_values)]
        if len(metric_values) == 0:
            row = {
                "scope_type": scope_type,
                "scope_value": scope_value,
                "metric": metric,
                "count": 0,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
            }
            for percentile in percentiles:
                percentile_key = f"p{int(round(percentile * 100)):02d}"
                row[percentile_key] = np.nan
            rows.append(row)
            continue
        quantile_values = np.quantile(metric_values, percentiles)
        row: dict[str, object] = {
            "scope_type": scope_type,
            "scope_value": scope_value,
            "metric": metric,
            "count": int(len(metric_values)),
            "mean": float(np.mean(metric_values)),
            "std": float(np.std(metric_values)),
            "min": float(np.min(metric_values)),
            "max": float(np.max(metric_values)),
        }
        for percentile, value in zip(percentiles, quantile_values, strict=True):
            percentile_key = f"p{int(round(percentile * 100)):02d}"
            row[percentile_key] = float(value)
        rows.append(row)
    return rows


def summarize_threshold_distributions(
    daily_df: pd.DataFrame,
    percentiles: tuple[float, ...] = DEFAULT_PERCENTILES,
) -> pd.DataFrame:
    summary_rows = _summarize_scope(
        daily_df=daily_df,
        scope_type="overall",
        scope_value="all",
        percentiles=percentiles,
    )
    for source_dataset, source_df in daily_df.groupby("source_dataset", sort=True):
        summary_rows.extend(
            _summarize_scope(
                daily_df=source_df.reset_index(drop=True),
                scope_type="source_dataset",
                scope_value=str(source_dataset),
                percentiles=percentiles,
            )
        )
    return pd.DataFrame(summary_rows)


def _build_candidate_examples(
    daily_df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    candidate_df = daily_df.copy()
    candidate_df["low_signal_score"] = (
        candidate_df["day_max_w"] * 0.6
        + candidate_df["day_range_w"] * 0.3
        + candidate_df["day_std_w"] * 0.1
    )
    sort_columns = [
        "matches_low_signal_rule",
        "is_full_zero_day",
        "low_signal_score",
        "day_max_w",
        "day_range_w",
        "day_std_w",
    ]
    ascending = [False, False, True, True, True, True]
    return (
        candidate_df.sort_values(sort_columns, ascending=ascending)
        .head(top_k)
        .reset_index(drop=True)
    )


def analyze_base_thresholds(
    base_dir: Path,
    output_dir: Path,
    top_k: int = 200,
) -> dict[str, object]:
    log_stage("加载基础时序并执行日级阈值分析")
    base_df = load_base_dataset(base_dir)
    daily_df = build_daily_threshold_frame(base_df)
    distribution_df = summarize_threshold_distributions(daily_df)
    candidate_df = _build_candidate_examples(daily_df, top_k=top_k)

    output_dir.mkdir(parents=True, exist_ok=True)
    daily_stats_path = output_dir / "daily_threshold_stats.csv"
    distribution_path = output_dir / "threshold_distribution_summary.csv"
    candidate_path = output_dir / "low_signal_candidates.csv"
    summary_path = output_dir / "threshold_analysis_summary.json"

    daily_df.to_csv(daily_stats_path, index=False)
    distribution_df.to_csv(distribution_path, index=False)
    candidate_df.to_csv(candidate_path, index=False)

    summary = {
        "base_dir": str(base_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "house_count": int(daily_df["house_id"].nunique()),
        "day_count": int(len(daily_df)),
        "source_count": int(daily_df["source_dataset"].nunique()),
        "full_zero_day_count": int(daily_df["is_full_zero_day"].sum()),
        "full_zero_day_ratio": float(daily_df["is_full_zero_day"].mean()),
        "low_signal_rule_count": int(daily_df["matches_low_signal_rule"].sum()),
        "low_signal_rule_ratio": float(daily_df["matches_low_signal_rule"].mean()),
        "near_flat_delta_w": NEAR_FLAT_DELTA_W,
        "low_signal_day_max_threshold_w": LOW_SIGNAL_DAY_MAX_THRESHOLD_W,
        "low_signal_day_range_threshold_w": LOW_SIGNAL_DAY_RANGE_THRESHOLD_W,
        "low_signal_day_std_threshold_w": LOW_SIGNAL_DAY_STD_THRESHOLD_W,
        "low_signal_day_non_zero_ratio_threshold": LOW_SIGNAL_DAY_NON_ZERO_RATIO_THRESHOLD,
        "low_signal_day_unique_value_count_threshold": LOW_SIGNAL_DAY_UNIQUE_VALUE_COUNT_THRESHOLD,
        "artifacts": {
            "daily_stats": str(daily_stats_path),
            "distribution_summary": str(distribution_path),
            "low_signal_candidates": str(candidate_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path)
    return summary


__all__ = [
    "analyze_base_thresholds",
    "build_daily_threshold_frame",
    "summarize_threshold_distributions",
]

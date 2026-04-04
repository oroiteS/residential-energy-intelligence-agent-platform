"""导出前后端联调用的代表性测试样本。"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from data.process.common.base import select_complete_days

LABEL_SCENARIOS = (
    "day_high_night_low",
    "day_low_night_high",
    "all_day_high",
    "all_day_low",
)


def _normalize_house_id(house_id: str | None) -> str | None:
    if house_id is None:
        return None
    normalized = house_id.strip()
    if normalized.startswith("house_"):
        normalized = normalized.removeprefix("house_")
    return normalized


def _load_labels(labels_path: Path) -> pd.DataFrame:
    if not labels_path.exists():
        raise FileNotFoundError(f"未找到分类标签文件: {labels_path}")

    labels_df = pd.read_csv(
        labels_path,
        usecols=["house_id", "date", "label_name", "day_mean", "night_mean", "full_mean"],
    )
    labels_df["house_id"] = labels_df["house_id"].astype(str)
    labels_df["date"] = pd.to_datetime(labels_df["date"]).dt.date
    return labels_df


def _infer_house_id(labels_df: pd.DataFrame) -> str:
    house_rank = (
        labels_df.groupby("house_id")
        .agg(
            label_count=("label_name", "nunique"),
            sample_count=("date", "size"),
        )
        .reset_index()
        .sort_values(
            ["label_count", "sample_count", "house_id"],
            ascending=[False, False, True],
        )
    )
    if house_rank.empty:
        raise ValueError("分类标签中没有可用的家庭数据")
    return str(house_rank.iloc[0]["house_id"])


def _load_house_base_data(base_dir: Path, house_id: str) -> pd.DataFrame:
    base_path = base_dir / f"house_{house_id}_base_15min.csv"
    if not base_path.exists():
        raise FileNotFoundError(f"未找到家庭基础时序文件: {base_path}")

    base_df = pd.read_csv(base_path, parse_dates=["timestamp"])
    base_df["house_id"] = base_df["house_id"].astype(str)
    base_df["date"] = pd.to_datetime(base_df["date"]).dt.date
    return base_df


def _has_contiguous_window(valid_dates: set[date], target_date: date, window_days: int) -> bool:
    return all(
        target_date - timedelta(days=offset) in valid_dates
        for offset in range(window_days)
    )


def _build_day_summary(
    house_base_df: pd.DataFrame,
    house_labels_df: pd.DataFrame,
    window_days: int,
) -> pd.DataFrame:
    complete_df = select_complete_days(house_base_df)
    day_summary = (
        complete_df.groupby("date")
        .agg(
            full_mean=("aggregate", "mean"),
            burst_sum=("burst_event_count", "sum"),
            active_mean=("active_appliance_count", "mean"),
            is_weekend=("is_weekend", "max"),
            row_count=("slot_index", "size"),
        )
        .reset_index()
    )
    day_summary = day_summary.merge(
        house_labels_df,
        on="date",
        how="left",
        suffixes=("", "_label"),
    )
    valid_dates = set(day_summary["date"])
    day_summary["window_ok"] = day_summary["date"].apply(
        lambda item: _has_contiguous_window(valid_dates, item, window_days)
    )
    return day_summary.sort_values("date").reset_index(drop=True)


def _pick_label_day(
    day_summary: pd.DataFrame,
    label_name: str,
    selected_dates: set[date],
) -> dict[str, object] | None:
    candidates = day_summary.loc[
        (day_summary["label_name"] == label_name)
        & day_summary["window_ok"]
        & (~day_summary["date"].isin(selected_dates))
    ].copy()
    if candidates.empty:
        return None

    label_median = float(candidates["full_mean"].median())
    candidates["score"] = (candidates["full_mean"] - label_median).abs()
    row = candidates.sort_values(
        ["score", "burst_sum", "date"],
        ascending=[True, False, True],
    ).iloc[0]
    return {
        "scenario_name": label_name,
        "target_date": row["date"],
        "label_name": row["label_name"],
    }


def _pick_by_ranking(
    day_summary: pd.DataFrame,
    selected_dates: set[date],
    scenario_name: str,
    ranking: Callable[[pd.DataFrame], pd.DataFrame],
) -> dict[str, object] | None:
    candidates = day_summary.loc[
        day_summary["window_ok"] & (~day_summary["date"].isin(selected_dates))
    ].copy()
    if candidates.empty:
        return None

    ranked = ranking(candidates)
    if ranked.empty:
        return None

    row = ranked.iloc[0]
    return {
        "scenario_name": scenario_name,
        "target_date": row["date"],
        "label_name": row.get("label_name"),
    }


def _choose_scenarios(day_summary: pd.DataFrame, count: int) -> list[dict[str, object]]:
    selections: list[dict[str, object]] = []
    selected_dates: set[date] = set()

    for label_name in LABEL_SCENARIOS:
        if len(selections) >= count:
            break
        picked = _pick_label_day(day_summary, label_name, selected_dates)
        if picked is None:
            continue
        selections.append(picked)
        selected_dates.add(picked["target_date"])

    extra_rankings: tuple[tuple[str, Callable[[pd.DataFrame], pd.DataFrame]], ...] = (
        (
            "burst_heavy",
            lambda df: df.sort_values(
                ["burst_sum", "full_mean", "date"],
                ascending=[False, False, True],
            ),
        ),
        (
            "weekend_peak",
            lambda df: df.loc[df["is_weekend"] == 1].sort_values(
                ["full_mean", "burst_sum", "date"],
                ascending=[False, False, True],
            ),
        ),
        (
            "highest_load",
            lambda df: df.sort_values(
                ["full_mean", "burst_sum", "date"],
                ascending=[False, False, True],
            ),
        ),
        (
            "lowest_load",
            lambda df: df.sort_values(
                ["full_mean", "burst_sum", "date"],
                ascending=[True, False, True],
            ),
        ),
        (
            "active_appliances",
            lambda df: df.sort_values(
                ["active_mean", "full_mean", "date"],
                ascending=[False, False, True],
            ),
        ),
    )

    for scenario_name, ranking in extra_rankings:
        if len(selections) >= count:
            break
        picked = _pick_by_ranking(day_summary, selected_dates, scenario_name, ranking)
        if picked is None:
            continue
        selections.append(picked)
        selected_dates.add(picked["target_date"])

    if len(selections) < count:
        remaining = day_summary.loc[
            day_summary["window_ok"] & (~day_summary["date"].isin(selected_dates))
        ].sort_values(["full_mean", "date"], ascending=[False, True])
        for row in remaining.itertuples(index=False):
            if len(selections) >= count:
                break
            selections.append(
                {
                    "scenario_name": f"extra_{len(selections) + 1:02d}",
                    "target_date": row.date,
                    "label_name": getattr(row, "label_name", None),
                }
            )
            selected_dates.add(row.date)

    return selections


def export_representative_test_samples(
    base_dir: Path,
    labels_path: Path,
    output_dir: Path,
    house_id: str | None = None,
    count: int = 5,
    window_days: int = 7,
) -> list[dict[str, object]]:
    """导出约 5 份代表不同用电场景的测试 CSV。"""
    if count <= 0:
        raise ValueError("count 必须大于 0")
    if window_days <= 0:
        raise ValueError("window_days 必须大于 0")

    labels_df = _load_labels(labels_path)
    normalized_house_id = _normalize_house_id(house_id) or _infer_house_id(labels_df)

    house_labels_df = labels_df.loc[labels_df["house_id"] == normalized_house_id].copy()
    if house_labels_df.empty:
        raise ValueError(f"在分类标签中未找到家庭 {normalized_house_id}")

    house_base_df = _load_house_base_data(base_dir, normalized_house_id)
    day_summary = _build_day_summary(
        house_base_df=house_base_df,
        house_labels_df=house_labels_df,
        window_days=window_days,
    )
    selections = _choose_scenarios(day_summary=day_summary, count=count)
    if not selections:
        raise ValueError(f"家庭 {normalized_house_id} 没有可导出的连续样本窗口")

    complete_df = select_complete_days(house_base_df)
    sample_output_dir = output_dir / f"house_{normalized_house_id}"
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    exported_samples: list[dict[str, object]] = []
    for index, selection in enumerate(selections, start=1):
        target_date = selection["target_date"]
        window_start = target_date - timedelta(days=window_days - 1)
        sample_df = complete_df.loc[
            (complete_df["date"] >= window_start)
            & (complete_df["date"] <= target_date)
        ].copy()
        expected_rows = window_days * 96
        if len(sample_df) != expected_rows:
            raise ValueError(
                f"家庭 {normalized_house_id} 在 {target_date} 的样本窗口数据不完整"
            )

        file_name = (
            f"house_{normalized_house_id}_"
            f"{index:02d}_{selection['scenario_name']}_"
            f"{target_date.isoformat()}.csv"
        )
        file_path = sample_output_dir / file_name
        sample_df.to_csv(file_path, index=False)
        exported_samples.append(
            {
                "house_id": normalized_house_id,
                "scenario_name": selection["scenario_name"],
                "label_name": selection["label_name"],
                "target_date": target_date.isoformat(),
                "window_start": window_start.isoformat(),
                "window_days": window_days,
                "row_count": len(sample_df),
                "output_path": str(file_path),
                "output_dir": str(sample_output_dir),
            }
        )

    return exported_samples

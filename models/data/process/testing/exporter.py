"""导出基于预测验证集的前后端联调用测试样本。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

from data.process.common.base import select_complete_days

MODELS_ROOT = Path(__file__).resolve().parents[3]


@dataclass(slots=True)
class ForecastSplitConfig:
    data_path: Path
    split_mode: str
    train_ratio: float
    val_ratio: float
    seed: int
    feature_names: list[str]


def _resolve_models_relative_path(path_value: str, models_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (models_root / path).resolve()


def _load_forecast_split_config(config_path: Path) -> ForecastSplitConfig:
    resolved_config_path = config_path.resolve()
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"未找到预测配置文件: {resolved_config_path}")

    raw_config = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8")) or {}
    data_raw = raw_config.get("data")
    train_raw = raw_config.get("train")
    if not isinstance(data_raw, dict):
        raise ValueError(f"预测配置缺少 data 段: {resolved_config_path}")
    if not isinstance(train_raw, dict):
        raise ValueError(f"预测配置缺少 train 段: {resolved_config_path}")

    data_path_value = data_raw.get("data_path")
    if not data_path_value:
        raise ValueError(f"预测配置缺少 data.data_path: {resolved_config_path}")

    feature_names = [
        str(item)
        for item in data_raw.get(
            "feature_names",
            ["aggregate", "slot_sin", "slot_cos", "weekday_sin", "weekday_cos"],
        )
    ]
    if not feature_names:
        raise ValueError(f"预测配置中的 data.feature_names 不能为空: {resolved_config_path}")

    return ForecastSplitConfig(
        data_path=_resolve_models_relative_path(str(data_path_value), MODELS_ROOT),
        split_mode=str(data_raw.get("split_mode", "by_house")),
        train_ratio=float(data_raw.get("train_ratio", 0.7)),
        val_ratio=float(data_raw.get("val_ratio", 0.15)),
        seed=int(train_raw.get("seed", 42)),
        feature_names=feature_names,
    )


def _load_validation_forecast_index(config_path: Path) -> pd.DataFrame:
    split_config = _load_forecast_split_config(config_path)
    sample_index = pd.read_csv(
        split_config.data_path,
        usecols=[
            "sample_id",
            "house_id",
            "input_start",
            "input_end",
            "target_start",
            "target_end",
        ],
    )
    sample_index["house_id"] = sample_index["house_id"].astype(str)
    if sample_index.empty:
        raise ValueError(f"预测样本文件为空: {split_config.data_path}")

    rng = random.Random(split_config.seed)
    if split_config.split_mode == "by_house":
        unique_houses = sorted(sample_index["house_id"].unique().tolist())
        if len(unique_houses) < 3:
            raise ValueError("按家庭切分至少需要 3 个不同家庭")
        shuffled_houses = unique_houses.copy()
        rng.shuffle(shuffled_houses)
        _, validation_houses, _ = _split_by_ratio(
            shuffled_houses,
            train_ratio=split_config.train_ratio,
            val_ratio=split_config.val_ratio,
        )
        validation_index = sample_index.loc[
            sample_index["house_id"].isin(set(validation_houses))
        ].copy()
    else:
        shuffled_positions = list(range(len(sample_index)))
        rng.shuffle(shuffled_positions)
        _, validation_positions, _ = _split_by_ratio(
            shuffled_positions,
            train_ratio=split_config.train_ratio,
            val_ratio=split_config.val_ratio,
        )
        validation_index = sample_index.iloc[validation_positions].copy()

    if validation_index.empty:
        raise ValueError("预测验证集为空，无法导出 testing 样本")

    validation_index["target_date"] = pd.to_datetime(
        validation_index["target_start"]
    ).dt.date
    return validation_index


def _split_by_ratio(
    values: list[object],
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[object], list[object], list[object]]:
    total = len(values)
    if total < 3:
        raise ValueError("样本数过少，无法切分训练/验证/测试集")

    train_end = max(1, int(total * train_ratio))
    val_end = max(train_end + 1, int(total * (train_ratio + val_ratio)))
    val_end = min(val_end, total - 1)

    train_values = values[:train_end]
    val_values = values[train_end:val_end]
    test_values = values[val_end:]

    if not val_values:
        val_values = [train_values.pop()]
    if not test_values:
        test_values = [val_values.pop()]
    return train_values, val_values, test_values


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

    labels_df = pd.read_csv(labels_path)
    required_columns = {"house_id", "date", "label_name"}
    missing_columns = required_columns.difference(labels_df.columns)
    if missing_columns:
        raise ValueError(f"分类标签文件缺少必要字段: {sorted(missing_columns)}")

    keep_columns = ["house_id", "date", "label_name"]
    if "cluster_id" in labels_df.columns:
        keep_columns.append("cluster_id")
    labels_df = labels_df.loc[:, keep_columns].copy()
    labels_df["house_id"] = labels_df["house_id"].astype(str)
    labels_df["date"] = pd.to_datetime(labels_df["date"]).dt.date
    return labels_df


def _infer_house_id(
    labels_df: pd.DataFrame,
    validation_index: pd.DataFrame,
) -> str:
    candidate_df = validation_index.merge(
        labels_df[["house_id", "date", "label_name"]],
        left_on=["house_id", "target_date"],
        right_on=["house_id", "date"],
        how="left",
    )
    house_rank = (
        candidate_df.groupby("house_id")
        .agg(
            label_count=("label_name", "nunique"),
            sample_count=("target_date", "size"),
        )
        .reset_index()
        .sort_values(
            ["label_count", "sample_count", "house_id"],
            ascending=[False, False, True],
        )
    )
    if house_rank.empty:
        raise ValueError("验证集中没有可用的家庭数据")
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
    validation_samples_df: pd.DataFrame,
    window_days: int,
) -> pd.DataFrame:
    complete_df = select_complete_days(house_base_df)
    valid_dates = set(complete_df["date"])
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
    day_summary["window_ok"] = day_summary["date"].apply(
        lambda item: _has_contiguous_window(valid_dates, item, window_days)
    )
    validation_target_dates = set(validation_samples_df["target_date"])
    day_summary = day_summary.loc[
        day_summary["date"].isin(validation_target_dates)
    ].copy()
    validation_sample_map = validation_samples_df.set_index("target_date")["sample_id"]
    day_summary["sample_id"] = day_summary["date"].map(validation_sample_map)
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
        "sample_id": row["sample_id"],
    }


def _list_label_scenarios(day_summary: pd.DataFrame) -> list[str]:
    labeled_rows = day_summary.loc[day_summary["label_name"].notna(), "label_name"].astype(str)
    if labeled_rows.empty:
        return []
    label_counts = labeled_rows.value_counts()
    return label_counts.index.tolist()


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
        "sample_id": row.get("sample_id"),
    }


def _choose_scenarios(day_summary: pd.DataFrame, count: int) -> list[dict[str, object]]:
    selections: list[dict[str, object]] = []
    selected_dates: set[date] = set()

    for label_name in _list_label_scenarios(day_summary):
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
                    "sample_id": getattr(row, "sample_id", None),
                }
            )
            selected_dates.add(row.date)

    return selections


def export_representative_test_samples(
    base_dir: Path,
    labels_path: Path,
    output_dir: Path,
    forecast_config_path: Path,
    house_id: str | None = None,
    count: int = 5,
    window_days: int = 7,
) -> list[dict[str, object]]:
    """仅基于预测验证集导出约 5 份代表不同用电场景的测试 CSV。"""
    if count <= 0:
        raise ValueError("count 必须大于 0")
    if window_days <= 0:
        raise ValueError("window_days 必须大于 0")

    labels_df = _load_labels(labels_path)
    validation_index = _load_validation_forecast_index(forecast_config_path)
    normalized_house_id = _normalize_house_id(house_id) or _infer_house_id(
        labels_df=labels_df,
        validation_index=validation_index,
    )

    house_validation_df = validation_index.loc[
        validation_index["house_id"] == normalized_house_id
    ].copy()
    if house_validation_df.empty:
        raise ValueError(f"在预测验证集中未找到家庭 {normalized_house_id}")

    house_labels_df = labels_df.loc[labels_df["house_id"] == normalized_house_id].copy()
    if house_labels_df.empty:
        raise ValueError(f"在分类标签中未找到家庭 {normalized_house_id}")

    house_base_df = _load_house_base_data(base_dir, normalized_house_id)
    day_summary = _build_day_summary(
        house_base_df=house_base_df,
        house_labels_df=house_labels_df,
        validation_samples_df=house_validation_df,
        window_days=window_days,
    )
    selections = _choose_scenarios(day_summary=day_summary, count=count)
    if not selections:
        raise ValueError(f"家庭 {normalized_house_id} 在验证集中没有可导出的连续样本窗口")

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
                "sample_id": selection["sample_id"],
                "target_date": target_date.isoformat(),
                "window_start": window_start.isoformat(),
                "window_days": window_days,
                "row_count": len(sample_df),
                "output_path": str(file_path),
                "output_dir": str(sample_output_dir),
            }
        )

    return exported_samples


def export_live_sample(
    base_dir: Path,
    labels_path: Path,
    output_path: Path,
    forecast_config_path: Path,
    house_id: str | None = None,
    window_days: int = 21,
) -> dict[str, object]:
    """从预测验证集导出一份供 live 模块循环播放的连续窗口样本。"""
    if window_days <= 0:
        raise ValueError("window_days 必须大于 0")

    labels_df = _load_labels(labels_path)
    validation_index = _load_validation_forecast_index(forecast_config_path)
    normalized_house_id = _normalize_house_id(house_id) or _infer_house_id(
        labels_df=labels_df,
        validation_index=validation_index,
    )

    house_validation_df = validation_index.loc[
        validation_index["house_id"] == normalized_house_id
    ].copy()
    if house_validation_df.empty:
        raise ValueError(f"在预测验证集中未找到家庭 {normalized_house_id}")

    house_labels_df = labels_df.loc[labels_df["house_id"] == normalized_house_id].copy()
    if house_labels_df.empty:
        raise ValueError(f"在分类标签中未找到家庭 {normalized_house_id}")

    house_base_df = _load_house_base_data(base_dir, normalized_house_id)
    day_summary = _build_day_summary(
        house_base_df=house_base_df,
        house_labels_df=house_labels_df,
        validation_samples_df=house_validation_df,
        window_days=window_days,
    )
    selections = _choose_scenarios(day_summary=day_summary, count=1)
    if not selections:
        raise ValueError(f"家庭 {normalized_house_id} 在验证集中没有可导出的连续样本窗口")

    selected = selections[0]
    target_date = selected["target_date"]
    window_start = target_date - timedelta(days=window_days - 1)
    complete_df = select_complete_days(house_base_df)
    live_df = complete_df.loc[
        (complete_df["date"] >= window_start)
        & (complete_df["date"] <= target_date)
    ].copy()

    expected_rows = window_days * 96
    if len(live_df) != expected_rows:
        raise ValueError(
            f"家庭 {normalized_house_id} 在 {target_date} 的 live 连续窗口样本数据不完整"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    live_df.to_csv(output_path, index=False)
    return {
        "house_id": normalized_house_id,
        "scenario_name": selected["scenario_name"],
        "label_name": selected["label_name"],
        "sample_id": selected["sample_id"],
        "target_date": target_date.isoformat(),
        "window_start": window_start.isoformat(),
        "window_days": window_days,
        "row_count": len(live_df),
        "output_path": str(output_path),
    }


def export_live_week_sample(
    base_dir: Path,
    labels_path: Path,
    output_path: Path,
    forecast_config_path: Path,
    house_id: str | None = None,
    window_days: int = 21,
) -> dict[str, object]:
    """兼容旧名称，内部统一走连续窗口样本导出。"""

    return export_live_sample(
        base_dir=base_dir,
        labels_path=labels_path,
        output_path=output_path,
        forecast_config_path=forecast_config_path,
        house_id=house_id,
        window_days=window_days,
    )

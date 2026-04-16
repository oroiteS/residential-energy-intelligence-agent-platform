"""从预测验证集导出上传模拟样本与 live 模拟样本。"""

from __future__ import annotations

import argparse
import json
import math
import numbers
import random
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable
from xml.sax.saxutils import escape

import pandas as pd
import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.process.common.base import select_complete_days

MODELS_ROOT = Path(__file__).resolve().parent
DEFAULT_FORECAST_CONFIG = Path("forecast/tft/configs/config.yaml")
DEFAULT_BASE_DIR = Path("data/processed/base_15min")
DEFAULT_LABELS_PATH = Path("data/processed/classification/classification_day_labels.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed/upload_simulation")

UPLOAD_COLUMNS = [
    "timestamp",
    "aggregate",
    "active_appliance_count",
    "burst_event_count",
]

LIVE_COLUMNS = [
    "house_id",
    "source_dataset",
    "timestamp",
    "date",
    "slot_index",
    "aggregate",
    "active_appliance_count",
    "burst_event_count",
    "is_weekend",
]


@dataclass(slots=True)
class ForecastSplitConfig:
    sample_index_path: Path
    split_mode: str
    train_ratio: float
    val_ratio: float
    seed: int


def _resolve_models_relative_path(path_value: Path) -> Path:
    if path_value.is_absolute():
        return path_value.resolve()
    return (MODELS_ROOT / path_value).resolve()


def _resolve_forecast_relative_path(path_value: str, models_root: Path) -> Path:
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

    data_path_value = data_raw.get("metadata_path") or data_raw.get("data_path")
    if not data_path_value:
        raise ValueError(
            "预测配置缺少 data.metadata_path / data.data_path: "
            f"{resolved_config_path}"
        )

    return ForecastSplitConfig(
        sample_index_path=_resolve_forecast_relative_path(
            str(data_path_value), MODELS_ROOT
        ),
        split_mode=str(data_raw.get("split_mode", "by_house")),
        train_ratio=float(data_raw.get("train_ratio", 0.7)),
        val_ratio=float(data_raw.get("val_ratio", 0.15)),
        seed=int(train_raw.get("seed", 42)),
    )


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


def load_validation_forecast_index(config_path: Path) -> pd.DataFrame:
    split_config = _load_forecast_split_config(config_path)
    sample_index = pd.read_csv(
        split_config.sample_index_path,
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
        raise ValueError(f"预测样本文件为空: {split_config.sample_index_path}")

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


def resolve_forecast_sample_index_path(config_path: Path) -> Path:
    return _load_forecast_split_config(config_path).sample_index_path


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


def _has_contiguous_window(
    valid_dates: set[date],
    target_date: date,
    window_days: int,
) -> bool:
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
    labeled_rows = day_summary.loc[
        day_summary["label_name"].notna(), "label_name"
    ].astype(str)
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
    if count <= 0:
        raise ValueError("count 必须大于 0")
    if window_days <= 0:
        raise ValueError("window_days 必须大于 0")

    labels_df = _load_labels(labels_path)
    validation_index = load_validation_forecast_index(forecast_config_path)
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
        raise ValueError(
            f"家庭 {normalized_house_id} 在验证集中没有可导出的连续样本窗口"
        )

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
    if window_days <= 0:
        raise ValueError("window_days 必须大于 0")

    labels_df = _load_labels(labels_path)
    validation_index = load_validation_forecast_index(forecast_config_path)
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
        raise ValueError(
            f"家庭 {normalized_house_id} 在验证集中没有可导出的连续样本窗口"
        )

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "从预测验证集导出 3 份上传模拟 CSV、3 份 XLSX，"
            "以及 1 份 live 模拟读取 CSV。"
        )
    )
    parser.add_argument(
        "--forecast-config",
        type=Path,
        default=DEFAULT_FORECAST_CONFIG,
        help="预测配置文件路径，默认: %(default)s",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="基础 15min 时序目录，默认: %(default)s",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help="分类标签文件路径，默认: %(default)s",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="导出根目录，默认: %(default)s",
    )
    parser.add_argument(
        "--house-id",
        type=str,
        default=None,
        help="指定家庭编号；为空时自动选择验证集中标签最完整的家庭",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=3,
        help="上传模拟样本数量，默认: %(default)s",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=7,
        help="上传模拟样本保留的连续天数，默认: %(default)s",
    )
    parser.add_argument(
        "--live-window-days",
        type=int,
        default=21,
        help="live 模拟样本保留的连续天数，默认: %(default)s",
    )
    return parser


def _ensure_columns(frame: pd.DataFrame, required_columns: list[str], source_path: Path) -> None:
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"文件缺少必要列 {missing_columns}: {source_path}")


def _read_subset_frame(source_path: Path, required_columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(source_path)
    _ensure_columns(frame, required_columns, source_path)
    return frame.loc[:, required_columns].copy()


def _write_csv(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def _excel_column_name(column_index: int) -> str:
    result = ""
    current = column_index + 1
    while current > 0:
        current, remainder = divmod(current - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _xml_cell(cell_ref: str, value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return f'<c r="{cell_ref}" t="inlineStr"><is><t></t></is></c>'

    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        numeric = format(float(value), ".15g")
        return f'<c r="{cell_ref}"><v>{numeric}</v></c>'

    text = str(value)
    preserve = ' xml:space="preserve"' if text != text.strip() else ""
    return (
        f'<c r="{cell_ref}" t="inlineStr"><is><t{preserve}>'
        f"{escape(text)}</t></is></c>"
    )


def _write_simple_xlsx(frame: pd.DataFrame, output_path: Path, sheet_name: str = "data") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[list[object]] = [frame.columns.tolist()]
    rows.extend(frame.where(pd.notna(frame), "").values.tolist())

    sheet_rows: list[str] = []
    for row_index, row_values in enumerate(rows, start=1):
        cell_xml = []
        for column_index, value in enumerate(row_values):
            cell_ref = f"{_excel_column_name(column_index)}{row_index}"
            cell_xml.append(_xml_cell(cell_ref, value))
        sheet_rows.append(f'<row r="{row_index}">{"".join(cell_xml)}</row>')

    worksheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(sheet_rows)}</sheetData>"
        "</worksheet>"
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheets>"
        f'<sheet name="{escape(sheet_name)}" sheetId="1" r:id="rId1"/>'
        "</sheets>"
        "</workbook>"
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/docProps/core.xml" '
        'ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        "</Types>"
    )
    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" '
        'Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" '
        'Target="docProps/app.xml"/>'
        "</Relationships>"
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )
    created_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    core_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        "<dc:creator>codex</dc:creator>"
        f"<dcterms:created xsi:type=\"dcterms:W3CDTF\">{created_at}</dcterms:created>"
        f"<dcterms:modified xsi:type=\"dcterms:W3CDTF\">{created_at}</dcterms:modified>"
        "</cp:coreProperties>"
    )
    app_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        "<Application>Codex</Application>"
        "</Properties>"
    )

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("_rels/.rels", root_rels_xml)
        archive.writestr("docProps/core.xml", core_xml)
        archive.writestr("docProps/app.xml", app_xml)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet_xml)


def _build_manifest_row(
    *,
    artifact_type: str,
    format_name: str,
    sample_meta: dict[str, object],
    output_path: Path,
    in_validation_index: bool,
    source_path: Path | None = None,
) -> dict[str, object]:
    return {
        "artifact_type": artifact_type,
        "format": format_name,
        "split": "validation",
        "in_validation_index": in_validation_index,
        "house_id": sample_meta["house_id"],
        "scenario_name": sample_meta["scenario_name"],
        "label_name": sample_meta.get("label_name"),
        "sample_id": sample_meta["sample_id"],
        "target_date": sample_meta["target_date"],
        "window_start": sample_meta["window_start"],
        "window_days": sample_meta["window_days"],
        "row_count": sample_meta["row_count"],
        "output_path": str(output_path.resolve()),
        "source_path": str(source_path.resolve()) if source_path else "",
    }


def run_export(args: argparse.Namespace) -> dict[str, object]:
    if args.sample_count < 3:
        raise ValueError("sample_count 至少为 3，才能满足上传模拟样本导出要求")
    if args.window_days < 7:
        raise ValueError("window_days 至少为 7")
    if args.live_window_days < 7:
        raise ValueError("live_window_days 至少为 7")

    forecast_config = _resolve_models_relative_path(args.forecast_config)
    base_dir = _resolve_models_relative_path(args.base_dir)
    labels_path = _resolve_models_relative_path(args.labels_path)
    output_root = _resolve_models_relative_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("validation_export_%Y%m%d_%H%M%S")
    run_output_dir = output_root / run_name
    upload_csv_dir = run_output_dir / "upload_csv"
    upload_xlsx_dir = run_output_dir / "upload_xlsx"
    live_dir = run_output_dir / "live"

    validation_index = load_validation_forecast_index(forecast_config)
    validation_sample_ids = set(validation_index["sample_id"].astype(str))

    manifest_rows: list[dict[str, object]] = []
    upload_csv_paths: list[str] = []
    upload_xlsx_paths: list[str] = []

    with tempfile.TemporaryDirectory(prefix="validation_export_", dir=run_output_dir.parent) as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        exported_samples = export_representative_test_samples(
            base_dir=base_dir,
            labels_path=labels_path,
            output_dir=temp_dir / "raw_upload_csv",
            forecast_config_path=forecast_config,
            house_id=args.house_id,
            count=args.sample_count,
            window_days=args.window_days,
        )
        if len(exported_samples) < args.sample_count:
            raise ValueError(
                f"验证集中仅导出了 {len(exported_samples)} 份上传模拟样本，少于要求的 {args.sample_count} 份"
            )

        for sample in exported_samples[: args.sample_count]:
            source_csv_path = Path(str(sample["output_path"]))
            upload_frame = _read_subset_frame(source_csv_path, UPLOAD_COLUMNS)

            house_dir_name = f"house_{sample['house_id']}"
            csv_output_path = upload_csv_dir / house_dir_name / source_csv_path.name
            xlsx_output_path = (
                upload_xlsx_dir / house_dir_name / source_csv_path.with_suffix(".xlsx").name
            )

            _write_csv(upload_frame, csv_output_path)
            _write_simple_xlsx(upload_frame, xlsx_output_path, sheet_name="upload_sample")

            in_validation_index = str(sample["sample_id"]) in validation_sample_ids
            if not in_validation_index:
                raise ValueError(
                    f"样本 {sample['sample_id']} 不在验证集索引中，终止导出"
                )

            manifest_rows.append(
                _build_manifest_row(
                    artifact_type="upload",
                    format_name="csv",
                    sample_meta=sample,
                    output_path=csv_output_path,
                    in_validation_index=in_validation_index,
                    source_path=None,
                )
            )
            manifest_rows.append(
                _build_manifest_row(
                    artifact_type="upload",
                    format_name="xlsx",
                    sample_meta=sample,
                    output_path=xlsx_output_path,
                    in_validation_index=in_validation_index,
                    source_path=csv_output_path,
                )
            )
            upload_csv_paths.append(str(csv_output_path.resolve()))
            upload_xlsx_paths.append(str(xlsx_output_path.resolve()))

        live_source_path = temp_dir / "raw_live" / "live_sample_validation.csv"
        live_meta = export_live_sample(
            base_dir=base_dir,
            labels_path=labels_path,
            output_path=live_source_path,
            forecast_config_path=forecast_config,
            house_id=args.house_id,
            window_days=args.live_window_days,
        )
        live_frame = _read_subset_frame(live_source_path, LIVE_COLUMNS)
        live_output_path = live_dir / "live_sample_validation.csv"
        _write_csv(live_frame, live_output_path)

    live_in_validation_index = str(live_meta["sample_id"]) in validation_sample_ids
    if not live_in_validation_index:
        raise ValueError(f"live 样本 {live_meta['sample_id']} 不在验证集索引中，终止导出")

    manifest_rows.append(
        _build_manifest_row(
            artifact_type="live",
                    format_name="csv",
                    sample_meta=live_meta,
                    output_path=live_output_path,
                    in_validation_index=live_in_validation_index,
                    source_path=None,
                )
    )

    run_output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_output_dir / "manifest.csv"
    summary_path = run_output_dir / "summary.json"
    manifest_frame = pd.DataFrame(manifest_rows)
    manifest_frame.to_csv(manifest_path, index=False)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "forecast_config_path": str(forecast_config),
        "validation_index_path": str(resolve_forecast_sample_index_path(forecast_config)),
        "validation_rows": int(len(validation_index)),
        "validation_households": int(validation_index["house_id"].nunique()),
        "house_id": str(exported_samples[0]["house_id"]),
        "upload_sample_count": len(upload_csv_paths),
        "upload_window_days": int(args.window_days),
        "live_window_days": int(args.live_window_days),
        "upload_csv_paths": upload_csv_paths,
        "upload_xlsx_paths": upload_xlsx_paths,
        "live_csv_path": str(live_output_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "all_artifacts_from_validation": bool(
            manifest_frame["in_validation_index"].astype(bool).all()
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    summary = run_export(args)
    print("已完成验证集上传模拟样本导出")
    print(f"输出目录: {Path(summary['manifest_path']).resolve().parent}")
    print(f"上传 CSV 数量: {summary['upload_sample_count']}")
    print(f"上传 XLSX 数量: {summary['upload_sample_count']}")
    print(f"live CSV: {summary['live_csv_path']}")
    print(f"清单文件: {summary['manifest_path']}")
    print(
        "验证集校验: "
        f"{'通过' if summary['all_artifacts_from_validation'] else '失败'}"
    )


if __name__ == "__main__":
    main()

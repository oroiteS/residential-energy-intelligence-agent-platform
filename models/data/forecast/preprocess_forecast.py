#!/usr/bin/env python3
"""预测任务数据预处理。

将居民日冻结宽表转换为两类预测任务数据：
1. 标准日级长表：一行表示一个用户某一天的用电情况。
2. 监督学习样本表：过去 30 天预测未来 7 天。
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev


TYPE_TOTAL = "1"
TYPE_PEAK = "2"
TYPE_VALLEY = "3"
WEEK_CYCLE = 7.0


@dataclass(frozen=True)
class DayRecord:
    user_id: str
    date: datetime
    energy: float
    peak_energy: float
    valley_energy: float

    @property
    def weekday(self) -> int:
        return self.date.weekday()

    @property
    def weekday_sin(self) -> float:
        return math.sin(2 * math.pi * self.weekday / WEEK_CYCLE)

    @property
    def weekday_cos(self) -> float:
        return math.cos(2 * math.pi * self.weekday / WEEK_CYCLE)

    @property
    def is_weekend(self) -> int:
        return 1 if self.weekday >= 5 else 0

    @property
    def month(self) -> int:
        return self.date.month

    @property
    def day_of_year(self) -> int:
        return int(self.date.strftime("%j"))

    @property
    def peak_ratio(self) -> float:
        return self.peak_energy / self.energy if self.energy > 1e-8 else math.nan


def normalize_id(raw_id: str) -> str:
    """去掉脱敏数据中的千分位逗号，保持用户标识可连接。"""
    return raw_id.replace(",", "").strip()


def parse_date(raw_date: str) -> datetime:
    return datetime.strptime(raw_date, "%Y/%m/%d")


def parse_float(raw_value: str, *, row_id: str, date: str) -> float:
    value = raw_value.strip()
    if value == "":
        raise ValueError(f"发现空电量值：用户={row_id}，日期={date}")
    return float(value)


def load_daily_records(source_path: Path) -> list[DayRecord]:
    """读取 1.2 宽表并转换为日级长表。"""
    user_type_values: dict[str, dict[str, list[str]]] = {}

    with source_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        date_columns = header[2:]

        for row in reader:
            user_id = normalize_id(row[0])
            energy_type = row[1].strip()
            user_type_values.setdefault(user_id, {})[energy_type] = row[2:]

    dates = [parse_date(column) for column in date_columns]
    records: list[DayRecord] = []

    for user_id, values_by_type in user_type_values.items():
        missing_types = {TYPE_TOTAL, TYPE_PEAK, TYPE_VALLEY} - set(values_by_type)
        if missing_types:
            raise ValueError(f"用户 {user_id} 缺少 type={sorted(missing_types)} 的记录")

        total_values = values_by_type[TYPE_TOTAL]
        peak_values = values_by_type[TYPE_PEAK]
        valley_values = values_by_type[TYPE_VALLEY]

        if not (len(total_values) == len(peak_values) == len(valley_values) == len(dates)):
            raise ValueError(f"用户 {user_id} 的日期列数量不一致")

        for index, date in enumerate(dates):
            date_text = date.strftime("%Y-%m-%d")
            energy = parse_float(total_values[index], row_id=user_id, date=date_text)
            peak_energy = parse_float(peak_values[index], row_id=user_id, date=date_text)
            valley_energy = parse_float(valley_values[index], row_id=user_id, date=date_text)

            if energy < 0 or peak_energy < 0 or valley_energy < 0:
                raise ValueError(f"发现负电量：用户={user_id}，日期={date_text}")

            records.append(
                DayRecord(
                    user_id=user_id,
                    date=date,
                    energy=energy,
                    peak_energy=peak_energy,
                    valley_energy=valley_energy,
                )
            )

    records.sort(key=lambda item: (item.user_id, item.date))
    return records


def write_daily_records(records: list[DayRecord], output_path: Path) -> None:
    fieldnames = [
        "user_id",
        "date",
        "energy",
        "peak_energy",
        "valley_energy",
        "weekday",
        "weekday_sin",
        "weekday_cos",
        "is_weekend",
        "month",
        "day_of_year",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "user_id": record.user_id,
                    "date": record.date.strftime("%Y-%m-%d"),
                    "energy": f"{record.energy:.6f}",
                    "peak_energy": f"{record.peak_energy:.6f}",
                    "valley_energy": f"{record.valley_energy:.6f}",
                    "weekday": record.weekday,
                    "weekday_sin": f"{record.weekday_sin:.6f}",
                    "weekday_cos": f"{record.weekday_cos:.6f}",
                    "is_weekend": record.is_weekend,
                    "month": record.month,
                    "day_of_year": record.day_of_year,
                }
            )


def build_sample_header(input_days: int, output_days: int) -> list[str]:
    fieldnames = [
        "user_id",
        "window_start_date",
        "window_end_date",
        "forecast_start_date",
        "forecast_end_date",
    ]

    for day in range(1, input_days + 1):
        fieldnames.extend(
            [
                f"x_energy_d{day:02d}",
                f"x_peak_energy_d{day:02d}",
                f"x_valley_energy_d{day:02d}",
                f"x_weekday_sin_d{day:02d}",
                f"x_weekday_cos_d{day:02d}",
                f"x_is_weekend_d{day:02d}",
            ]
        )

    for day in range(1, output_days + 1):
        fieldnames.extend(
            [
                f"future_weekday_sin_d{day:02d}",
                f"future_weekday_cos_d{day:02d}",
                f"future_is_weekend_d{day:02d}",
            ]
        )

    fieldnames.extend(
        [
            "hist_mean_7",
            "hist_mean_14",
            "hist_mean_30",
            "hist_std_7",
            "hist_std_30",
            "hist_min_30",
            "hist_max_30",
            "hist_last_value",
            "hist_trend_7",
            "hist_weekend_mean",
            "hist_workday_mean",
            "hist_weekend_workday_diff",
            "hist_peak_ratio",
            "hist_peak_mean_7",
            "hist_peak_mean_30",
            "hist_peak_std_7",
            "hist_peak_std_30",
            "hist_peak_min_30",
            "hist_peak_max_30",
            "hist_peak_last_value",
            "hist_peak_trend_7",
            "hist_valley_mean_7",
            "hist_valley_mean_30",
            "hist_valley_std_7",
            "hist_valley_std_30",
            "hist_valley_min_30",
            "hist_valley_max_30",
            "hist_valley_last_value",
            "hist_valley_trend_7",
        ]
    )

    for day in range(1, output_days + 1):
        fieldnames.append(f"y_energy_d{day:02d}")

    for day in range(1, output_days + 1):
        fieldnames.append(f"y_peak_d{day:02d}")

    for day in range(1, output_days + 1):
        fieldnames.append(f"y_valley_d{day:02d}")

    return fieldnames


def mean_or_nan(values: list[float]) -> float:
    return mean(values) if values else math.nan


def build_sample_row(
    user_id: str,
    history: list[DayRecord],
    future: list[DayRecord],
    input_days: int,
    output_days: int,
) -> dict[str, str | int]:
    history_energy = [record.energy for record in history]
    history_peak = [record.peak_energy for record in history]
    history_valley = [record.valley_energy for record in history]
    last_7 = history_energy[-7:]
    last_14 = history_energy[-14:]
    last_7_peak = history_peak[-7:]
    last_7_valley = history_valley[-7:]

    weekend_values = [record.energy for record in history if record.is_weekend]
    workday_values = [record.energy for record in history if not record.is_weekend]
    weekend_mean = mean_or_nan(weekend_values)
    workday_mean = mean_or_nan(workday_values)

    # 历史 30 天峰时占比的中位数，用于推导预测峰/谷（baseline）
    peak_ratios = sorted(
        r.peak_ratio for r in history if not math.isnan(r.peak_ratio)
    )
    if peak_ratios:
        mid = len(peak_ratios) // 2
        hist_peak_ratio = (
            peak_ratios[mid] if len(peak_ratios) % 2 == 1
            else (peak_ratios[mid - 1] + peak_ratios[mid]) / 2
        )
    else:
        hist_peak_ratio = math.nan

    row: dict[str, str | int] = {
        "user_id": user_id,
        "window_start_date": history[0].date.strftime("%Y-%m-%d"),
        "window_end_date": history[-1].date.strftime("%Y-%m-%d"),
        "forecast_start_date": future[0].date.strftime("%Y-%m-%d"),
        "forecast_end_date": future[-1].date.strftime("%Y-%m-%d"),
    }

    for index, record in enumerate(history, start=1):
        row[f"x_energy_d{index:02d}"] = f"{record.energy:.6f}"
        row[f"x_peak_energy_d{index:02d}"] = f"{record.peak_energy:.6f}"
        row[f"x_valley_energy_d{index:02d}"] = f"{record.valley_energy:.6f}"
        row[f"x_weekday_sin_d{index:02d}"] = f"{record.weekday_sin:.6f}"
        row[f"x_weekday_cos_d{index:02d}"] = f"{record.weekday_cos:.6f}"
        row[f"x_is_weekend_d{index:02d}"] = record.is_weekend

    for index, record in enumerate(future, start=1):
        row[f"future_weekday_sin_d{index:02d}"] = f"{record.weekday_sin:.6f}"
        row[f"future_weekday_cos_d{index:02d}"] = f"{record.weekday_cos:.6f}"
        row[f"future_is_weekend_d{index:02d}"] = record.is_weekend

    row.update(
        {
            "hist_mean_7": f"{mean(last_7):.6f}",
            "hist_mean_14": f"{mean(last_14):.6f}",
            "hist_mean_30": f"{mean(history_energy):.6f}",
            "hist_std_7": f"{pstdev(last_7):.6f}",
            "hist_std_30": f"{pstdev(history_energy):.6f}",
            "hist_min_30": f"{min(history_energy):.6f}",
            "hist_max_30": f"{max(history_energy):.6f}",
            "hist_last_value": f"{history_energy[-1]:.6f}",
            "hist_trend_7": f"{(last_7[-1] - last_7[0]) / 6:.6f}",
            "hist_weekend_mean": f"{weekend_mean:.6f}",
            "hist_workday_mean": f"{workday_mean:.6f}",
            "hist_weekend_workday_diff": f"{(weekend_mean - workday_mean):.6f}",
            "hist_peak_ratio": f"{hist_peak_ratio:.6f}",
            "hist_peak_mean_7": f"{mean(last_7_peak):.6f}",
            "hist_peak_mean_30": f"{mean(history_peak):.6f}",
            "hist_peak_std_7": f"{pstdev(last_7_peak):.6f}",
            "hist_peak_std_30": f"{pstdev(history_peak):.6f}",
            "hist_peak_min_30": f"{min(history_peak):.6f}",
            "hist_peak_max_30": f"{max(history_peak):.6f}",
            "hist_peak_last_value": f"{history_peak[-1]:.6f}",
            "hist_peak_trend_7": f"{(last_7_peak[-1] - last_7_peak[0]) / 6:.6f}",
            "hist_valley_mean_7": f"{mean(last_7_valley):.6f}",
            "hist_valley_mean_30": f"{mean(history_valley):.6f}",
            "hist_valley_std_7": f"{pstdev(last_7_valley):.6f}",
            "hist_valley_std_30": f"{pstdev(history_valley):.6f}",
            "hist_valley_min_30": f"{min(history_valley):.6f}",
            "hist_valley_max_30": f"{max(history_valley):.6f}",
            "hist_valley_last_value": f"{history_valley[-1]:.6f}",
            "hist_valley_trend_7": f"{(last_7_valley[-1] - last_7_valley[0]) / 6:.6f}",
        }
    )

    for index, record in enumerate(future, start=1):
        row[f"y_energy_d{index:02d}"] = f"{record.energy:.6f}"

    for index, record in enumerate(future, start=1):
        row[f"y_peak_d{index:02d}"] = f"{record.peak_energy:.6f}"

    for index, record in enumerate(future, start=1):
        row[f"y_valley_d{index:02d}"] = f"{record.valley_energy:.6f}"

    if len(history) != input_days or len(future) != output_days:
        raise ValueError("样本窗口长度异常")

    return row


def write_supervised_samples(
    records: list[DayRecord],
    output_path: Path,
    *,
    input_days: int,
    output_days: int,
) -> int:
    records_by_user: dict[str, list[DayRecord]] = {}
    for record in records:
        records_by_user.setdefault(record.user_id, []).append(record)

    fieldnames = build_sample_header(input_days, output_days)
    sample_count = 0

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for user_id, user_records in sorted(records_by_user.items()):
            max_start = len(user_records) - input_days - output_days + 1
            for start in range(max_start):
                history = user_records[start : start + input_days]
                future = user_records[start + input_days : start + input_days + output_days]
                writer.writerow(
                    build_sample_row(
                        user_id=user_id,
                        history=history,
                        future=future,
                        input_days=input_days,
                        output_days=output_days,
                    )
                )
                sample_count += 1

    return sample_count


def parse_args() -> argparse.Namespace:
    data_root = Path(__file__).resolve().parents[1]
    default_source = data_root / "1.2用电查询-用户日冻结-清洗前.csv"
    default_output_dir = Path(__file__).resolve().parent / "output"

    parser = argparse.ArgumentParser(description="生成居民用电预测任务数据集")
    parser.add_argument("--source", type=Path, default=default_source, help="1.2 原始宽表路径")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir, help="输出目录")
    parser.add_argument("--input-days", type=int, default=30, help="历史输入天数")
    parser.add_argument("--output-days", type=int, default=7, help="未来预测天数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_daily_records(args.source)

    daily_output_path = args.output_dir / "daily_energy_long.csv"
    sample_output_path = args.output_dir / f"forecast_samples_{args.input_days}_to_{args.output_days}.csv"

    write_daily_records(records, daily_output_path)
    sample_count = write_supervised_samples(
        records,
        sample_output_path,
        input_days=args.input_days,
        output_days=args.output_days,
    )

    user_count = len({record.user_id for record in records})
    print(f"日级长表已生成：{daily_output_path}")
    print(f"监督学习样本已生成：{sample_output_path}")
    print(f"用户数：{user_count}")
    print(f"日记录数：{len(records)}")
    print(f"样本数：{sample_count}")


if __name__ == "__main__":
    main()

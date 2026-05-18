#!/usr/bin/env python3
"""分类任务数据预处理。

从 1.2 宽表生成以 7 天为窗口的用户级行为特征表，作为聚类和分类的输入。
峰谷分段对应未来用户从 15 分钟粒度数据自行聚合的总量/峰/谷。
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev, median


WINDOW_DAYS = 7
TYPE_TOTAL = "1"
TYPE_PEAK = "2"
TYPE_VALLEY = "3"

# 上海居民峰谷时段（仅用于标注，实际峰谷电量已由 type 字段提供）
PEAK_MONTHS_SUMMER = {6, 7, 8, 9}   # 夏季尖峰月
PEAK_MONTHS_WINTER = {1, 12}         # 冬季尖峰月


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
    def is_weekend(self) -> bool:
        return self.weekday >= 5

    @property
    def month(self) -> int:
        return self.date.month


def normalize_id(raw_id: str) -> str:
    return raw_id.replace(",", "").strip()


def parse_date(raw_date: str) -> datetime:
    return datetime.strptime(raw_date, "%Y/%m/%d")


def load_daily_records(source_path: Path) -> dict[str, list[DayRecord]]:
    """读取 1.2 宽表，按用户分组返回日记录列表（已排序）。"""
    user_type_values: dict[str, dict[str, list[str]]] = {}
    date_columns: list[str] = []

    with source_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        date_columns = header[2:]
        for row in reader:
            uid = normalize_id(row[0])
            etype = row[1].strip()
            user_type_values.setdefault(uid, {})[etype] = row[2:]

    dates = [parse_date(col) for col in date_columns]
    records_by_user: dict[str, list[DayRecord]] = {}

    for uid, by_type in user_type_values.items():
        missing = {TYPE_TOTAL, TYPE_PEAK, TYPE_VALLEY} - set(by_type)
        if missing:
            continue
        totals = by_type[TYPE_TOTAL]
        peaks = by_type[TYPE_PEAK]
        valleys = by_type[TYPE_VALLEY]
        user_records: list[DayRecord] = []
        for i, date in enumerate(dates):
            try:
                e = float(totals[i])
                p = float(peaks[i])
                v = float(valleys[i])
            except (ValueError, IndexError):
                continue
            if e < 0 or p < 0 or v < 0:
                continue
            user_records.append(DayRecord(uid, date, e, p, v))
        user_records.sort(key=lambda r: r.date)
        records_by_user[uid] = user_records

    return records_by_user


def extract_window_features(window: list[DayRecord]) -> dict[str, float]:
    """从一个 7 天窗口提取行为特征。所有特征均可从用户上传的日电量明细计算。"""
    energies = [r.energy for r in window]
    peaks = [r.peak_energy for r in window]
    valleys = [r.valley_energy for r in window]

    avg_e = mean(energies)
    std_e = pstdev(energies)
    max_e = max(energies)
    min_e = min(energies)

    avg_peak = mean(peaks)
    avg_valley = mean(valleys)
    total_peak = sum(peaks)
    total_valley = sum(valleys)
    total_e = sum(energies)

    # 峰谷比（谷接近0时保护分母）
    peak_valley_ratio = total_peak / (total_valley + 1e-6)
    # 峰电占比
    peak_ratio = total_peak / (total_e + 1e-6)
    # 谷电占比
    valley_ratio = total_valley / (total_e + 1e-6)
    # 负荷均衡度（越接近1越均衡）
    load_factor = avg_e / (max_e + 1e-6)

    workday_e = [r.energy for r in window if not r.is_weekend]
    weekend_e = [r.energy for r in window if r.is_weekend]
    workday_avg = mean(workday_e) if workday_e else avg_e
    weekend_avg = mean(weekend_e) if weekend_e else avg_e
    # 周末相对工作日的比率
    weekend_workday_ratio = weekend_avg / (workday_avg + 1e-6)

    # 7 天用电趋势斜率（简单线性回归斜率 / 均值，归一化为相对变化）
    n = len(energies)
    x_mean = (n - 1) / 2
    numerator = sum((i - x_mean) * (energies[i] - avg_e) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    slope = numerator / (denominator + 1e-9)
    trend_rel = slope / (avg_e + 1e-6)

    # 逐日变化幅度（相邻天差值的标准差）
    day_diffs = [abs(energies[i + 1] - energies[i]) for i in range(n - 1)]
    volatility = mean(day_diffs) / (avg_e + 1e-6)

    # 中位数（抗异常值）
    med_e = median(energies)
    # 中位数与均值之比（右偏说明有极高日）
    med_mean_ratio = med_e / (avg_e + 1e-6)

    return {
        "avg_energy": avg_e,
        "std_energy": std_e,
        "max_energy": max_e,
        "min_energy": min_e,
        "avg_peak": avg_peak,
        "avg_valley": avg_valley,
        "peak_valley_ratio": peak_valley_ratio,
        "peak_ratio": peak_ratio,
        "valley_ratio": valley_ratio,
        "load_factor": load_factor,
        "workday_avg": workday_avg,
        "weekend_avg": weekend_avg,
        "weekend_workday_ratio": weekend_workday_ratio,
        "trend_rel": trend_rel,
        "volatility": volatility,
        "med_mean_ratio": med_mean_ratio,
    }


FEATURE_COLUMNS = [
    "avg_energy",
    "std_energy",
    "max_energy",
    "min_energy",
    "avg_peak",
    "avg_valley",
    "peak_valley_ratio",
    "peak_ratio",
    "valley_ratio",
    "load_factor",
    "workday_avg",
    "weekend_avg",
    "weekend_workday_ratio",
    "trend_rel",
    "volatility",
    "med_mean_ratio",
]


def build_window_samples(
    records_by_user: dict[str, list[DayRecord]],
    window_days: int,
) -> list[dict]:
    """滑动切分所有用户的时间序列，生成特征样本列表。"""
    samples = []
    for uid, records in sorted(records_by_user.items()):
        n = len(records)
        if n < window_days:
            continue
        for start in range(n - window_days + 1):
            window = records[start : start + window_days]
            feats = extract_window_features(window)
            row = {
                "user_id": uid,
                "window_start": window[0].date.strftime("%Y-%m-%d"),
                "window_end": window[-1].date.strftime("%Y-%m-%d"),
            }
            row.update(feats)
            samples.append(row)
    return samples


def write_samples(samples: list[dict], output_path: Path) -> None:
    fieldnames = ["user_id", "window_start", "window_end"] + FEATURE_COLUMNS
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in samples:
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in row.items()})


def parse_args() -> argparse.Namespace:
    data_root = Path(__file__).resolve().parents[1]
    default_source = data_root / "1.2用电查询-用户日冻结-清洗前.csv"
    default_output_dir = Path(__file__).resolve().parent / "output"

    parser = argparse.ArgumentParser(description="生成分类任务 7 天窗口特征表")
    parser.add_argument("--source", type=Path, default=default_source)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records_by_user = load_daily_records(args.source)
    samples = build_window_samples(records_by_user, args.window_days)

    output_path = args.output_dir / "window_features.csv"
    write_samples(samples, output_path)

    user_count = len(records_by_user)
    print(f"用户数：{user_count}")
    print(f"样本数：{len(samples)}")
    print(f"已写入：{output_path}")


if __name__ == "__main__":
    main()

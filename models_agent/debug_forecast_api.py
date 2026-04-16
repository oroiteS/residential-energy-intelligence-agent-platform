#!/usr/bin/env python3
"""直接调试 models_agent 的 forecast 接口。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any
from urllib import error, request


DAY_SLOTS = 96
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
PROFILE_LABELS = (
    "afternoon_peak",
    "all_day_low",
    "day_low_night_high",
    "morning_peak",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "使用本地 CSV 构造 7 天历史序列，"
            "直接调用 models_agent 的 /internal/model/v1/forecast 接口。"
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("../live/data/live_sample_validation.csv"),
        help="输入 CSV，默认使用 live 模拟数据",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8001",
        help="models_agent 服务地址",
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        default=1,
        help="请求中的 dataset_id",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=7,
        help="历史窗口天数，默认 7",
    )
    parser.add_argument(
        "--target-day-index",
        type=int,
        default=7,
        help="目标日索引，从 0 开始；默认 7 表示第 8 天",
    )
    parser.add_argument(
        "--with-profile-days",
        action="store_true",
        help="附带 profile_probability_days，一并测试带日型概率的推理链路",
    )
    parser.add_argument(
        "--save-payload",
        type=Path,
        default=Path("/tmp/models_agent_forecast_payload.json"),
        help="保存构造出的请求体，便于复查",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只构造 payload，不实际发送请求",
    )
    return parser.parse_args()


def load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 不存在: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required_fields = {
            "timestamp",
            "date",
            "aggregate",
            "active_appliance_count",
            "burst_event_count",
        }
        missing_fields = sorted(required_fields - fieldnames)
        if missing_fields:
            raise ValueError(f"CSV 缺少字段: {', '.join(missing_fields)}")
        return list(reader)


def group_rows_by_date(rows: list[dict[str, str]]) -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["date"]].append(row)

    dates = sorted(grouped)
    if not dates:
        raise ValueError("CSV 中没有可用日期")

    for date in dates:
        day_rows = grouped[date]
        day_rows.sort(key=lambda item: item["timestamp"])
        if len(day_rows) != DAY_SLOTS:
            raise ValueError(f"{date} 的点数不是 96，而是 {len(day_rows)}")
    return dates, grouped


def to_iso8601(timestamp_text: str) -> str:
    try:
        parsed = datetime.strptime(timestamp_text, TIMESTAMP_FORMAT).astimezone()
        return parsed.isoformat(timespec="seconds")
    except ValueError:
        return timestamp_text.replace(" ", "T")


def build_profile_probabilities(day_rows: list[dict[str, str]]) -> dict[str, float]:
    values = [float(row["aggregate"]) for row in day_rows]
    full_mean = mean(values)
    day_mean = mean(values[32:72])
    night_mean = mean(values[:32] + values[72:])
    morning_mean = mean(values[24:48])
    afternoon_mean = mean(values[52:72])

    winner = "afternoon_peak"
    if full_mean < 200:
        winner = "all_day_low"
    elif night_mean >= day_mean * 1.2:
        winner = "day_low_night_high"
    elif morning_mean >= afternoon_mean * 1.1:
        winner = "morning_peak"

    probabilities = {label: 0.03 for label in PROFILE_LABELS}
    probabilities[winner] = 0.91
    return probabilities


def build_payload(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = load_csv_rows(args.csv)
    dates, grouped = group_rows_by_date(rows)

    if args.target_day_index < 0:
        raise ValueError("--target-day-index 不能为负数")
    if args.history_days <= 0:
        raise ValueError("--history-days 必须大于 0")
    if args.target_day_index >= len(dates):
        raise ValueError(
            f"目标日索引超出范围: {args.target_day_index}，当前共有 {len(dates)} 天"
        )

    history_start_index = args.target_day_index - args.history_days
    if history_start_index < 0:
        raise ValueError(
            f"历史窗口不足：需要至少 {args.history_days} 天历史，"
            f"但目标日前只有 {args.target_day_index} 天"
        )

    history_dates = dates[history_start_index : args.target_day_index]
    target_date = dates[args.target_day_index]
    history_rows: list[dict[str, str]] = []
    for date in history_dates:
        history_rows.extend(grouped[date])
    target_rows = grouped[target_date]

    payload: dict[str, Any] = {
        "model_type": "tft",
        "dataset_id": args.dataset_id,
        "forecast_start": to_iso8601(target_rows[0]["timestamp"]),
        "forecast_end": to_iso8601(target_rows[-1]["timestamp"]),
        "granularity": "15min",
        "series": [
            {
                "timestamp": to_iso8601(row["timestamp"]),
                "aggregate": float(row["aggregate"]),
                "active_appliance_count": int(float(row["active_appliance_count"])),
                "burst_event_count": int(float(row["burst_event_count"])),
            }
            for row in history_rows
        ],
        "metadata": {
            "granularity": "15min",
            "unit": "w",
        },
    }

    if args.with_profile_days:
        payload["profile_probability_days"] = [
            {
                "date": date,
                "probabilities": build_profile_probabilities(grouped[date]),
            }
            for date in history_dates
        ]

    summary = {
        "csv": str(args.csv.resolve()),
        "history_dates": history_dates,
        "target_date": target_date,
        "history_points": len(history_rows),
        "target_points": len(target_rows),
        "with_profile_days": args.with_profile_days,
    }
    return payload, summary


def save_payload(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def post_json(base_url: str, payload: dict[str, Any]) -> tuple[int, str]:
    url = base_url.rstrip("/") + "/internal/model/v1/forecast"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=60) as response:
            return response.status, response.read().decode("utf-8")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, body
    except error.URLError as exc:
        raise RuntimeError(f"请求失败，无法连接到 {url}: {exc}") from exc


def pretty_print_json(text: str) -> str:
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return text


def main() -> int:
    args = parse_args()
    try:
        payload, summary = build_payload(args)
        save_payload(payload, args.save_payload)
    except Exception as exc:
        print(f"[构造失败] {exc}", file=sys.stderr)
        return 1

    print("[请求摘要]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[payload 已保存] {args.save_payload}")

    if args.dry_run:
        print("[dry-run] 未发送请求")
        return 0

    try:
        status_code, body = post_json(args.base_url, payload)
    except Exception as exc:
        print(f"[请求失败] {exc}", file=sys.stderr)
        return 2

    print(f"[HTTP {status_code}]")
    print(pretty_print_json(body))
    return 0 if status_code < 400 else 3


if __name__ == "__main__":
    raise SystemExit(main())

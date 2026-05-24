from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = DATA_DIR / "1.2用电查询-用户日冻结-清洗前.csv"
DEFAULT_OUTPUT = DATA_DIR / "output"


def parse_args() -> argparse.Namespace:
    """解析上传样例导出参数。"""

    parser = argparse.ArgumentParser(description="从日冻结数据生成标准上传样例")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="1.2 日冻结宽表路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出目录")
    parser.add_argument("--sample-count", type=int, default=5, help="导出用户数量")
    parser.add_argument("--days", type=int, default=35, help="每个样例覆盖天数")
    parser.add_argument("--start-date", default="2017/5/1", help="起始日期列")
    return parser.parse_args()


def build_daily_table(source: Path, start_date: str, days: int) -> pd.DataFrame:
    """从日冻结宽表中抽取一段连续日期的用户日级数据。

    输出仍然是日级数据，但列名拆成 date__total、date__peak、date__valley，
    方便后续把每日总量分解成 15 分钟负荷曲线。
    """

    raw = pd.read_csv(source, dtype={"id": str})
    date_cols = [col for col in raw.columns if col not in {"id", "type"}]
    if start_date not in date_cols:
        raise ValueError(f"起始日期列不存在：{start_date}")

    start_index = date_cols.index(start_date)
    selected_dates = date_cols[start_index : start_index + days]
    if len(selected_dates) < days:
        raise ValueError(f"从 {start_date} 开始不足 {days} 天")

    raw[selected_dates] = raw[selected_dates].apply(pd.to_numeric, errors="coerce")
    rows = []
    for user_id, group in raw.groupby("id", sort=False):
        # 每个用户需要同时具备总量、峰时、谷时三类日冻结记录。
        by_type = group.set_index("type")
        if not {1, 2, 3}.issubset(set(by_type.index)):
            continue
        values = {"id": user_id}
        valid = True
        for date_col in selected_dates:
            total = float(by_type.at[1, date_col])
            peak = float(by_type.at[2, date_col])
            valley = float(by_type.at[3, date_col])
            if not np.isfinite(total + peak + valley) or total <= 0:
                valid = False
                break
            values[f"{date_col}__total"] = total
            values[f"{date_col}__peak"] = max(peak, 0.0)
            values[f"{date_col}__valley"] = max(valley, 0.0)
        if valid:
            rows.append(values)
    return pd.DataFrame(rows)


def choose_users(daily: pd.DataFrame, sample_count: int, date_cols: list[str]) -> list[str]:
    """选择适合导出为上传样例的用户。

    这里过滤掉均值过低或过高的用户，再优先选择用电量和波动较高的样本，
    目的是让前端上传样例更容易展示分析、预测和检测效果。
    """

    total_cols = [f"{col}__total" for col in date_cols]
    scored = daily.assign(
        mean_kwh=daily[total_cols].mean(axis=1),
        std_kwh=daily[total_cols].std(axis=1),
    )
    scored = scored[(scored["mean_kwh"] > 1.0) & (scored["mean_kwh"] < 60.0)]
    scored = scored.sort_values(["mean_kwh", "std_kwh", "id"], ascending=[False, False, True])
    return scored["id"].head(sample_count).tolist()


def daily_kwh_to_15min_power(total_kwh: float, peak_kwh: float, valley_kwh: float, day_index: int) -> np.ndarray:
    """将日级峰谷电量近似展开为 15 分钟平均功率曲线。

    输出长度固定为 96，因为一天 24 小时、每小时 4 个 15 分钟槽位。
    这只是用于生成演示上传样例，不代表原始数据本身具有 15 分钟粒度。
    """

    # 15 分钟粒度下，一天共有 24 * 4 = 96 个时间槽。
    slots = np.arange(96)
    hours = slots / 4

    # 峰时段采用项目中的默认口径：07:00-11:00、18:00-23:00。
    # 非峰时段在样例中统一视为谷时段。
    peak_mask = ((hours >= 7) & (hours < 11)) | ((hours >= 18) & (hours < 23))
    valley_mask = ~peak_mask

    # 如果峰谷拆分缺失，则使用一个保守比例兜底；
    # 如果峰谷之和与总量不一致，则按总量重新缩放。
    split_sum = peak_kwh + valley_kwh
    if split_sum <= 0:
        peak_kwh = total_kwh * 0.42
        valley_kwh = total_kwh - peak_kwh
    else:
        scale = total_kwh / split_sum
        peak_kwh *= scale
        valley_kwh *= scale

    # 使用几个平滑峰形模拟居民早晚高峰、日间基础负荷和夜间负荷。
    # 这些形状只是为了生成更像真实负荷曲线的演示数据。
    morning = np.exp(-0.5 * ((hours - 7.5) / 1.4) ** 2)
    evening = np.exp(-0.5 * ((hours - 20.0) / 2.0) ** 2)
    daytime = np.exp(-0.5 * ((hours - 13.0) / 4.0) ** 2)
    night = np.exp(-0.5 * ((hours - 1.0) / 3.2) ** 2)
    weekend_boost = 1.12 if day_index % 7 in {5, 6} else 1.0

    peak_profile = np.where(peak_mask, 0.15 + 0.6 * morning + 1.0 * evening, 0.0)
    valley_profile = np.where(valley_mask, 0.45 + 0.18 * daytime + 0.25 * night, 0.0) * weekend_boost
    peak_profile = peak_profile / peak_profile.sum()
    valley_profile = valley_profile / valley_profile.sum()

    slot_kwh = peak_kwh * peak_profile + valley_kwh * valley_profile
    # 每个 15 分钟槽位电量 kWh 转为平均功率 W：kWh / 0.25h * 1000
    return slot_kwh / 0.25 * 1000


def build_upload_frame(user_row: pd.Series, date_cols: list[str]) -> pd.DataFrame:
    """构造单个用户的标准上传文件内容。

    输出列为 timestamp 和 aggregate_w，与后端导入接口要求一致。
    """

    frames = []
    for day_index, date_col in enumerate(date_cols):
        day = pd.to_datetime(date_col, format="%Y/%m/%d")
        power = daily_kwh_to_15min_power(
            total_kwh=float(user_row[f"{date_col}__total"]),
            peak_kwh=float(user_row[f"{date_col}__peak"]),
            valley_kwh=float(user_row[f"{date_col}__valley"]),
            day_index=day_index,
        )
        timestamps = pd.date_range(day, periods=96, freq="15min")
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
                    "aggregate_w": np.round(power).astype(int),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def safe_user_id(user_id: str) -> str:
    """把用户编号转换为安全的文件名片段。"""

    return "".join(ch for ch in user_id if ch.isdigit()) or "unknown"


def export_samples(source: Path, output: Path, sample_count: int, days: int, start_date: str) -> None:
    """导出多个标准上传样例文件。

    每个用户同时导出 CSV 和 XLSX，并生成 manifest 记录样例文件、行数和时间范围。
    """

    daily = build_daily_table(source, start_date, days)
    date_cols = [
        col.removesuffix("__total")
        for col in daily.columns
        if col.endswith("__total")
    ]
    user_ids = choose_users(daily, sample_count, date_cols)
    if len(user_ids) < sample_count:
        raise ValueError(f"可用用户不足：需要 {sample_count}，实际 {len(user_ids)}")

    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, user_id in enumerate(user_ids, start=1):
        # 每个用户独立生成一个上传样例，便于前端或答辩演示时逐个导入。
        row = daily[daily["id"] == user_id].iloc[0]
        frame = build_upload_frame(row, date_cols)
        stem = f"upload_sample_{index:02d}_user_{safe_user_id(user_id)}_{days}d"
        csv_path = output / f"{stem}.csv"
        xlsx_path = output / f"{stem}.xlsx"
        frame.to_csv(csv_path, index=False, encoding="utf-8")
        frame.to_excel(xlsx_path, index=False)
        manifest.append(
            {
                "user_id": user_id,
                "csv": csv_path.name,
                "xlsx": xlsx_path.name,
                "rows": len(frame),
                "time_start": frame["timestamp"].iloc[0],
                "time_end": frame["timestamp"].iloc[-1],
            }
        )

    pd.DataFrame(manifest).to_csv(output / "upload_samples_manifest.csv", index=False, encoding="utf-8")
    print(f"已导出 {len(user_ids)} 个用户样例到：{output}")


def main() -> None:
    args = parse_args()
    export_samples(
        source=args.source,
        output=args.output,
        sample_count=args.sample_count,
        days=args.days,
        start_date=args.start_date,
    )


if __name__ == "__main__":
    main()

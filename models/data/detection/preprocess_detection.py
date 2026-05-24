#!/usr/bin/env python3
"""异常检测任务数据预处理。

从 1.2 宽表生成以 7 天为窗口的用户级行为特征表，
作为 Isolation Forest、统计规则、分类漂移检测的输入。
与 classification 预处理解耦，各自维护独立的输出目录。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 复用 classification 预处理的核心函数，避免重复维护特征逻辑
# 异常检测和分类使用同一套 7 天窗口特征，便于后续把“行为类别”和“异常原因”放在同一解释口径下。
_CLS_DIR = Path(__file__).resolve().parents[1] / "classification"
if str(_CLS_DIR) not in sys.path:
    sys.path.insert(0, str(_CLS_DIR))

from preprocess_classification import (  # noqa: E402
    FEATURE_COLUMNS,
    build_window_samples,
    load_daily_records,
    write_samples,
)

WINDOW_DAYS = 7


def parse_args() -> argparse.Namespace:
    """解析异常检测预处理命令行参数。"""

    data_root = Path(__file__).resolve().parents[1]
    default_source = data_root / "1.2用电查询-用户日冻结-清洗前.csv"
    default_output_dir = Path(__file__).resolve().parent / "output"

    parser = argparse.ArgumentParser(description="生成异常检测任务 7 天窗口特征表")
    parser.add_argument("--source", type=Path, default=default_source)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    return parser.parse_args()


def main() -> None:
    """异常检测预处理脚本入口。

    输出 window_features.csv，后续供 Isolation Forest 和统计规则引擎读取。
    """

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

"""TFT 统一入口。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    script_dir = str(Path(__file__).resolve().parent)
    project_root = str(Path(__file__).resolve().parents[2])
    sys.path[:] = [path for path in sys.path if path != script_dir]
    sys.path.insert(0, project_root)

# 屏蔽 Triton autotune 详细日志，避免终端被内核搜索信息刷屏。
os.environ["TRITON_PRINT_AUTOTUNING"] = "0"

from forecast.tft.config import DEFAULT_CONFIG_PATH
from forecast.tft.test import main as test_main
from forecast.tft.train import main as train_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TFT 预测任务统一入口，可执行 train 或 test",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="训练 TFT 预测模型")
    train_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="配置文件路径，默认: %(default)s",
    )

    test_parser = subparsers.add_parser("test", help="测试 TFT 预测模型")
    test_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="配置文件路径，默认: %(default)s",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        train_main(config_path=args.config)
        return
    if args.command == "test":
        test_main(config_path=args.config)
        return
    raise ValueError(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()

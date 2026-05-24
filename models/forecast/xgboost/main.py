#!/usr/bin/env python3
"""XGBoost 预测模型命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .train import train
except ImportError:  # 兼容 `python forecast/xgboost/main.py`
    from train import train


def parse_args() -> argparse.Namespace:
    """解析 XGBoost baseline 训练参数。"""

    parser = argparse.ArgumentParser(description="训练 30 天预测 7 天的 XGBoost 模型")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "default.yaml",
        help="训练配置 YAML 路径",
    )
    parser.add_argument("--no-resume", action="store_true", help="忽略已有断点，从头训练")
    return parser.parse_args()


def main() -> None:
    """XGBoost baseline 命令行入口。"""

    args = parse_args()
    train(args.config, no_resume=args.no_resume)


if __name__ == "__main__":
    main()

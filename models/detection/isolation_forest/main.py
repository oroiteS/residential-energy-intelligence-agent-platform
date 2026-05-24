#!/usr/bin/env python3
"""Isolation Forest 异常检测命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .train import train
except ImportError:  # 兼容 `python detection/isolation_forest/main.py`
    from train import train


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    default_config = Path(__file__).resolve().parent / "config" / "default.yaml"
    parser = argparse.ArgumentParser(description="Isolation Forest 异常检测")
    parser.add_argument("--config", type=Path, default=default_config)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args.config)


if __name__ == "__main__":
    main()

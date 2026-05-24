#!/usr/bin/env python3
"""KMeans 居民用电行为聚类命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .train import train
except ImportError:  # 兼容 `python classification/kmeans/main.py`
    from train import train


def parse_args() -> argparse.Namespace:
    """解析 KMeans 训练入口参数。"""

    default_config = Path(__file__).resolve().parent / "config" / "default.yaml"
    parser = argparse.ArgumentParser(description="KMeans 居民用电行为聚类")
    parser.add_argument("--config", type=Path, default=default_config)
    return parser.parse_args()


def main() -> None:
    """KMeans 命令行入口。"""

    args = parse_args()
    train(args.config)


if __name__ == "__main__":
    main()

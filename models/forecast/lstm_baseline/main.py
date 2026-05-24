#!/usr/bin/env python3
"""LSTM 残差 baseline 任务命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .test import test
    from .train import train
except ImportError:  # 兼容 `python forecast/lstm_baseline/main.py`
    from test import test
    from train import train


def build_parser() -> argparse.ArgumentParser:
    """构建 LSTM 残差 baseline 的 train/test 子命令。"""

    parser = argparse.ArgumentParser(description="LSTM 残差 baseline 任务")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="训练 LSTM 残差预测模型")
    train_parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "default.yaml",
        help="训练配置 YAML 路径",
    )
    train_parser.add_argument("--no-resume", action="store_true", help="忽略 last.ckpt，从头训练")

    test_parser = subparsers.add_parser("test", help="测试 LSTM 残差预测模型")
    test_parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "default.yaml",
        help="测试配置 YAML 路径",
    )
    test_parser.add_argument("--checkpoint", type=Path, default=None, help="模型 checkpoint 路径，默认自动寻找 best")
    return parser


def parse_train_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 LSTM 残差预测模型")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "default.yaml",
        help="训练配置 YAML 路径",
    )
    parser.add_argument("--no-resume", action="store_true", help="忽略 last.ckpt，从头训练")
    return parser.parse_args()


def parse_test_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试 LSTM 残差预测模型")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "default.yaml",
        help="测试配置 YAML 路径",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="模型 checkpoint 路径，默认自动寻找 best")
    return parser.parse_args()


def train_main() -> None:
    args = parse_train_args()
    train(args.config, no_resume=args.no_resume)


def test_main() -> None:
    args = parse_test_args()
    test(args.config, checkpoint_path=args.checkpoint)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        train(args.config, no_resume=args.no_resume)
    elif args.command == "test":
        test(args.config, checkpoint_path=args.checkpoint)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

"""Decoder-only Transformer 预测统一入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forecast.GPT.config import DEFAULT_CONFIG_PATH
from forecast.GPT.predict import main as predict_main
from forecast.GPT.test import main as test_main
from forecast.GPT.train import main as train_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decoder-only Transformer 预测任务统一入口，可选择训练、测试或推理模式。",
        epilog=(
            "常用命令示例:\n"
            "  python forecast/GPT/main.py train\n"
            "  python forecast/GPT/main.py test\n"
            "  python forecast/GPT/main.py predict --input forecast/GPT/output/predict_input.json\n"
            "  python forecast/GPT/main.py train --config forecast/GPT/configs/config.yaml\n"
            "  python forecast/GPT/main.py predict --input data.csv --output forecast/GPT/output/prediction.csv"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True, help="可执行模式")

    train_parser = subparsers.add_parser(
        "train",
        help="训练 decoder-only Transformer 预测模型",
        description="读取 yaml 配置，训练 decoder-only Transformer 并保存最优权重。",
    )
    train_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="yaml 配置文件路径，默认: %(default)s",
    )

    test_parser = subparsers.add_parser(
        "test",
        help="测试 decoder-only Transformer 预测模型",
        description="读取 yaml 配置和 checkpoint，对测试集执行评估。",
    )
    test_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="yaml 配置文件路径，默认: %(default)s",
    )

    predict_parser = subparsers.add_parser(
        "predict",
        help="执行 decoder-only Transformer 预测推理",
        description="读取 checkpoint，对单样本 json 或批量 csv 执行多步预测。",
    )
    predict_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="yaml 配置文件路径，默认: %(default)s",
    )
    predict_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="推理输入路径，支持单样本 json 或批量 csv",
    )
    predict_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="推理结果输出路径；不传时使用配置中的默认输出目录",
    )
    return parser


def main() -> None:
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()
    if args.mode == "train":
        train_main(config_path=args.config)
        return
    if args.mode == "test":
        test_main(config_path=args.config)
        return
    if args.mode == "predict":
        predict_main(
            input_path=args.input,
            config_path=args.config,
            output_path=args.output,
        )
        return
    raise ValueError(f"未知模式: {args.mode}")


if __name__ == "__main__":
    main()


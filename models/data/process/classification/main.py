"""分类聚类统一入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data.process.classification.config import DEFAULT_CONFIG_PATH
from data.process.classification.fit import main as fit_main
from data.process.classification.relabel import main as relabel_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="分类聚类统一入口，支持 KMeans 聚类分析与人工映射打标。",
        epilog=(
            "常用命令示例:\n"
            "  python data/process/classification/main.py fit\n"
            "  python data/process/classification/main.py relabel\n"
            "  python data/process/classification/main.py fit --config data/process/classification/configs/config.yaml"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True, help="可执行模式")

    fit_parser = subparsers.add_parser(
        "fit",
        help="运行 KMeans 聚类并导出簇分析结果",
        description="读取 yaml 配置，执行 KMeans 聚类并导出平均曲线、代表样本和映射模板。",
    )
    fit_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="yaml 配置文件路径，默认: %(default)s",
    )

    relabel_parser = subparsers.add_parser(
        "relabel",
        help="根据人工映射结果重新生成标签文件",
        description="读取聚类分配结果和簇到标签映射文件，输出新的标签数据文件。",
    )
    relabel_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="yaml 配置文件路径，默认: %(default)s",
    )
    return parser


def main() -> None:
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()
    if args.mode == "fit":
        fit_main(config_path=args.config)
        return
    if args.mode == "relabel":
        relabel_main(config_path=args.config)
        return
    raise ValueError(f"未知模式: {args.mode}")


if __name__ == "__main__":
    main()

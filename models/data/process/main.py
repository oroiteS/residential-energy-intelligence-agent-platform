"""预处理命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.process.classification.builder import build_classification_features
from data.process.common.base import build_base_dataset
from data.process.forecast.builder import build_forecast_dataset
from data.process.testing import export_live_sample, export_representative_test_samples


def _parse_csv_set(raw_value: str | None) -> set[str] | None:
    if raw_value is None:
        return None
    values = {
        item.strip().lower()
        for item in raw_value.split(",")
        if item.strip()
    }
    return values or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="居民用电项目数据预处理工具",
        epilog=(
            "常用命令示例:\n"
            "  python main.py preprocess-base\n"
            "  python main.py build-classification\n"
            "  python main.py build-forecast\n"
            "  python main.py cluster-kmeans\n"
            "  python main.py relabel-kmeans\n"
            "\n"
            "分类标签正式流程:\n"
            "  1. python main.py build-classification\n"
            "  2. python main.py cluster-kmeans\n"
            "  3. 人工修改 cluster_label_mapping.yaml\n"
            "  4. python main.py relabel-kmeans\n"
            "\n"
            "查看某个命令的详细帮助:\n"
            "  python main.py preprocess-base -h"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="可执行的预处理子命令"
    )

    base_parser = subparsers.add_parser(
        "preprocess-base",
        help="生成15分钟基础时序数据",
        description="从 raw 目录下支持的多数据源生成共享的 15 分钟基础时序数据。",
    )
    base_parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="原始数据根目录或某个数据源目录，默认: %(default)s",
    )
    base_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/base_15min"),
        help="基础时序输出目录，默认: %(default)s",
    )
    base_parser.add_argument(
        "--sources",
        type=str,
        default="refit,ukdale,slovakia,opensynth",
        help="要处理的数据源，逗号分隔；可选: refit,ukdale,slovakia,opensynth",
    )
    base_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="若基础时序文件已存在则跳过，可用于断点续跑",
    )

    classification_parser = subparsers.add_parser(
        "build-classification",
        help="生成分类任务日级特征",
        description=(
            "从基础 15 分钟时序生成日级分类特征文件。"
            "此阶段不再生成规则标签，正式标签统一由 KMeans 聚类与人工映射产生。"
            "分类特征固定为 96x5：aggregate、slot_sin、slot_cos、weekday_sin、weekday_cos。"
        ),
    )
    classification_parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/processed/base_15min"),
        help="基础时序目录，默认: %(default)s",
    )
    classification_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/classification"),
        help="分类数据输出目录，默认: %(default)s",
    )

    forecast_parser = subparsers.add_parser(
        "build-forecast",
        help="生成预测任务数据",
        description="从基础 15 分钟时序生成 7 天输入到 1 天输出的预测样本。",
    )
    forecast_parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/processed/base_15min"),
        help="基础时序目录，默认: %(default)s",
    )
    forecast_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/forecast"),
        help="预测数据输出目录，默认: %(default)s",
    )

    kmeans_fit_parser = subparsers.add_parser(
        "cluster-kmeans",
        help="对分类日曲线执行 KMeans 聚类分析",
        description=(
            "读取分类日曲线文件，执行 KMeans 聚类，并导出每簇平均曲线、"
            "样本分配、代表样本与人工打标映射模板。"
        ),
    )
    kmeans_fit_parser.add_argument(
        "--config",
        type=Path,
        default=Path("data/process/classification/configs/config.yaml"),
        help="KMeans 聚类配置文件路径，默认: %(default)s",
    )

    kmeans_relabel_parser = subparsers.add_parser(
        "relabel-kmeans",
        help="根据 KMeans 人工映射结果重新生成标签文件",
        description=(
            "读取 KMeans 聚类分配结果和簇到标签映射文件，"
            "生成新的标签文件、完整带标签特征文件，以及正式分类训练文件。"
        ),
    )
    kmeans_relabel_parser.add_argument(
        "--config",
        type=Path,
        default=Path("data/process/classification/configs/config.yaml"),
        help="KMeans 聚类配置文件路径，默认: %(default)s",
    )

    export_parser = subparsers.add_parser(
        "export-testing-samples",
        help="从预测验证集导出前后端联调用的代表性测试样本",
        description=(
            "复用预测任务配置中的 train/val/test 划分，"
            "仅从验证集里为某个家庭导出约 5 份代表不同用电场景的测试 CSV。"
        ),
    )
    export_parser.add_argument(
        "--forecast-config",
        type=Path,
        default=Path("forecast/LSTM/configs/config.yaml"),
        help="预测模型配置文件路径，复用其中的数据切分参数，默认: %(default)s",
    )
    export_parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/processed/base_15min"),
        help="基础时序目录，默认: %(default)s",
    )
    export_parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path("data/processed/classification/classification_day_labels.csv"),
        help="分类标签文件路径，默认: %(default)s",
    )
    export_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/testing"),
        help="testing 样本输出目录，默认: %(default)s",
    )
    export_parser.add_argument(
        "--house-id",
        type=str,
        default=None,
        help="目标家庭编号，例如 refit_1、slovakia_1 或 house_refit_1；默认自动挑选标签最完整的家庭",
    )
    export_parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="导出样本数量，默认: %(default)s",
    )
    export_parser.add_argument(
        "--window-days",
        type=int,
        default=21,
        help="每个样本保留的连续天数窗口，支持超过两周，默认: %(default)s",
    )

    live_parser = subparsers.add_parser(
        "export-live-sample",
        help="导出 live 模块使用的独立循环连续窗口样本",
        description=(
            "从预测验证集里挑选一段连续窗口样本，"
            "默认直接输出 21 天数据供 live 模块独立运行。"
        ),
    )
    live_parser.add_argument(
        "--forecast-config",
        type=Path,
        default=Path("forecast/transformer_encoder_direct/configs/config.yaml"),
        help="预测模型配置文件路径，默认: %(default)s",
    )
    live_parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/processed/base_15min"),
        help="基础时序目录，默认: %(default)s",
    )
    live_parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path("data/processed/classification/classification_day_labels.csv"),
        help="分类标签文件路径，默认: %(default)s",
    )
    live_parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("../live/data/live_sample.csv"),
        help="live 连续窗口样本输出路径，默认: %(default)s",
    )
    live_parser.add_argument(
        "--house-id",
        type=str,
        default=None,
        help="目标家庭编号；默认自动挑选标签最完整的验证集家庭",
    )
    live_parser.add_argument(
        "--window-days",
        type=int,
        default=21,
        help="live 样本保留的连续天数窗口，支持超过两周，默认: %(default)s",
    )

    return parser


def main() -> None:
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()

    if args.command == "preprocess-base":
        summary_df = build_base_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            allowed_sources=_parse_csv_set(args.sources),
            skip_existing=args.skip_existing,
        )
        print(f"已生成基础时序数据，家庭数量: {len(summary_df)}")
        return

    if args.command == "build-classification":
        features_df = build_classification_features(
            base_dir=args.base_dir, output_dir=args.output_dir
        )
        print(
            "已生成分类日级特征，"
            f"样本数: {len(features_df)}，"
            "特征维度: 96x5"
        )
        return

    if args.command == "build-forecast":
        forecast_df = build_forecast_dataset(
            base_dir=args.base_dir, output_dir=args.output_dir
        )
        print(f"已生成预测数据，样本数: {len(forecast_df)}")
        return

    if args.command == "cluster-kmeans":
        from data.process.classification.fit import run_fit as run_kmeans_fit

        result = run_kmeans_fit(config_path=args.config)
        print(
            "已完成 KMeans 聚类分析，"
            f"平均曲线文件: {result['profiles_path']}，"
            f"映射模板: {result['mapping_template']}"
        )
        return

    if args.command == "relabel-kmeans":
        from data.process.classification.relabel import run_relabel as run_kmeans_relabel

        result = run_kmeans_relabel(config_path=args.config)
        print(
            "已完成 KMeans 人工映射打标，"
            f"特征文件: {result['labeled_features_path']}，"
            f"标签文件: {result['labeled_labels_path']}，"
            f"正式训练文件: {result['canonical_labeled_path']}"
        )
        return

    if args.command == "export-testing-samples":
        exported_samples = export_representative_test_samples(
            base_dir=args.base_dir,
            labels_path=args.labels_path,
            output_dir=args.output_dir,
            forecast_config_path=args.forecast_config,
            house_id=args.house_id,
            count=args.count,
            window_days=args.window_days,
        )
        if not exported_samples:
            print("未导出任何测试样本")
            return
        print(
            "已从验证集导出代表性测试样本，"
            f"家庭: {exported_samples[0]['house_id']}，"
            f"样本数: {len(exported_samples)}，"
            f"输出目录: {exported_samples[0]['output_dir']}"
        )
        return

    if args.command == "export-live-sample":
        result = export_live_sample(
            base_dir=args.base_dir,
            labels_path=args.labels_path,
            output_path=args.output_path,
            forecast_config_path=args.forecast_config,
            house_id=args.house_id,
            window_days=args.window_days,
        )
        print(
            "已导出 live 连续窗口样本，"
            f"家庭: {result['house_id']}，"
            f"场景: {result['scenario_name']}，"
            f"窗口天数: {result['window_days']}，"
            f"行数: {result['row_count']}，"
            f"输出文件: {result['output_path']}"
        )
        return

    raise ValueError(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()

"""XGBoost 分类任务配置加载。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from common.config_validation import validate_config_schema


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "config.yaml"
TOP_LEVEL_KEYS = {"data", "model", "train", "test", "predict"}
SECTION_KEYS = {
    "data": {"data_path", "train_ratio", "val_ratio"},
    "model": {
        "num_classes",
        "num_boost_round",
        "learning_rate",
        "max_depth",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "gamma",
        "reg_alpha",
        "reg_lambda",
        "max_bin",
        "tree_method",
        "nthread",
    },
    "train": {
        "output_dir",
        "seed",
        "eval_metric",
        "early_stopping_rounds",
        "early_stopping_min_delta",
        "verbose_eval",
    },
    "test": {"checkpoint_path", "output_dir"},
    "predict": {"checkpoint_path", "output_dir"},
}


@dataclass(slots=True)
class DataConfig:
    data_path: Path
    train_ratio: float
    val_ratio: float


@dataclass(slots=True)
class ModelConfig:
    num_classes: int
    num_boost_round: int
    learning_rate: float
    max_depth: int
    subsample: float
    colsample_bytree: float
    min_child_weight: float
    gamma: float
    reg_alpha: float
    reg_lambda: float
    max_bin: int
    tree_method: str
    nthread: int


@dataclass(slots=True)
class TrainConfig:
    output_dir: Path
    seed: int
    eval_metric: str
    early_stopping_rounds: int
    early_stopping_min_delta: float
    verbose_eval: int | bool


@dataclass(slots=True)
class TestConfig:
    checkpoint_path: Path | None
    output_dir: Path


@dataclass(slots=True)
class PredictConfig:
    checkpoint_path: Path | None
    output_dir: Path


@dataclass(slots=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    test: TestConfig
    predict: PredictConfig

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["data"]["data_path"] = str(self.data.data_path)
        result["train"]["output_dir"] = str(self.train.output_dir)
        result["test"]["output_dir"] = str(self.test.output_dir)
        result["predict"]["output_dir"] = str(self.predict.output_dir)
        result["test"]["checkpoint_path"] = (
            str(self.test.checkpoint_path) if self.test.checkpoint_path is not None else None
        )
        result["predict"]["checkpoint_path"] = (
            str(self.predict.checkpoint_path) if self.predict.checkpoint_path is not None else None
        )
        return result


def _resolve_path(path_value: str | None, base_dir: Path) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_experiment_config(config_path: Path = DEFAULT_CONFIG_PATH) -> ExperimentConfig:
    config_path = config_path.resolve()
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_dir = Path(__file__).resolve().parents[2]
    sections = validate_config_schema(
        raw_config,
        config_path=config_path,
        allowed_top_level_keys=TOP_LEVEL_KEYS,
        allowed_section_keys=SECTION_KEYS,
    )
    data_raw = sections["data"]
    model_raw = sections["model"]
    train_raw = sections["train"]
    test_raw = sections["test"]
    predict_raw = sections["predict"]

    data_config = DataConfig(
        data_path=_resolve_path(data_raw["data_path"], base_dir),
        train_ratio=float(data_raw.get("train_ratio", 0.7)),
        val_ratio=float(data_raw.get("val_ratio", 0.15)),
    )
    model_config = ModelConfig(
        num_classes=int(model_raw.get("num_classes", 0)),
        num_boost_round=int(model_raw.get("num_boost_round", 500)),
        learning_rate=float(model_raw.get("learning_rate", 0.05)),
        max_depth=int(model_raw.get("max_depth", 6)),
        subsample=float(model_raw.get("subsample", 0.8)),
        colsample_bytree=float(model_raw.get("colsample_bytree", 0.8)),
        min_child_weight=float(model_raw.get("min_child_weight", 2.0)),
        gamma=float(model_raw.get("gamma", 0.0)),
        reg_alpha=float(model_raw.get("reg_alpha", 0.0)),
        reg_lambda=float(model_raw.get("reg_lambda", 1.0)),
        max_bin=int(model_raw.get("max_bin", 256)),
        tree_method=str(model_raw.get("tree_method", "hist")),
        nthread=int(model_raw.get("nthread", 0)),
    )
    train_config = TrainConfig(
        output_dir=_resolve_path(train_raw["output_dir"], base_dir),
        seed=int(train_raw.get("seed", 42)),
        eval_metric=str(train_raw.get("eval_metric", "mlogloss")),
        early_stopping_rounds=int(train_raw.get("early_stopping_rounds", 30)),
        early_stopping_min_delta=float(train_raw.get("early_stopping_min_delta", 1e-4)),
        verbose_eval=(
            bool(train_raw.get("verbose_eval"))
            if isinstance(train_raw.get("verbose_eval"), bool)
            else int(train_raw.get("verbose_eval", 20))
        ),
    )
    test_checkpoint = test_raw.get("checkpoint_path")
    test_config = TestConfig(
        checkpoint_path=_resolve_path(test_checkpoint, base_dir) if test_checkpoint else None,
        output_dir=_resolve_path(test_raw["output_dir"], base_dir),
    )
    predict_checkpoint = predict_raw.get("checkpoint_path")
    predict_config = PredictConfig(
        checkpoint_path=_resolve_path(predict_checkpoint, base_dir) if predict_checkpoint else None,
        output_dir=_resolve_path(predict_raw["output_dir"], base_dir),
    )

    if data_config.train_ratio + data_config.val_ratio >= 1:
        raise ValueError("配置错误：train_ratio + val_ratio 必须小于 1")
    if model_config.num_classes < 0:
        raise ValueError("配置错误：model.num_classes 不能小于 0")
    if train_config.early_stopping_rounds <= 0:
        raise ValueError("配置错误：train.early_stopping_rounds 必须大于 0")
    if train_config.early_stopping_min_delta < 0:
        raise ValueError("配置错误：train.early_stopping_min_delta 不能小于 0")
    return ExperimentConfig(
        data=data_config,
        model=model_config,
        train=train_config,
        test=test_config,
        predict=predict_config,
    )

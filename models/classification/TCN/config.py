"""TCN 分类任务配置加载。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from common.config_validation import validate_config_schema
from common.device import detect_device
from classification.TCN.constants import INPUT_CHANNELS


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "config.yaml"
TOP_LEVEL_KEYS = {"data", "model", "train", "test", "predict"}
SECTION_KEYS = {
    "data": {"data_path", "train_ratio", "val_ratio"},
    "model": {"input_channels", "num_classes", "channel_sizes", "kernel_size", "dropout"},
    "train": {
        "output_dir",
        "batch_size",
        "epochs",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "seed",
        "scheduler",
        "label_smoothing",
        "scheduler_t0",
        "scheduler_t_mult",
        "scheduler_eta_min",
        "early_stopping_patience",
        "early_stopping_min_delta",
    },
    "test": {"checkpoint_path", "output_dir", "batch_size"},
    "predict": {"checkpoint_path", "output_dir", "batch_size"},
}


@dataclass(slots=True)
class DataConfig:
    data_path: Path
    train_ratio: float
    val_ratio: float


@dataclass(slots=True)
class ModelConfig:
    input_channels: int
    num_classes: int
    channel_sizes: list[int]
    kernel_size: int
    dropout: float


@dataclass(slots=True)
class TrainConfig:
    output_dir: Path
    batch_size: int
    epochs: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    seed: int
    scheduler: str
    label_smoothing: float
    scheduler_t0: int
    scheduler_t_mult: int
    scheduler_eta_min: float
    early_stopping_patience: int
    early_stopping_min_delta: float


@dataclass(slots=True)
class TestConfig:
    checkpoint_path: Path | None
    output_dir: Path
    batch_size: int


@dataclass(slots=True)
class PredictConfig:
    checkpoint_path: Path | None
    output_dir: Path
    batch_size: int


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
        input_channels=int(model_raw.get("input_channels", INPUT_CHANNELS)),
        num_classes=int(model_raw.get("num_classes", 4)),
        channel_sizes=[int(value) for value in model_raw.get("channel_sizes", [32, 64, 128])],
        kernel_size=int(model_raw.get("kernel_size", 3)),
        dropout=float(model_raw.get("dropout", 0.2)),
    )
    train_config = TrainConfig(
        output_dir=_resolve_path(train_raw["output_dir"], base_dir),
        batch_size=int(train_raw.get("batch_size", 128)),
        epochs=int(train_raw.get("epochs", 20)),
        optimizer=str(train_raw.get("optimizer", "adam")),
        learning_rate=float(train_raw.get("learning_rate", 1e-3)),
        weight_decay=float(train_raw.get("weight_decay", 1e-4)),
        seed=int(train_raw.get("seed", 42)),
        scheduler=str(train_raw.get("scheduler", "none")),
        label_smoothing=float(train_raw.get("label_smoothing", 0.0)),
        scheduler_t0=int(train_raw.get("scheduler_t0", 10)),
        scheduler_t_mult=int(train_raw.get("scheduler_t_mult", 2)),
        scheduler_eta_min=float(train_raw.get("scheduler_eta_min", 1e-6)),
        early_stopping_patience=int(train_raw.get("early_stopping_patience", 20)),
        early_stopping_min_delta=float(train_raw.get("early_stopping_min_delta", 0.0)),
    )
    test_checkpoint = test_raw.get("checkpoint_path")
    test_config = TestConfig(
        checkpoint_path=_resolve_path(test_checkpoint, base_dir) if test_checkpoint else None,
        output_dir=_resolve_path(test_raw["output_dir"], base_dir),
        batch_size=int(test_raw.get("batch_size", train_config.batch_size)),
    )
    predict_checkpoint = predict_raw.get("checkpoint_path")
    predict_config = PredictConfig(
        checkpoint_path=_resolve_path(predict_checkpoint, base_dir) if predict_checkpoint else None,
        output_dir=_resolve_path(predict_raw["output_dir"], base_dir),
        batch_size=int(predict_raw.get("batch_size", test_config.batch_size)),
    )

    if data_config.train_ratio + data_config.val_ratio >= 1:
        raise ValueError("配置错误：train_ratio + val_ratio 必须小于 1")
    if model_config.input_channels != INPUT_CHANNELS:
        raise ValueError(f"配置错误：model.input_channels 必须固定为 {INPUT_CHANNELS}")
    return ExperimentConfig(
        data=data_config,
        model=model_config,
        train=train_config,
        test=test_config,
        predict=predict_config,
    )

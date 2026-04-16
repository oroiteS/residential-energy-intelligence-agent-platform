"""TFT 训练/测试运行时配置。"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from forecast.tft.config import DataConfig, TrainConfig


def resolve_runtime_settings(train_config: TrainConfig) -> dict[str, Any]:
    """按当前硬件解析训练运行时设置。"""

    cuda_available = torch.cuda.is_available()
    accelerator = str(train_config.accelerator).lower()
    use_cuda = cuda_available and accelerator in {"auto", "gpu", "cuda"}

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(train_config.matmul_precision)

    precision = train_config.precision
    if not use_cuda and precision in {"bf16-mixed", "16-mixed"}:
        precision = "32-true"

    benchmark = bool(train_config.benchmark and use_cuda and not train_config.deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = benchmark

    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = bool(train_config.enable_tf32)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = bool(train_config.enable_tf32)
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)

    # 当前 torch + bf16 mixed 下，compile 容易因 dtype 变化触发重复重编译。
    # 因此仅在纯 32 位精度时允许启用 compile。
    compile_model = bool(
        train_config.compile_model
        and use_cuda
        and hasattr(torch, "compile")
        and precision == "32-true"
    )
    device_name = None
    total_memory_gb = None
    if use_cuda:
        properties = torch.cuda.get_device_properties(0)
        device_name = properties.name
        total_memory_gb = round(properties.total_memory / (1024**3), 2)

    return {
        "use_cuda": use_cuda,
        "precision": precision,
        "benchmark": benchmark,
        "compile_model": compile_model,
        "compile_mode": train_config.compile_mode,
        "device_name": device_name,
        "total_memory_gb": total_memory_gb,
    }


def maybe_compile_model(model: nn.Module, runtime_settings: dict[str, Any]) -> nn.Module:
    """仅在 CUDA 场景下按配置启用 torch.compile。"""

    if not runtime_settings["compile_model"]:
        return model
    # 仅编译 forward，避免 OptimizedModule 改写 state_dict key，
    # 从而影响 Lightning checkpoint 的保存和加载。
    model.forward = torch.compile(  # type: ignore[method-assign]
        model.forward,
        mode=str(runtime_settings["compile_mode"]),
    )
    return model

def collect_runtime_summary(
    data_config: DataConfig,
    train_config: TrainConfig,
    runtime_settings: dict[str, Any],
) -> dict[str, Any]:
    """汇总训练硬件与性能相关配置，方便写入输出目录。"""

    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": runtime_settings["device_name"],
        "cuda_total_memory_gb": runtime_settings["total_memory_gb"],
        "accelerator": train_config.accelerator,
        "devices": train_config.devices,
        "precision": runtime_settings["precision"],
        "deterministic": train_config.deterministic,
        "benchmark": runtime_settings["benchmark"],
        "enable_tf32": train_config.enable_tf32,
        "matmul_precision": train_config.matmul_precision,
        "compile_model": runtime_settings["compile_model"],
        "compile_mode": runtime_settings["compile_mode"],
        "batch_size": data_config.batch_size,
        "num_workers": data_config.num_workers,
        "prefetch_factor": data_config.prefetch_factor,
        "pin_memory": data_config.pin_memory,
    }

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GpuInfo:
    id: int
    name: str
    total_memory_gb: float


def get_compatible_cuda_gpus() -> list[GpuInfo]:
    if not torch.cuda.is_available():
        return []

    gpus = []
    for gpu_id in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(gpu_id)
            gpus.append(
                GpuInfo(
                    id=int(gpu_id),
                    name=str(torch.cuda.get_device_name(gpu_id)),
                    total_memory_gb=float(props.total_memory) / float(1024**3),
                )
            )
        except Exception:
            continue

    return gpus


def format_gpu_label(gpu: GpuInfo) -> str:
    return f"GPU {gpu.id} - {gpu.name} ({gpu.total_memory_gb:.1f} GB)"


def get_default_preprocess_device() -> str:
    gpus = get_compatible_cuda_gpus()
    if gpus:
        return f"cuda:{gpus[0].id}"
    return "cpu"


def get_default_training_gpu_ids() -> list[int]:
    gpus = get_compatible_cuda_gpus()
    if gpus:
        return [int(gpus[0].id)]
    return []


def normalize_training_gpu_ids(value) -> list[int]:
    available_ids = {gpu.id for gpu in get_compatible_cuda_gpus()}

    if isinstance(value, list):
        parsed = []
        for item in value:
            try:
                parsed.append(int(item))
            except Exception:
                continue
    elif isinstance(value, tuple):
        parsed = []
        for item in value:
            try:
                parsed.append(int(item))
            except Exception:
                continue
    elif isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        parsed = []
        for part in parts:
            try:
                parsed.append(int(part))
            except Exception:
                continue
    else:
        parsed = []

    normalized = []
    seen = set()

    for item in parsed:
        if item in available_ids and item not in seen:
            normalized.append(int(item))
            seen.add(int(item))

    return normalized


def normalize_preprocess_device(value) -> str:
    gpus = get_compatible_cuda_gpus()
    available = {f"cuda:{gpu.id}" for gpu in gpus}

    if isinstance(value, str):
        text = value.strip().lower()
        if text == "cpu":
            return "cpu"
        if text in available:
            return text

    if gpus:
        return f"cuda:{gpus[0].id}"

    return "cpu"
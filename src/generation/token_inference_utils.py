import random

import numpy as np
import torch


MAX_SEED = 2**32 - 1


def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_seed(seed: str | int) -> int:
    try:
        seed_value = int(seed)
    except Exception:
        seed_value = random.randint(0, MAX_SEED)
    seed_value = seed_value % (MAX_SEED + 1)
    return int(seed_value)


def set_generation_seed(seed: str | int) -> int:
    seed_value = normalize_seed(seed)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    return seed_value


def format_time_string(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    minutes = total // 60
    secs = total % 60
    return f"{minutes:02d}:{secs:02d}"


def normalize_waveform_tensor(waveform) -> torch.Tensor:
    if isinstance(waveform, torch.Tensor):
        tensor = waveform.detach().cpu().float()
    else:
        tensor = torch.as_tensor(waveform, dtype=torch.float32)

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2:
        pass
    else:
        tensor = tensor.reshape(1, -1)

    if tensor.shape[0] > 1:
        tensor = tensor.mean(dim=0, keepdim=True)

    return tensor.contiguous()
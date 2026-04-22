from datetime import datetime
import gc
import json
import math
from pathlib import Path

import pandas as pd
import torch
from torch.optim.lr_scheduler import LambdaLR

from src.core.device_utils import normalize_training_gpu_ids


def release_torch_memory():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def resolve_training_setup(training_gpu_ids) -> tuple[torch.device, list[int]]:
    normalized_gpu_ids = normalize_training_gpu_ids(training_gpu_ids)

    if normalized_gpu_ids:
        return torch.device(f"cuda:{normalized_gpu_ids[0]}"), normalized_gpu_ids

    return torch.device("cpu"), []


def load_codec_info(codec_info_path: str | Path) -> dict:
    codec_info_path = Path(codec_info_path)
    with codec_info_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_sequence_lengths(token_manifest_path: str | Path) -> tuple[dict, pd.DataFrame]:
    token_manifest_path = Path(token_manifest_path)
    manifest_df = pd.read_csv(token_manifest_path)

    if manifest_df.empty:
        raise ValueError("Token manifest is empty.")

    if "token_length" not in manifest_df.columns or manifest_df["token_length"].empty:
        raise ValueError("Token manifest is missing token lengths.")

    if "chunk_length_sec" in manifest_df.columns and not manifest_df["chunk_length_sec"].empty:
        chunk_length_sec = float(manifest_df["chunk_length_sec"].iloc[0])
    else:
        chunk_length_sec = 8.0

    token_lengths = manifest_df["token_length"].astype(int)
    max_observed_token_length = int(token_lengths.max())
    median_token_length = int(token_lengths.median())

    if max_observed_token_length <= 0:
        raise ValueError("Token manifest contains invalid token lengths.")

    effective_target_token_length = int(max_observed_token_length)
    derived_max_token_length = int(effective_target_token_length + 1)
    effective_clip_length_sec = float(chunk_length_sec)

    return {
        "chunk_length_sec": float(chunk_length_sec),
        "median_token_length": int(median_token_length),
        "max_observed_token_length": int(max_observed_token_length),
        "effective_target_token_length": int(effective_target_token_length),
        "derived_max_token_length": int(derived_max_token_length),
        "effective_clip_length_sec": float(effective_clip_length_sec),
    }, manifest_df


def count_model_parameters(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _format_parameter_count_short(parameter_count: int) -> str:
    value = int(parameter_count)

    if value >= 1_000_000_000:
        scaled = value / 1_000_000_000
        suffix = "b"
    elif value >= 1_000_000:
        scaled = value / 1_000_000
        suffix = "m"
    elif value >= 1_000:
        scaled = value / 1_000
        suffix = "k"
    else:
        return str(value)

    rounded_one_decimal = round(scaled, 1)
    if float(rounded_one_decimal).is_integer():
        return f"{int(rounded_one_decimal)}{suffix}"
    return f"{rounded_one_decimal:.1f}{suffix}"


def build_model_file_name(parameter_count: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_count = _format_parameter_count_short(parameter_count)
    return f"model_{short_count}_{timestamp}.pt"


def _read_json_if_exists(path_value: str | Path) -> dict | None:
    path = Path(path_value)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_warmup_cosine_scheduler(
    optimizer,
    total_steps: int,
    warmup_ratio: float,
    min_lr_ratio: float,
):
    resolved_total_steps = max(1, int(total_steps))
    resolved_warmup_ratio = max(0.0, min(0.95, float(warmup_ratio)))
    resolved_min_lr_ratio = max(0.0, min(1.0, float(min_lr_ratio)))
    warmup_steps = int(round(resolved_total_steps * resolved_warmup_ratio))

    if resolved_total_steps > 1:
        warmup_steps = min(warmup_steps, resolved_total_steps - 1)
    else:
        warmup_steps = 0

    def lr_lambda(step_index: int) -> float:
        current_step = int(step_index) + 1

        if warmup_steps > 0 and current_step <= warmup_steps:
            return max(1e-8, current_step / max(1, warmup_steps))

        if resolved_total_steps <= warmup_steps:
            return 1.0

        decay_progress = (current_step - warmup_steps) / max(1, resolved_total_steps - warmup_steps)
        decay_progress = max(0.0, min(1.0, decay_progress))
        cosine_value = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return resolved_min_lr_ratio + (1.0 - resolved_min_lr_ratio) * cosine_value

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    return {
        "scheduler": scheduler,
        "total_steps": int(resolved_total_steps),
        "warmup_steps": int(warmup_steps),
        "warmup_ratio": float(resolved_warmup_ratio),
        "min_lr_ratio": float(resolved_min_lr_ratio),
        "name": "warmup_cosine",
    }


def save_training_payload(
    output_path: str | Path,
    saved_model,
    model_config: dict,
    codec_info: dict,
    total_len: int,
    train_len: int,
    val_len: int,
    length_info: dict,
    processed_training_data_folder,
    resolved_gpu_ids: list[int],
    primary_device,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    validation_split: float,
    weight_decay: float,
    mixed_precision: bool,
    effective_pin_memory: bool,
    use_genre_metadata: bool,
    use_year_metadata: bool,
    use_mood_metadata: bool,
    use_key_metadata: bool,
    use_bpm_metadata: bool,
    parameter_count: int,
    output_file_name: str,
    metadata_vocab_path: str | Path,
    token_manifest_path: str | Path,
    history: list,
    token_budget_per_batch: int,
    max_examples_per_batch: int,
    lr_scheduler_name: str,
    lr_scheduler_total_steps: int,
    lr_warmup_ratio: float,
    lr_warmup_steps: int,
    lr_min_ratio: float,
    use_ema: bool,
    ema_decay: float,
    top_k_checkpoint_average: int,
    final_selection: str,
    averaged_checkpoint_count: int,
):
    output_path = Path(output_path)
    embedded_codec_info = dict(codec_info)
    embedded_metadata_vocab = _read_json_if_exists(metadata_vocab_path)

    payload = {
        "model_state_dict": saved_model.state_dict(),
        "token_model_config": model_config,
        "training_config": {
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "validation_split": float(validation_split),
            "weight_decay": float(weight_decay),
            "mixed_precision": bool(mixed_precision),
            "pin_memory": bool(effective_pin_memory),
            "use_genre_metadata": bool(use_genre_metadata),
            "use_year_metadata": bool(use_year_metadata),
            "use_mood_metadata": bool(use_mood_metadata),
            "use_key_metadata": bool(use_key_metadata),
            "use_bpm_metadata": bool(use_bpm_metadata),
            "training_gpu_ids": list(resolved_gpu_ids),
            "device": str(primary_device),
            "token_budget_per_batch": int(token_budget_per_batch),
            "max_examples_per_batch": int(max_examples_per_batch),
            "lr_scheduler_name": str(lr_scheduler_name),
            "lr_scheduler_total_steps": int(lr_scheduler_total_steps),
            "lr_warmup_ratio": float(lr_warmup_ratio),
            "lr_warmup_steps": int(lr_warmup_steps),
            "lr_min_ratio": float(lr_min_ratio),
            "use_ema": bool(use_ema),
            "ema_decay": float(ema_decay),
            "top_k_checkpoint_average": int(top_k_checkpoint_average),
            "final_selection": str(final_selection),
            "averaged_checkpoint_count": int(averaged_checkpoint_count),
        },
        "dataset_info": {
            "dataset_size": int(total_len),
            "train_size": int(train_len),
            "validation_size": int(val_len),
            "chunk_length_sec": float(length_info["chunk_length_sec"]),
            "median_token_length": int(length_info["median_token_length"]),
            "max_observed_token_length": int(length_info["max_observed_token_length"]),
            "effective_target_token_length": int(length_info["effective_target_token_length"]),
            "effective_clip_length_sec": float(length_info["effective_clip_length_sec"]),
            "sample_rate": int(codec_info.get("sample_rate", 24000)),
            "run_dir": str(output_path.parent),
            "processed_training_data_folder": str(processed_training_data_folder) if processed_training_data_folder else "",
            "parameter_count": int(parameter_count),
            "model_file_name": str(output_file_name),
        },
        "codec_info": embedded_codec_info,
        "metadata_vocab": embedded_metadata_vocab,
        "token_manifest_path": str(token_manifest_path),
        "history": history,
    }

    torch.save(payload, output_path)
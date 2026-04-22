from dataclasses import dataclass
from pathlib import Path

import torch

from src.data.metadata_encoder import MetadataEncoder
from src.models.metadata_token_transformer import MetadataTokenTransformer
from src.token_codec.dac_codec import DACCodec


@dataclass
class LoadedTokenModelBundle:
    model: MetadataTokenTransformer
    metadata_encoder: MetadataEncoder
    codec: DACCodec
    codec_info: dict
    model_config: dict
    training_config: dict
    dataset_info: dict
    target_token_length: int
    clip_length_sec: float
    reference_track_duration_sec: float
    device: torch.device
    payload: dict


def _load_payload(model_file: str | Path) -> tuple[Path, dict]:
    path = Path(model_file)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    payload = torch.load(path, map_location="cpu")

    if not isinstance(payload, dict):
        raise RuntimeError("Model file payload must be a dictionary.")

    if "model_state_dict" not in payload:
        raise RuntimeError("Model file is missing 'model_state_dict'.")

    if "token_model_config" not in payload:
        raise RuntimeError("Model file is missing 'token_model_config'.")

    return path, payload


def _ensure_supported_architecture(model_config: dict):
    architecture_name = str(model_config.get("architecture_name", ""))
    architecture_version = int(model_config.get("architecture_version", 0))

    if architecture_name != "token_decoder_rope_cache" or architecture_version != 1:
        raise RuntimeError("This model uses an unsupported architecture. Retrain it with the new decoder architecture.")


def _resolve_target_token_length(model_config: dict, dataset_info: dict) -> int:
    if "effective_target_token_length" in dataset_info:
        return max(1, int(dataset_info["effective_target_token_length"]))

    if "max_observed_token_length" in dataset_info:
        return max(1, int(dataset_info["max_observed_token_length"]))

    max_token_length = int(model_config.get("max_token_length", 2048))
    return max(1, max_token_length - 1)


def _resolve_clip_length_sec(dataset_info: dict) -> float:
    if "effective_clip_length_sec" in dataset_info:
        return float(dataset_info["effective_clip_length_sec"])

    return float(dataset_info.get("chunk_length_sec", 0.0))


def _resolve_metadata_flags(model_config: dict) -> dict:
    return {
        "use_genre_metadata": bool(model_config.get("use_genre_metadata", True)),
        "use_year_metadata": bool(model_config.get("use_year_metadata", True)),
        "use_mood_metadata": bool(model_config.get("use_mood_metadata", False)),
        "use_key_metadata": bool(model_config.get("use_key_metadata", False)),
        "use_bpm_metadata": bool(model_config.get("use_bpm_metadata", False)),
    }


def _build_metadata_encoder_from_payload(data: dict) -> MetadataEncoder:
    if not isinstance(data, dict) or not data:
        raise RuntimeError("Model file is missing embedded metadata vocabulary.")
    return MetadataEncoder(
        artist_to_id=data["artist_to_id"],
        genre_to_id=data["genre_to_id"],
        mood_to_id=data["mood_to_id"],
        key_to_id=data["key_to_id"],
        char_to_id=data["char_to_id"],
        id_to_artist=data["id_to_artist"],
        id_to_genre=data["id_to_genre"],
        id_to_mood=data["id_to_mood"],
        id_to_key=data["id_to_key"],
        id_to_char=data["id_to_char"],
        title_max_length=int(data["title_max_length"]),
        year_min=int(data["year_min"]),
        year_max=int(data["year_max"]),
        bpm_min=float(data["bpm_min"]),
        bpm_max=float(data["bpm_max"]),
    )


def load_token_model_summary(model_file: str | Path) -> dict:
    model_path, payload = _load_payload(model_file)

    model_config = dict(payload.get("token_model_config", {}))
    dataset_info = dict(payload.get("dataset_info", {}))
    training_config = dict(payload.get("training_config", {}))
    _ensure_supported_architecture(model_config)
    metadata_flags = _resolve_metadata_flags(model_config)

    n_heads = int(model_config.get("n_heads", 0))
    n_kv_heads = int(model_config.get("n_kv_heads", n_heads if n_heads > 0 else 0))
    gqa_groups = int(n_heads // n_kv_heads) if n_heads > 0 and n_kv_heads > 0 else 0

    return {
        "clip_length_sec": float(_resolve_clip_length_sec(dataset_info)),
        "sample_rate": int(dataset_info.get("sample_rate", 0)),
        "architecture_name": str(model_config.get("architecture_name", "")),
        "architecture_version": int(model_config.get("architecture_version", 0)),
        "d_model": int(model_config.get("d_model", 0)),
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "gqa_groups": gqa_groups,
        "n_layers": int(model_config.get("n_layers", 0)),
        "rope_base": float(model_config.get("rope_base", 0.0)),
        "dataset_size": int(dataset_info.get("dataset_size", 0)),
        "target_token_length": int(_resolve_target_token_length(model_config, dataset_info)),
        "run_dir": str(dataset_info.get("run_dir", "")),
        "use_genre_metadata": bool(metadata_flags["use_genre_metadata"]),
        "use_year_metadata": bool(metadata_flags["use_year_metadata"]),
        "use_mood_metadata": bool(metadata_flags["use_mood_metadata"]),
        "use_key_metadata": bool(metadata_flags["use_key_metadata"]),
        "use_bpm_metadata": bool(metadata_flags["use_bpm_metadata"]),
        "reference_track_duration_sec": float(model_config.get("reference_track_duration_sec", 240.0)),
        "max_token_length": int(model_config.get("max_token_length", 0)),
        "parameter_count": int(dataset_info.get("parameter_count", 0)),
        "model_file_name": str(dataset_info.get("model_file_name", model_path.name)),
        "device": str(training_config.get("device", "")),
    }


def load_trained_token_model(
    model_file: str | Path,
    device: str | torch.device | None = None,
) -> LoadedTokenModelBundle:
    _, payload = _load_payload(model_file)

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model_config = dict(payload.get("token_model_config", {}))
    training_config = dict(payload.get("training_config", {}))
    dataset_info = dict(payload.get("dataset_info", {}))
    _ensure_supported_architecture(model_config)
    metadata_flags = _resolve_metadata_flags(model_config)

    codec_info = payload.get("codec_info")
    metadata_vocab_payload = payload.get("metadata_vocab")

    if not isinstance(codec_info, dict) or not codec_info:
        raise RuntimeError("Model file is missing embedded codec info.")

    metadata_encoder = _build_metadata_encoder_from_payload(metadata_vocab_payload)

    codec_model_type = str(codec_info.get("codec_model_type", "24khz"))
    codec = DACCodec(model_type=codec_model_type, device=str(resolved_device))

    model = MetadataTokenTransformer(
        token_vocab_size=int(model_config["token_vocab_size"]),
        artist_vocab_size=int(model_config["artist_vocab_size"]),
        genre_vocab_size=int(model_config["genre_vocab_size"]),
        mood_vocab_size=int(model_config["mood_vocab_size"]),
        key_vocab_size=int(model_config["key_vocab_size"]),
        title_vocab_size=int(model_config["title_vocab_size"]),
        pad_token_id=int(model_config["pad_token_id"]),
        d_model=int(model_config["d_model"]),
        n_heads=int(model_config["n_heads"]),
        n_kv_heads=int(model_config.get("n_kv_heads", model_config["n_heads"])),
        n_layers=int(model_config["n_layers"]),
        ff_mult=int(model_config["ff_mult"]),
        dropout=float(model_config["dropout"]),
        max_token_length=int(model_config["max_token_length"]),
        metadata_prefix_tokens=int(model_config.get("metadata_prefix_tokens", 8)),
        use_genre_metadata=bool(metadata_flags["use_genre_metadata"]),
        use_year_metadata=bool(metadata_flags["use_year_metadata"]),
        use_mood_metadata=bool(metadata_flags["use_mood_metadata"]),
        use_key_metadata=bool(metadata_flags["use_key_metadata"]),
        use_bpm_metadata=bool(metadata_flags["use_bpm_metadata"]),
        rope_base=float(model_config.get("rope_base", 10000.0)),
    ).to(resolved_device)

    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    target_token_length = _resolve_target_token_length(model_config, dataset_info)
    clip_length_sec = _resolve_clip_length_sec(dataset_info)
    reference_track_duration_sec = float(model_config.get("reference_track_duration_sec", 240.0))

    return LoadedTokenModelBundle(
        model=model,
        metadata_encoder=metadata_encoder,
        codec=codec,
        codec_info=codec_info,
        model_config=model_config,
        training_config=training_config,
        dataset_info=dataset_info,
        target_token_length=int(target_token_length),
        clip_length_sec=float(clip_length_sec),
        reference_track_duration_sec=float(reference_track_duration_sec),
        device=resolved_device,
        payload=payload,
    )
import torch

from src.core.task_cancelled import TaskCancelledError
from src.generation.prompt_mapper import build_prompt_row


def resolve_metadata_flags(bundle) -> dict:
    return {
        "use_genre_metadata": bool(bundle.model_config.get("use_genre_metadata", True)),
        "use_year_metadata": bool(bundle.model_config.get("use_year_metadata", True)),
        "use_mood_metadata": bool(bundle.model_config.get("use_mood_metadata", False)),
        "use_key_metadata": bool(bundle.model_config.get("use_key_metadata", False)),
        "use_bpm_metadata": bool(bundle.model_config.get("use_bpm_metadata", False)),
    }


def encode_condition_tensors(
    bundle,
    artist: str,
    title: str,
    year,
    genre: str,
    relative_position: float,
    mood: str,
    initial_key: str,
    bpm,
):
    device = bundle.device
    metadata_flags = resolve_metadata_flags(bundle)

    effective_genre = genre if metadata_flags["use_genre_metadata"] else ""
    effective_year = year if metadata_flags["use_year_metadata"] else 0
    effective_mood = mood if metadata_flags["use_mood_metadata"] else ""
    effective_key = initial_key if metadata_flags["use_key_metadata"] else ""
    effective_bpm = bpm if metadata_flags["use_bpm_metadata"] else 0

    row = build_prompt_row(
        artist=artist,
        title=title,
        year=effective_year,
        genre=effective_genre,
        relative_position=float(relative_position),
        mood=effective_mood,
        initial_key=effective_key,
        bpm=effective_bpm,
    )
    encoded = bundle.metadata_encoder.encode_row(row)

    if not metadata_flags["use_genre_metadata"]:
        encoded["genre_id"] = 0

    if not metadata_flags["use_year_metadata"]:
        encoded["year_value"] = 0.0

    if not metadata_flags["use_mood_metadata"]:
        encoded["mood_id"] = 0

    if not metadata_flags["use_key_metadata"]:
        encoded["key_id"] = 0

    if not metadata_flags["use_bpm_metadata"]:
        encoded["bpm_value"] = 0.0

    artist_id = torch.tensor([encoded["artist_id"]], dtype=torch.long, device=device)
    genre_id = torch.tensor([encoded["genre_id"]], dtype=torch.long, device=device)
    mood_id = torch.tensor([encoded["mood_id"]], dtype=torch.long, device=device)
    key_id = torch.tensor([encoded["key_id"]], dtype=torch.long, device=device)
    title_tokens = torch.tensor([encoded["title_tokens"]], dtype=torch.long, device=device)
    title_length = torch.tensor([encoded["title_length"]], dtype=torch.long, device=device)
    year_value = torch.tensor([encoded["year_value"]], dtype=torch.float32, device=device)
    position_value = torch.tensor([encoded["position_value"]], dtype=torch.float32, device=device)
    bpm_value = torch.tensor([encoded["bpm_value"]], dtype=torch.float32, device=device)

    return {
        "artist_id": artist_id,
        "genre_id": genre_id,
        "mood_id": mood_id,
        "key_id": key_id,
        "title_tokens": title_tokens,
        "title_length": title_length,
        "year_value": year_value,
        "position_value": position_value,
        "bpm_value": bpm_value,
        "use_genre_metadata": metadata_flags["use_genre_metadata"],
        "use_year_metadata": metadata_flags["use_year_metadata"],
        "use_mood_metadata": metadata_flags["use_mood_metadata"],
        "use_key_metadata": metadata_flags["use_key_metadata"],
        "use_bpm_metadata": metadata_flags["use_bpm_metadata"],
        "effective_genre": effective_genre,
        "effective_year": effective_year,
        "effective_mood": effective_mood,
        "effective_key": effective_key,
        "effective_bpm": effective_bpm,
    }


def _sample_next_token(
    logits: torch.Tensor,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    base_audio_vocab: int,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    next_logits = logits.clone()
    next_logits[:, pad_token_id] = -1e9
    next_logits[:, bos_token_id] = -1e9
    next_logits[:, eos_token_id] = -1e9

    if next_logits.shape[-1] > base_audio_vocab:
        next_logits[:, base_audio_vocab:] = -1e9

    if temperature <= 0:
        return torch.argmax(next_logits, dim=-1, keepdim=True)

    next_logits = next_logits / max(float(temperature), 1e-6)

    if int(top_k) > 0 and int(top_k) < next_logits.shape[-1]:
        values, indices = torch.topk(next_logits, k=int(top_k), dim=-1)
        filtered = torch.full_like(next_logits, -1e9)
        filtered.scatter_(1, indices, values)
        next_logits = filtered

    probs = torch.softmax(next_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_audio_tokens(
    bundle,
    artist_id: torch.Tensor,
    genre_id: torch.Tensor,
    mood_id: torch.Tensor,
    key_id: torch.Tensor,
    title_tokens: torch.Tensor,
    title_length: torch.Tensor,
    year_value: torch.Tensor,
    position_value: torch.Tensor,
    bpm_value: torch.Tensor,
    temperature: float,
    top_k: int,
    progress_callback=None,
    is_cancelled=None,
    progress_token_offset: int = 0,
    progress_token_total: int | None = None,
    progress_status_prefix: str = "Generated token",
) -> torch.Tensor:
    model = bundle.model
    device = bundle.device

    bos_token_id = int(bundle.codec_info["bos_token_id"])
    pad_token_id = int(bundle.codec_info["pad_token_id"])
    eos_token_id = int(bundle.codec_info["eos_token_id"])
    base_audio_vocab = int(bundle.codec_info["codebook_size"]) * int(bundle.codec_info["n_codebooks"])

    target_token_length = max(1, int(bundle.target_token_length))
    display_total = int(progress_token_total) if progress_token_total is not None else int(target_token_length)

    generated = torch.empty((1, target_token_length), dtype=torch.long, device=device)

    with torch.inference_mode():
        kv_cache = model.init_kv_cache()
        prompt_token_ids = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)

        next_logits, kv_cache = model.prefill(
            prompt_token_ids=prompt_token_ids,
            artist_id=artist_id,
            genre_id=genre_id,
            mood_id=mood_id,
            key_id=key_id,
            title_tokens=title_tokens,
            title_length=title_length,
            year_value=year_value,
            position_value=position_value,
            bpm_value=bpm_value,
            kv_cache=kv_cache,
        )

        for step in range(target_token_length):
            if is_cancelled and is_cancelled():
                raise TaskCancelledError("Token generation cancelled.")

            next_token = _sample_next_token(
                logits=next_logits,
                pad_token_id=pad_token_id,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                base_audio_vocab=base_audio_vocab,
                temperature=float(temperature),
                top_k=int(top_k),
            )

            generated[:, step : step + 1] = next_token

            if progress_callback is not None:
                display_current = int(progress_token_offset) + int(step + 1)
                progress_callback(
                    "Generating Tokens",
                    display_current,
                    display_total,
                    f"{progress_status_prefix} {display_current}/{display_total}",
                )

            if step + 1 < target_token_length:
                next_logits, kv_cache = model.decode_step(
                    next_input_ids=next_token,
                    kv_cache=kv_cache,
                )

    generated = generated.squeeze(0).detach().cpu().long()
    usable = (generated.numel() // int(bundle.codec_info["n_codebooks"])) * int(bundle.codec_info["n_codebooks"])
    generated = generated[:usable]

    if generated.numel() == 0:
        raise RuntimeError("No audio tokens were generated.")

    return generated
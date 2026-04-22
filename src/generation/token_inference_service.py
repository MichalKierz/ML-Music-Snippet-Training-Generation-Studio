from pathlib import Path

import torch

from src.core.task_cancelled import TaskCancelledError
from src.generation.mp3_exporter import export_waveform_to_mp3
from src.generation.prompt_mapper import resolve_relative_position
from src.generation.token_inference_generation import (
    encode_condition_tensors,
    generate_audio_tokens,
)
from src.generation.token_inference_utils import (
    format_time_string,
    normalize_waveform_tensor,
    resolve_device,
    set_generation_seed,
)
from src.generation.token_model_loader import load_trained_token_model


class TokenInferenceService:
    def __init__(self, generate_defaults: dict):
        self.generate_defaults = dict(generate_defaults)

    def run(
        self,
        model_file: str | Path,
        output_folder: str | Path,
        artist: str,
        title: str,
        year,
        genre: str,
        position_mode: str,
        start_time: str,
        relative_position: float,
        section: str,
        mood: str = "",
        initial_key: str = "",
        bpm=0,
        temperature: float = 1.0,
        top_k: int = 50,
        seed: str = "",
        generate_song: bool = False,
        song_snippet_count: int = 1,
        progress_callback=None,
        is_cancelled=None,
    ) -> dict:
        def ensure_not_cancelled():
            if is_cancelled and is_cancelled():
                raise TaskCancelledError("Token generation cancelled.")

        device = resolve_device()
        used_seed = set_generation_seed(seed)

        if progress_callback is not None:
            progress_callback("Loading Token Model", 0, 1, "Loading trained token model")

        bundle = load_trained_token_model(model_file=model_file, device=device)
        ensure_not_cancelled()

        if bool(generate_song):
            return self._run_song_generation(
                bundle=bundle,
                model_file=model_file,
                output_folder=output_folder,
                artist=artist,
                title=title,
                year=year,
                genre=genre,
                mood=mood,
                initial_key=initial_key,
                bpm=bpm,
                temperature=float(temperature),
                top_k=int(top_k),
                seed_value=int(used_seed),
                snippet_count=max(1, int(song_snippet_count)),
                progress_callback=progress_callback,
                is_cancelled=is_cancelled,
            )

        return self._run_single_snippet_generation(
            bundle=bundle,
            model_file=model_file,
            output_folder=output_folder,
            artist=artist,
            title=title,
            year=year,
            genre=genre,
            position_mode=position_mode,
            start_time=start_time,
            relative_position=float(relative_position),
            section=section,
            mood=mood,
            initial_key=initial_key,
            bpm=bpm,
            temperature=float(temperature),
            top_k=int(top_k),
            seed_value=int(used_seed),
            progress_callback=progress_callback,
            is_cancelled=is_cancelled,
        )

    def _run_single_snippet_generation(
        self,
        bundle,
        model_file: str | Path,
        output_folder: str | Path,
        artist: str,
        title: str,
        year,
        genre: str,
        position_mode: str,
        start_time: str,
        relative_position: float,
        section: str,
        mood: str,
        initial_key: str,
        bpm,
        temperature: float,
        top_k: int,
        seed_value: int,
        progress_callback=None,
        is_cancelled=None,
    ) -> dict:
        def ensure_not_cancelled():
            if is_cancelled and is_cancelled():
                raise TaskCancelledError("Token generation cancelled.")

        resolved_position = resolve_relative_position(
            position_mode=position_mode,
            start_time=start_time,
            relative_position=relative_position,
            section=section,
            reference_track_duration_sec=float(bundle.reference_track_duration_sec),
        )

        condition = encode_condition_tensors(
            bundle=bundle,
            artist=artist,
            title=title,
            year=year,
            genre=genre,
            relative_position=float(resolved_position),
            mood=mood,
            initial_key=initial_key,
            bpm=bpm,
        )

        set_generation_seed(seed_value)

        generated_tokens = generate_audio_tokens(
            bundle=bundle,
            artist_id=condition["artist_id"],
            genre_id=condition["genre_id"],
            mood_id=condition["mood_id"],
            key_id=condition["key_id"],
            title_tokens=condition["title_tokens"],
            title_length=condition["title_length"],
            year_value=condition["year_value"],
            position_value=condition["position_value"],
            bpm_value=condition["bpm_value"],
            temperature=float(temperature),
            top_k=int(top_k),
            progress_callback=progress_callback,
            is_cancelled=is_cancelled,
            progress_token_offset=0,
            progress_token_total=max(1, int(bundle.target_token_length)),
            progress_status_prefix="Generated token",
        )

        ensure_not_cancelled()

        if progress_callback is not None:
            progress_callback(
                "Generating Tokens",
                max(1, int(bundle.target_token_length)),
                max(1, int(bundle.target_token_length)),
                "Decoding waveform",
            )

        with torch.inference_mode():
            waveform = bundle.codec.decode_tokens(generated_tokens)

        ensure_not_cancelled()

        if progress_callback is not None:
            progress_callback(
                "Generating Tokens",
                max(1, int(bundle.target_token_length)),
                max(1, int(bundle.target_token_length)),
                "Exporting MP3",
            )

        output_path = export_waveform_to_mp3(
            waveform=waveform,
            sample_rate=int(bundle.codec.sample_rate),
            output_folder=output_folder,
            artist=artist,
            title=title,
            year=condition["effective_year"],
            genre=condition["effective_genre"],
            position_mode=position_mode,
            start_time=start_time,
            duration_sec=float(bundle.clip_length_sec),
            resolved_position=float(resolved_position),
            reference_track_duration_sec=float(bundle.reference_track_duration_sec),
            bitrate=self.generate_defaults["bitrate"],
        )

        if progress_callback is not None:
            progress_callback(
                "Completed",
                1,
                1,
                "Token generation completed",
            )

        return {
            "output_path": str(output_path),
            "used_seed": int(seed_value),
            "resolved_position": float(resolved_position),
            "duration_sec": float(bundle.clip_length_sec),
            "sample_rate": int(bundle.codec.sample_rate),
            "model_file": str(Path(model_file)),
            "use_genre_metadata": bool(condition["use_genre_metadata"]),
            "use_year_metadata": bool(condition["use_year_metadata"]),
            "use_mood_metadata": bool(condition["use_mood_metadata"]),
            "use_key_metadata": bool(condition["use_key_metadata"]),
            "use_bpm_metadata": bool(condition["use_bpm_metadata"]),
            "song_mode": False,
            "generated_snippet_count": 1,
            "total_duration_sec": float(bundle.clip_length_sec),
        }

    def _run_song_generation(
        self,
        bundle,
        model_file: str | Path,
        output_folder: str | Path,
        artist: str,
        title: str,
        year,
        genre: str,
        mood: str,
        initial_key: str,
        bpm,
        temperature: float,
        top_k: int,
        seed_value: int,
        snippet_count: int,
        progress_callback=None,
        is_cancelled=None,
    ) -> dict:
        def ensure_not_cancelled():
            if is_cancelled and is_cancelled():
                raise TaskCancelledError("Token generation cancelled.")

        total_token_target = max(1, int(bundle.target_token_length)) * max(1, int(snippet_count))
        waveform_parts = []

        for snippet_index in range(max(1, int(snippet_count))):
            ensure_not_cancelled()

            start_seconds = float(snippet_index) * float(bundle.clip_length_sec)
            if snippet_count <= 1:
                song_relative_position = 0.0
            else:
                song_relative_position = float(snippet_index) / float(snippet_count - 1)

            condition = encode_condition_tensors(
                bundle=bundle,
                artist=artist,
                title=title,
                year=year,
                genre=genre,
                relative_position=float(song_relative_position),
                mood=mood,
                initial_key=initial_key,
                bpm=bpm,
            )

            set_generation_seed(int(seed_value) + int(snippet_index))

            generated_tokens = generate_audio_tokens(
                bundle=bundle,
                artist_id=condition["artist_id"],
                genre_id=condition["genre_id"],
                mood_id=condition["mood_id"],
                key_id=condition["key_id"],
                title_tokens=condition["title_tokens"],
                title_length=condition["title_length"],
                year_value=condition["year_value"],
                position_value=condition["position_value"],
                bpm_value=condition["bpm_value"],
                temperature=float(temperature),
                top_k=int(top_k),
                progress_callback=progress_callback,
                is_cancelled=is_cancelled,
                progress_token_offset=int(snippet_index) * int(bundle.target_token_length),
                progress_token_total=int(total_token_target),
                progress_status_prefix=f"Snippet {snippet_index + 1}/{snippet_count} | Generated token",
            )

            ensure_not_cancelled()

            if progress_callback is not None:
                progress_callback(
                    "Generating Tokens",
                    int(snippet_index + 1) * int(bundle.target_token_length),
                    int(total_token_target),
                    f"Snippet {snippet_index + 1}/{snippet_count} | Decoding waveform [{format_time_string(start_seconds)} - {format_time_string(start_seconds + float(bundle.clip_length_sec))}]",
                )

            with torch.inference_mode():
                snippet_waveform = bundle.codec.decode_tokens(generated_tokens)

            waveform_parts.append(normalize_waveform_tensor(snippet_waveform))

        ensure_not_cancelled()

        stitched_waveform = waveform_parts[0] if len(waveform_parts) == 1 else torch.cat(waveform_parts, dim=-1)
        total_duration_sec = float(bundle.clip_length_sec) * float(snippet_count)

        if progress_callback is not None:
            progress_callback(
                "Generating Tokens",
                int(total_token_target),
                int(total_token_target),
                "Exporting stitched MP3",
            )

        output_path = export_waveform_to_mp3(
            waveform=stitched_waveform,
            sample_rate=int(bundle.codec.sample_rate),
            output_folder=output_folder,
            artist=artist,
            title=title,
            year=condition["effective_year"],
            genre=condition["effective_genre"],
            position_mode="Start Time",
            start_time="00:00",
            duration_sec=float(total_duration_sec),
            resolved_position=0.0,
            reference_track_duration_sec=float(max(bundle.reference_track_duration_sec, total_duration_sec)),
            bitrate=self.generate_defaults["bitrate"],
        )

        if progress_callback is not None:
            progress_callback(
                "Completed",
                1,
                1,
                "Song generation completed",
            )

        return {
            "output_path": str(output_path),
            "used_seed": int(seed_value),
            "resolved_position": 0.0,
            "duration_sec": float(bundle.clip_length_sec),
            "sample_rate": int(bundle.codec.sample_rate),
            "model_file": str(Path(model_file)),
            "use_genre_metadata": bool(condition["use_genre_metadata"]),
            "use_year_metadata": bool(condition["use_year_metadata"]),
            "use_mood_metadata": bool(condition["use_mood_metadata"]),
            "use_key_metadata": bool(condition["use_key_metadata"]),
            "use_bpm_metadata": bool(condition["use_bpm_metadata"]),
            "song_mode": True,
            "generated_snippet_count": int(snippet_count),
            "total_duration_sec": float(total_duration_sec),
        }
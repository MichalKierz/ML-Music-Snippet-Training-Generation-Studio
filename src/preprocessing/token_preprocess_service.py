import json
import shutil
from pathlib import Path

import pandas as pd

from src.core.device_utils import normalize_preprocess_device
from src.core.task_cancelled import TaskCancelledError
from src.preprocessing.chunk_builder import build_track_chunks
from src.preprocessing.duration_reader import read_duration_seconds
from src.preprocessing.filename_parser import resolve_artist_title
from src.preprocessing.metadata_reader import read_metadata
from src.preprocessing.scanner import scan_audio_files
from src.preprocessing.token_cache_builder import TokenCacheBuilder
from src.token_codec.dac_codec import DACCodec


class TokenPreprocessService:
    def __init__(self, preprocess_defaults: dict, token_model_defaults: dict):
        self.preprocess_defaults = dict(preprocess_defaults)
        self.token_model_defaults = dict(token_model_defaults)

    def run(
        self,
        raw_music_folder: str | Path,
        processed_training_data_folder: str | Path,
        chunk_length_sec: float | None = None,
        chunk_stride_sec: float | None = None,
        preprocess_device: str | None = None,
        clear_existing: bool = True,
        progress_callback=None,
        is_cancelled=None,
    ) -> dict:
        def cancelled() -> bool:
            return bool(is_cancelled and is_cancelled())

        def ensure_not_cancelled():
            if cancelled():
                raise TaskCancelledError("Token preprocessing cancelled.")

        effective_config = self._build_effective_config(
            chunk_length_sec=chunk_length_sec,
            chunk_stride_sec=chunk_stride_sec,
        )

        raw_folder = Path(raw_music_folder)
        output_folder = Path(processed_training_data_folder)
        resolved_preprocess_device = normalize_preprocess_device(preprocess_device)

        def emit(stage: str, current: int, total: int, status: str):
            if progress_callback is not None:
                progress_callback(stage, current, total, status)

        ensure_not_cancelled()

        manifests_dir, token_cache_dir, vocab_dir, library_manifest_path, token_manifest_path, preprocess_config_path, codec_info_path = self._prepare_output_layout(
            output_folder=output_folder,
            clear_existing=clear_existing,
        )
        emit("Preparing Output", 0, 0, "Preparing token output folders")

        ensure_not_cancelled()

        audio_files = scan_audio_files(raw_folder, effective_config["audio_extensions"])
        total_files = len(audio_files)
        emit("Scanning Library", 0, 0, f"Found {total_files} audio files")

        library_rows = self._build_library_rows(audio_files, effective_config, emit, is_cancelled)
        library_df = pd.DataFrame(library_rows)
        library_df.to_csv(library_manifest_path, index=False, encoding="utf-8")

        ensure_not_cancelled()

        chunk_rows = self._build_chunk_rows(library_df, effective_config, emit, is_cancelled)
        chunk_df = pd.DataFrame(chunk_rows)

        if not chunk_df.empty:
            chunk_df.insert(0, "row_id", range(len(chunk_df)))
        else:
            chunk_df["row_id"] = pd.Series(dtype="int64")

        total_chunks = len(chunk_df)
        completed_units_before_cache = 1 + total_files + total_files
        total_units = completed_units_before_cache + total_chunks + 1

        emit(
            "Preparing Token Cache",
            completed_units_before_cache,
            total_units,
            f"Indexed {total_files} files and built {total_chunks} chunks",
        )

        codec = DACCodec(
            model_type=effective_config["codec_model_type"],
            device=resolved_preprocess_device,
        )

        builder = TokenCacheBuilder(
            codec=codec,
            chunk_sample_rate=int(effective_config["chunk_sample_rate"]),
        )

        def token_progress(stage: str, current: int, total: int, status: str):
            if total <= 0:
                emit(stage, completed_units_before_cache, total_units, status)
                return
            global_current = completed_units_before_cache + int(current)
            emit(stage, global_current, total_units, status)

        token_df, codec_info = builder.build(
            chunk_df=chunk_df,
            token_cache_dir=token_cache_dir,
            progress_callback=token_progress,
            is_cancelled=is_cancelled,
        )

        codec_info["codec_model_type"] = str(effective_config["codec_model_type"])
        codec_info["reference_track_duration_sec"] = float(effective_config["reference_track_duration_sec"])
        codec_info["preprocess_device"] = str(resolved_preprocess_device)

        token_df.to_csv(token_manifest_path, index=False, encoding="utf-8")

        ensure_not_cancelled()

        with preprocess_config_path.open("w", encoding="utf-8") as handle:
            json.dump(effective_config, handle, indent=2, ensure_ascii=False)

        with codec_info_path.open("w", encoding="utf-8") as handle:
            json.dump(codec_info, handle, indent=2, ensure_ascii=False)

        emit("Saving Metadata", total_units, total_units, "Token preprocessing completed")

        return {
            "raw_music_folder": str(raw_folder),
            "processed_training_data_folder": str(output_folder),
            "library_manifest_path": str(library_manifest_path),
            "token_manifest_path": str(token_manifest_path),
            "token_preprocess_config_path": str(preprocess_config_path),
            "codec_info_path": str(codec_info_path),
            "audio_file_count": int(len(library_df)),
            "chunk_count": int(len(token_df)),
            "token_cache_dir": str(token_cache_dir),
            "vocab_dir": str(vocab_dir),
        }

    def _build_effective_config(
        self,
        chunk_length_sec: float | None,
        chunk_stride_sec: float | None,
    ) -> dict:
        return {
            "audio_extensions": list(self.preprocess_defaults["audio_extensions"]),
            "filename_delimiter": str(self.preprocess_defaults["filename_delimiter"]),
            "chunk_length_sec": float(chunk_length_sec if chunk_length_sec is not None else self.preprocess_defaults["chunk_length_sec"]),
            "chunk_stride_sec": float(chunk_stride_sec if chunk_stride_sec is not None else self.preprocess_defaults["chunk_stride_sec"]),
            "codec_name": str(self.token_model_defaults["codec_name"]),
            "codec_model_type": str(self.token_model_defaults["codec_model_type"]),
            "chunk_sample_rate": int(self.token_model_defaults["chunk_sample_rate"]),
            "reference_track_duration_sec": float(self.token_model_defaults["reference_track_duration_sec"]),
        }

    def _prepare_output_layout(
        self,
        output_folder: Path,
        clear_existing: bool,
    ):
        manifests_dir = output_folder / "manifests"
        token_cache_dir = output_folder / "token_cache"
        vocab_dir = output_folder / "vocab"
        preprocess_config_path = output_folder / "token_preprocess_config.json"
        codec_info_path = output_folder / "codec_info.json"
        library_manifest_path = manifests_dir / "token_library_manifest.csv"
        token_manifest_path = manifests_dir / "token_chunk_manifest.csv"

        output_folder.mkdir(parents=True, exist_ok=True)

        if clear_existing:
            if manifests_dir.exists():
                shutil.rmtree(manifests_dir)
            if token_cache_dir.exists():
                shutil.rmtree(token_cache_dir)
            if vocab_dir.exists():
                shutil.rmtree(vocab_dir)
            if preprocess_config_path.exists():
                preprocess_config_path.unlink()
            if codec_info_path.exists():
                codec_info_path.unlink()

        manifests_dir.mkdir(parents=True, exist_ok=True)
        token_cache_dir.mkdir(parents=True, exist_ok=True)
        vocab_dir.mkdir(parents=True, exist_ok=True)

        return (
            manifests_dir,
            token_cache_dir,
            vocab_dir,
            library_manifest_path,
            token_manifest_path,
            preprocess_config_path,
            codec_info_path,
        )

    def _build_library_rows(self, audio_files: list[Path], config: dict, emit, is_cancelled) -> list[dict]:
        rows = []
        total = max(1, len(audio_files))

        for index, audio_file in enumerate(audio_files, start=1):
            if is_cancelled and is_cancelled():
                raise TaskCancelledError("Token preprocessing cancelled.")

            metadata = read_metadata(audio_file)
            resolved = resolve_artist_title(
                file_path=audio_file,
                delimiter=config["filename_delimiter"],
                tag_artist=metadata["tag_artist"],
                tag_title=metadata["tag_title"],
            )
            duration_sec = read_duration_seconds(audio_file)

            rows.append(
                {
                    "path": str(audio_file),
                    "filename": audio_file.name,
                    "filename_stem": resolved["filename_stem"],
                    "tag_artist": metadata["tag_artist"],
                    "tag_title": metadata["tag_title"],
                    "tag_year": metadata["tag_year"],
                    "tag_genre": metadata["tag_genre"],
                    "tag_mood": metadata["tag_mood"],
                    "tag_initial_key": metadata["tag_initial_key"],
                    "tag_bpm": metadata["tag_bpm"],
                    "artist": resolved["artist"],
                    "title": resolved["title"],
                    "year": metadata["tag_year"],
                    "genre": metadata["tag_genre"],
                    "mood": metadata["tag_mood"],
                    "initial_key": metadata["tag_initial_key"],
                    "bpm": metadata["tag_bpm"],
                    "artist_source": resolved["artist_source"],
                    "title_source": resolved["title_source"],
                    "duration_sec": round(float(duration_sec), 6),
                }
            )

            emit("Building Library Index", index, total, f"Indexed {index}/{total} files")

        return rows

    def _build_chunk_rows(self, library_df: pd.DataFrame, config: dict, emit, is_cancelled) -> list[dict]:
        rows = []
        total = max(1, len(library_df))

        for file_index, row in enumerate(library_df.itertuples(index=False), start=1):
            if is_cancelled and is_cancelled():
                raise TaskCancelledError("Token preprocessing cancelled.")

            chunks = build_track_chunks(
                track_duration=float(row.duration_sec),
                chunk_length_sec=float(config["chunk_length_sec"]),
                chunk_stride_sec=float(config["chunk_stride_sec"]),
            )

            for chunk in chunks:
                rows.append(
                    {
                        "path": row.path,
                        "filename": row.filename,
                        "artist": row.artist,
                        "title": row.title,
                        "year": row.year,
                        "genre": row.genre,
                        "mood": row.mood,
                        "initial_key": row.initial_key,
                        "bpm": row.bpm,
                        "duration_sec": row.duration_sec,
                        "chunk_length_sec": float(config["chunk_length_sec"]),
                        "chunk_stride_sec": float(config["chunk_stride_sec"]),
                        "chunk_index": chunk["chunk_index"],
                        "chunk_count": chunk["chunk_count"],
                        "chunk_start_sec": chunk["chunk_start_sec"],
                        "chunk_end_sec": chunk["chunk_end_sec"],
                        "relative_position": chunk["relative_position"],
                    }
                )

            emit("Building Chunk Index", file_index, total, f"Built chunks for {file_index}/{total} files")

        return rows
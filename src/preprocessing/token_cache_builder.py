from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.core.task_cancelled import TaskCancelledError
from src.preprocessing.audio_loader import load_audio_with_fallback
from src.token_codec.dac_codec import DACCodec


class TokenCacheBuilder:
    def __init__(self, codec: DACCodec, chunk_sample_rate: int):
        self.codec = codec
        self.chunk_sample_rate = int(chunk_sample_rate)

    def _extract_frames(self, waveform: torch.Tensor, start_sec: float, end_sec: float) -> torch.Tensor:
        start_frame = max(0, int(round(float(start_sec) * self.chunk_sample_rate)))
        end_frame = max(start_frame, int(round(float(end_sec) * self.chunk_sample_rate)))
        chunk = waveform[:, start_frame:end_frame]

        target_frames = max(1, int(round((float(end_sec) - float(start_sec)) * self.chunk_sample_rate)))

        if chunk.shape[1] < target_frames:
            pad_frames = target_frames - chunk.shape[1]
            chunk = torch.nn.functional.pad(chunk, (0, pad_frames))

        if chunk.shape[1] > target_frames:
            chunk = chunk[:, :target_frames]

        return chunk

    def build(
        self,
        chunk_df: pd.DataFrame,
        token_cache_dir: str | Path,
        progress_callback=None,
        is_cancelled=None,
    ) -> tuple[pd.DataFrame, dict]:
        cache_dir = Path(token_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        result_df = chunk_df.copy()
        result_df["token_cache_path"] = ""
        result_df["token_length"] = 0

        total = len(result_df)
        grouped_items = list(result_df.groupby("path", sort=False))
        grouped_total = len(grouped_items)
        processed_rows = 0

        for file_index, (audio_path, group_df) in enumerate(grouped_items, start=1):
            if is_cancelled and is_cancelled():
                raise TaskCancelledError("Token preprocessing cancelled.")

            if progress_callback is not None:
                progress_callback(
                    "Building Token Cache",
                    processed_rows,
                    total,
                    f"Loading source file {file_index}/{grouped_total}",
                )

            waveform, _ = load_audio_with_fallback(
                file_path=audio_path,
                sample_rate=self.chunk_sample_rate,
                mono=True,
                timeout_sec=30.0,
            )

            for row_index, row in group_df.iterrows():
                if is_cancelled and is_cancelled():
                    raise TaskCancelledError("Token preprocessing cancelled.")

                chunk = self._extract_frames(
                    waveform=waveform,
                    start_sec=float(row["chunk_start_sec"]),
                    end_sec=float(row["chunk_end_sec"]),
                )

                codes = self.codec.encode_waveform(chunk, self.chunk_sample_rate)
                flat_tokens = self.codec.flatten_codes(codes).numpy().astype(np.int32)

                token_file = cache_dir / f"chunk_{int(row['row_id']):08d}.npz"
                np.savez_compressed(token_file, tokens=flat_tokens)

                result_df.at[row_index, "token_cache_path"] = str(token_file)
                result_df.at[row_index, "token_length"] = int(flat_tokens.shape[0])

                processed_rows += 1
                if progress_callback is not None:
                    progress_callback(
                        "Building Token Cache",
                        processed_rows,
                        total,
                        f"Cached {processed_rows}/{total} token chunks",
                    )

        spec = self.codec.get_codec_spec()

        token_info = {
            "codec_name": "dac",
            "sample_rate": int(spec.sample_rate),
            "codebook_size": int(spec.codebook_size),
            "n_codebooks": int(spec.n_codebooks),
            "pad_token_id": int(spec.pad_token_id),
            "bos_token_id": int(spec.bos_token_id),
            "eos_token_id": int(spec.eos_token_id),
            "vocab_size": int(spec.vocab_size),
        }

        return result_df, token_info
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.metadata_encoder import MetadataEncoder


class TokenSequenceDataset(Dataset):
    def __init__(
        self,
        token_manifest_path: str | Path,
        codec_info: dict,
        metadata_encoder: MetadataEncoder | None = None,
        build_metadata_encoder: bool = False,
        metadata_vocab_path: str | Path | None = None,
        title_max_length: int = 64,
        max_token_length: int = 2048,
        use_genre_metadata: bool = True,
        use_year_metadata: bool = True,
        use_mood_metadata: bool = False,
        use_key_metadata: bool = False,
        use_bpm_metadata: bool = False,
    ):
        self.token_manifest_path = Path(token_manifest_path)
        self.codec_info = dict(codec_info)
        self.max_token_length = int(max_token_length)
        self.use_genre_metadata = bool(use_genre_metadata)
        self.use_year_metadata = bool(use_year_metadata)
        self.use_mood_metadata = bool(use_mood_metadata)
        self.use_key_metadata = bool(use_key_metadata)
        self.use_bpm_metadata = bool(use_bpm_metadata)

        self.df = pd.read_csv(self.token_manifest_path)
        self.df = self.df[self.df["token_cache_path"].fillna("").map(lambda value: Path(str(value)).exists())].reset_index(drop=True)

        if metadata_encoder is not None:
            self.metadata_encoder = metadata_encoder
        else:
            if metadata_vocab_path is None:
                raise ValueError("metadata_vocab_path is required when metadata_encoder is not provided")

            metadata_vocab_path = Path(metadata_vocab_path)

            if build_metadata_encoder or not metadata_vocab_path.exists():
                self.metadata_encoder = MetadataEncoder.fit_from_chunk_dataframe(
                    self.df,
                    title_max_length=title_max_length,
                )
                self.metadata_encoder.save(metadata_vocab_path)
            else:
                self.metadata_encoder = MetadataEncoder.load(metadata_vocab_path)

        self.pad_token_id = int(self.codec_info["pad_token_id"])
        self.bos_token_id = int(self.codec_info["bos_token_id"])
        self.eos_token_id = int(self.codec_info["eos_token_id"])
        self.vocab_size = int(self.codec_info["vocab_size"])

    def __len__(self) -> int:
        return len(self.df)

    def _load_tokens(self, path: str | Path) -> torch.Tensor:
        loaded = np.load(str(path))
        tokens = loaded["tokens"].astype(np.int64, copy=False)
        return torch.from_numpy(tokens)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index].to_dict()
        tokens = self._load_tokens(row["token_cache_path"])

        max_payload_length = max(1, self.max_token_length - 1)
        tokens = tokens[:max_payload_length]

        input_ids = torch.cat(
            [
                torch.tensor([self.bos_token_id], dtype=torch.long),
                tokens.long(),
            ],
            dim=0,
        )

        target_ids = torch.cat(
            [
                tokens.long(),
                torch.tensor([self.eos_token_id], dtype=torch.long),
            ],
            dim=0,
        )

        encoded = self.metadata_encoder.encode_row(row)

        genre_id = int(encoded["genre_id"]) if self.use_genre_metadata else 0
        year_value = float(encoded["year_value"]) if self.use_year_metadata else 0.0
        mood_id = int(encoded["mood_id"]) if self.use_mood_metadata else 0
        key_id = int(encoded["key_id"]) if self.use_key_metadata else 0
        bpm_value = float(encoded["bpm_value"]) if self.use_bpm_metadata else 0.0

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "artist_id": torch.tensor(encoded["artist_id"], dtype=torch.long),
            "genre_id": torch.tensor(genre_id, dtype=torch.long),
            "mood_id": torch.tensor(mood_id, dtype=torch.long),
            "key_id": torch.tensor(key_id, dtype=torch.long),
            "title_tokens": torch.tensor(encoded["title_tokens"], dtype=torch.long),
            "title_length": torch.tensor(encoded["title_length"], dtype=torch.long),
            "year_value": torch.tensor(year_value, dtype=torch.float32),
            "position_value": torch.tensor(encoded["position_value"], dtype=torch.float32),
            "bpm_value": torch.tensor(bpm_value, dtype=torch.float32),
            "row_id": torch.tensor(int(row["row_id"]), dtype=torch.long),
        }


def collate_token_batch(batch: list[dict], pad_token_id: int) -> dict:
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids = []
    target_ids = []
    attention_mask = []

    for item in batch:
        seq_len = int(item["input_ids"].shape[0])
        pad_len = max_len - seq_len

        input_ids.append(
            torch.nn.functional.pad(
                item["input_ids"],
                (0, pad_len),
                value=int(pad_token_id),
            )
        )

        target_ids.append(
            torch.nn.functional.pad(
                item["target_ids"],
                (0, pad_len),
                value=-100,
            )
        )

        mask = torch.ones(seq_len, dtype=torch.bool)
        if pad_len > 0:
            mask = torch.nn.functional.pad(mask, (0, pad_len), value=False)
        attention_mask.append(mask)

    result = {
        "input_ids": torch.stack(input_ids, dim=0),
        "target_ids": torch.stack(target_ids, dim=0),
        "attention_mask": torch.stack(attention_mask, dim=0),
        "artist_id": torch.stack([item["artist_id"] for item in batch], dim=0),
        "genre_id": torch.stack([item["genre_id"] for item in batch], dim=0),
        "mood_id": torch.stack([item["mood_id"] for item in batch], dim=0),
        "key_id": torch.stack([item["key_id"] for item in batch], dim=0),
        "title_tokens": torch.stack([item["title_tokens"] for item in batch], dim=0),
        "title_length": torch.stack([item["title_length"] for item in batch], dim=0),
        "year_value": torch.stack([item["year_value"] for item in batch], dim=0),
        "position_value": torch.stack([item["position_value"] for item in batch], dim=0),
        "bpm_value": torch.stack([item["bpm_value"] for item in batch], dim=0),
        "row_id": torch.stack([item["row_id"] for item in batch], dim=0),
    }

    return result
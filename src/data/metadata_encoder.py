import json
from pathlib import Path

import pandas as pd


def normalize_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if text == "nan":
        return ""
    return text


def parse_year(value) -> int | None:
    text = normalize_text(value)
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) < 4:
        return None
    year = int(digits[:4])
    if year < 1000 or year > 3000:
        return None
    return year


def parse_bpm(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    filtered = "".join(ch for ch in text if ch.isdigit() or ch in {".", ","})
    if not filtered:
        return None
    filtered = filtered.replace(",", ".")
    try:
        numeric = float(filtered)
    except Exception:
        return None
    if numeric <= 0:
        return None
    return float(numeric)


class MetadataEncoder:
    def __init__(
        self,
        artist_to_id: dict[str, int],
        genre_to_id: dict[str, int],
        mood_to_id: dict[str, int],
        key_to_id: dict[str, int],
        char_to_id: dict[str, int],
        id_to_artist: list[str],
        id_to_genre: list[str],
        id_to_mood: list[str],
        id_to_key: list[str],
        id_to_char: list[str],
        title_max_length: int,
        year_min: int,
        year_max: int,
        bpm_min: float,
        bpm_max: float,
    ):
        self.artist_to_id = artist_to_id
        self.genre_to_id = genre_to_id
        self.mood_to_id = mood_to_id
        self.key_to_id = key_to_id
        self.char_to_id = char_to_id
        self.id_to_artist = id_to_artist
        self.id_to_genre = id_to_genre
        self.id_to_mood = id_to_mood
        self.id_to_key = id_to_key
        self.id_to_char = id_to_char
        self.title_max_length = int(title_max_length)
        self.year_min = int(year_min)
        self.year_max = int(year_max)
        self.bpm_min = float(bpm_min)
        self.bpm_max = float(bpm_max)

    @classmethod
    def fit_from_chunk_dataframe(cls, chunk_df: pd.DataFrame, title_max_length: int = 64):
        artists = sorted(
            {
                normalize_text(value)
                for value in chunk_df["artist"].fillna("").tolist()
                if normalize_text(value)
            }
        )

        genres = sorted(
            {
                normalize_text(value)
                for value in chunk_df.get("genre", pd.Series(dtype=str)).fillna("").tolist()
                if normalize_text(value)
            }
        )

        moods = sorted(
            {
                normalize_text(value)
                for value in chunk_df.get("mood", pd.Series(dtype=str)).fillna("").tolist()
                if normalize_text(value)
            }
        )

        keys = sorted(
            {
                normalize_text(value)
                for value in chunk_df.get("initial_key", pd.Series(dtype=str)).fillna("").tolist()
                if normalize_text(value)
            }
        )

        titles = [
            normalize_text(value)
            for value in chunk_df["title"].fillna("").tolist()
        ]

        chars = sorted({char for title in titles for char in title if char})

        id_to_artist = ["<unk>"] + artists
        id_to_genre = ["<unk>"] + genres
        id_to_mood = ["<unk>"] + moods
        id_to_key = ["<unk>"] + keys
        id_to_char = ["<pad>", "<unk>"] + chars

        artist_to_id = {value: index for index, value in enumerate(id_to_artist)}
        genre_to_id = {value: index for index, value in enumerate(id_to_genre)}
        mood_to_id = {value: index for index, value in enumerate(id_to_mood)}
        key_to_id = {value: index for index, value in enumerate(id_to_key)}
        char_to_id = {value: index for index, value in enumerate(id_to_char)}

        years = [parse_year(value) for value in chunk_df.get("year", pd.Series(dtype=str)).fillna("").tolist()]
        valid_years = [year for year in years if year is not None]

        if valid_years:
            year_min = min(valid_years)
            year_max = max(valid_years)
        else:
            year_min = 0
            year_max = 1

        bpms = [parse_bpm(value) for value in chunk_df.get("bpm", pd.Series(dtype=str)).fillna("").tolist()]
        valid_bpms = [bpm for bpm in bpms if bpm is not None]

        if valid_bpms:
            bpm_min = min(valid_bpms)
            bpm_max = max(valid_bpms)
        else:
            bpm_min = 0.0
            bpm_max = 1.0

        return cls(
            artist_to_id=artist_to_id,
            genre_to_id=genre_to_id,
            mood_to_id=mood_to_id,
            key_to_id=key_to_id,
            char_to_id=char_to_id,
            id_to_artist=id_to_artist,
            id_to_genre=id_to_genre,
            id_to_mood=id_to_mood,
            id_to_key=id_to_key,
            id_to_char=id_to_char,
            title_max_length=title_max_length,
            year_min=year_min,
            year_max=year_max,
            bpm_min=bpm_min,
            bpm_max=bpm_max,
        )

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(
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
            title_max_length=data["title_max_length"],
            year_min=data["year_min"],
            year_max=data["year_max"],
            bpm_min=float(data["bpm_min"]),
            bpm_max=float(data["bpm_max"]),
        )

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "artist_to_id": self.artist_to_id,
            "genre_to_id": self.genre_to_id,
            "mood_to_id": self.mood_to_id,
            "key_to_id": self.key_to_id,
            "char_to_id": self.char_to_id,
            "id_to_artist": self.id_to_artist,
            "id_to_genre": self.id_to_genre,
            "id_to_mood": self.id_to_mood,
            "id_to_key": self.id_to_key,
            "id_to_char": self.id_to_char,
            "title_max_length": self.title_max_length,
            "year_min": self.year_min,
            "year_max": self.year_max,
            "bpm_min": self.bpm_min,
            "bpm_max": self.bpm_max,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    @property
    def pad_token_id(self) -> int:
        return self.char_to_id["<pad>"]

    @property
    def unk_token_id(self) -> int:
        return self.char_to_id["<unk>"]

    @property
    def artist_vocab_size(self) -> int:
        return len(self.id_to_artist)

    @property
    def genre_vocab_size(self) -> int:
        return len(self.id_to_genre)

    @property
    def mood_vocab_size(self) -> int:
        return len(self.id_to_mood)

    @property
    def key_vocab_size(self) -> int:
        return len(self.id_to_key)

    @property
    def title_vocab_size(self) -> int:
        return len(self.id_to_char)

    def encode_artist(self, value) -> int:
        key = normalize_text(value)
        return int(self.artist_to_id.get(key, self.artist_to_id["<unk>"]))

    def encode_genre(self, value) -> int:
        key = normalize_text(value)
        return int(self.genre_to_id.get(key, self.genre_to_id["<unk>"]))

    def encode_mood(self, value) -> int:
        key = normalize_text(value)
        return int(self.mood_to_id.get(key, self.mood_to_id["<unk>"]))

    def encode_key(self, value) -> int:
        key = normalize_text(value)
        return int(self.key_to_id.get(key, self.key_to_id["<unk>"]))

    def encode_title(self, value) -> tuple[list[int], int]:
        text = normalize_text(value)
        tokens = [self.char_to_id.get(char, self.unk_token_id) for char in text[: self.title_max_length]]
        length = len(tokens)

        if length < self.title_max_length:
            tokens.extend([self.pad_token_id] * (self.title_max_length - length))

        return tokens, length

    def encode_year(self, value) -> float:
        year = parse_year(value)
        if year is None:
            return 0.0

        if self.year_max <= self.year_min:
            return 0.0

        normalized = (year - self.year_min) / (self.year_max - self.year_min)
        return float(max(0.0, min(1.0, normalized)))

    def encode_position(self, value) -> float:
        try:
            numeric = float(value)
        except Exception:
            numeric = 0.0
        return float(max(0.0, min(1.0, numeric)))

    def encode_bpm(self, value) -> float:
        bpm = parse_bpm(value)
        if bpm is None:
            return 0.0

        if self.bpm_max <= self.bpm_min:
            return 0.0

        normalized = (bpm - self.bpm_min) / (self.bpm_max - self.bpm_min)
        return float(max(0.0, min(1.0, normalized)))

    def encode_row(self, row: dict) -> dict:
        title_tokens, title_length = self.encode_title(row.get("title", ""))
        return {
            "artist_id": self.encode_artist(row.get("artist", "")),
            "genre_id": self.encode_genre(row.get("genre", "")),
            "mood_id": self.encode_mood(row.get("mood", "")),
            "key_id": self.encode_key(row.get("initial_key", "")),
            "title_tokens": title_tokens,
            "title_length": int(title_length),
            "year_value": self.encode_year(row.get("year", "")),
            "position_value": self.encode_position(row.get("relative_position", 0.0)),
            "bpm_value": self.encode_bpm(row.get("bpm", "")),
        }
import re
from pathlib import Path

from mutagen import File


def _normalize_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="ignore").strip()
        except Exception:
            return str(value).strip()
    if isinstance(value, list):
        if not value:
            return ""
        return _normalize_value(value[0])
    if hasattr(value, "text"):
        return _normalize_value(value.text)
    if hasattr(value, "value"):
        return _normalize_value(value.value)
    return str(value).strip()


def _first_existing(tags, keys: list[str]) -> str:
    if not tags:
        return ""
    for key in keys:
        if key in tags:
            value = _normalize_value(tags.get(key))
            if value:
                return value
    return ""


def _extract_year(value: str) -> str:
    if not value:
        return ""
    match = re.search(r"(19|20)\d{2}", value)
    if not match:
        return ""
    return match.group(0)


def _extract_bpm(value: str) -> str:
    if not value:
        return ""
    match = re.search(r"\d+(?:[.,]\d+)?", value)
    if not match:
        return ""
    return match.group(0).replace(",", ".")


def _read_easy_tags(file_path: str | Path) -> dict:
    file_obj = File(file_path, easy=True)
    tags = getattr(file_obj, "tags", None) if file_obj is not None else None

    return {
        "tag_artist": _first_existing(tags, ["artist", "albumartist", "performer", "composer"]),
        "tag_title": _first_existing(tags, ["title"]),
        "tag_year_raw": _first_existing(tags, ["date", "year", "originaldate"]),
        "tag_genre": _first_existing(tags, ["genre"]),
        "tag_mood": _first_existing(tags, ["mood"]),
        "tag_initial_key": _first_existing(tags, ["initialkey", "key"]),
        "tag_bpm_raw": _first_existing(tags, ["bpm", "tempo"]),
    }


def _read_raw_tags(file_path: str | Path) -> dict:
    file_obj = File(file_path, easy=False)
    tags = getattr(file_obj, "tags", None) if file_obj is not None else None

    return {
        "tag_artist": _first_existing(
            tags,
            [
                "TPE1",
                "TPE2",
                "\xa9ART",
                "artist",
                "albumartist",
                "AUTHOR",
            ],
        ),
        "tag_title": _first_existing(
            tags,
            [
                "TIT2",
                "\xa9nam",
                "title",
                "TITLE",
            ],
        ),
        "tag_year_raw": _first_existing(
            tags,
            [
                "TDRC",
                "TYER",
                "\xa9day",
                "date",
                "year",
                "YEAR",
            ],
        ),
        "tag_genre": _first_existing(
            tags,
            [
                "TCON",
                "\xa9gen",
                "genre",
                "GENRE",
            ],
        ),
        "tag_mood": _first_existing(
            tags,
            [
                "TMOO",
                "MOOD",
                "mood",
            ],
        ),
        "tag_initial_key": _first_existing(
            tags,
            [
                "TKEY",
                "INITIALKEY",
                "initialkey",
                "key",
            ],
        ),
        "tag_bpm_raw": _first_existing(
            tags,
            [
                "TBPM",
                "bpm",
                "BPM",
                "tempo",
                "TEMPO",
            ],
        ),
    }


def read_metadata(file_path: str | Path) -> dict:
    easy_data = _read_easy_tags(file_path)
    raw_data = _read_raw_tags(file_path)

    tag_artist = easy_data["tag_artist"] or raw_data["tag_artist"]
    tag_title = easy_data["tag_title"] or raw_data["tag_title"]
    raw_year = easy_data["tag_year_raw"] or raw_data["tag_year_raw"]
    tag_year = _extract_year(raw_year)
    tag_genre = easy_data["tag_genre"] or raw_data["tag_genre"]
    tag_mood = easy_data["tag_mood"] or raw_data["tag_mood"]
    tag_initial_key = easy_data["tag_initial_key"] or raw_data["tag_initial_key"]
    raw_bpm = easy_data["tag_bpm_raw"] or raw_data["tag_bpm_raw"]
    tag_bpm = _extract_bpm(raw_bpm)

    return {
        "tag_artist": tag_artist,
        "tag_title": tag_title,
        "tag_year": tag_year,
        "tag_genre": tag_genre,
        "tag_mood": tag_mood,
        "tag_initial_key": tag_initial_key,
        "tag_bpm": tag_bpm,
    }
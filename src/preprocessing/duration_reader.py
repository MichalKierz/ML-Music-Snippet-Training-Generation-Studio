from pathlib import Path

from mutagen import File


def read_duration_seconds(file_path: str | Path) -> float:
    file_obj = File(file_path)
    if file_obj is None:
        return 0.0

    info = getattr(file_obj, "info", None)
    if info is None:
        return 0.0

    length = getattr(info, "length", 0.0)
    if length is None:
        return 0.0

    return float(length)
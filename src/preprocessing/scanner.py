from pathlib import Path


def scan_audio_files(root_folder: str | Path, audio_extensions: list[str]) -> list[Path]:
    root = Path(root_folder)
    normalized_extensions = {ext.lower() for ext in audio_extensions}
    files = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in normalized_extensions:
            continue
        files.append(path)

    return sorted(files, key=lambda item: str(item).lower())
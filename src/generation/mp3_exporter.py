from pathlib import Path
import re
import shutil

import numpy as np
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3NoHeaderError
from mutagen.mp3 import MP3
from pydub import AudioSegment
from pydub.exceptions import CouldntEncodeError

from src.generation.prompt_mapper import parse_time_string_to_seconds


def sanitize_filename_part(value: str, fallback: str) -> str:
    text = str(value).strip()
    if not text:
        text = fallback
    text = re.sub(r'[\\/:*?"<>|]+', "_", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text[:120].strip()
    return text or fallback


def format_time_for_filename(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes = total // 60
    secs = total % 60
    return f"{minutes:02d}-{secs:02d}"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def estimate_window_from_relative_position(
    resolved_position: float,
    duration_sec: float,
    reference_track_duration_sec: float,
) -> tuple[float, float]:
    reference_track_duration_sec = max(float(duration_sec), float(reference_track_duration_sec))
    available_start_range = max(0.0, float(reference_track_duration_sec) - float(duration_sec))
    estimated_start_sec = clamp01(resolved_position) * available_start_range
    estimated_end_sec = estimated_start_sec + float(duration_sec)
    return estimated_start_sec, estimated_end_sec


def build_output_file_name(
    artist: str,
    title: str,
    position_mode: str,
    start_time: str,
    duration_sec: float,
    resolved_position: float | None = None,
    reference_track_duration_sec: float = 240.0,
) -> str:
    safe_artist = sanitize_filename_part(artist, "Unknown Artist")
    safe_title = sanitize_filename_part(title, "Untitled")

    mode = str(position_mode).strip()

    if mode == "Start Time":
        start_seconds = parse_time_string_to_seconds(start_time)
        end_seconds = start_seconds + float(duration_sec)
    else:
        resolved = 0.5 if resolved_position is None else clamp01(resolved_position)
        start_seconds, end_seconds = estimate_window_from_relative_position(
            resolved_position=resolved,
            duration_sec=duration_sec,
            reference_track_duration_sec=reference_track_duration_sec,
        )

    start_label = format_time_for_filename(start_seconds)
    end_label = format_time_for_filename(end_seconds)

    return f"{safe_artist} - {safe_title} [{start_label} - {end_label}].mp3"


def ensure_unique_file_path(output_folder: str | Path, file_name: str) -> Path:
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate = output_dir / file_name
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix

    counter = 2
    while True:
        numbered = output_dir / f"{stem} ({counter}){suffix}"
        if not numbered.exists():
            return numbered
        counter += 1


def resolve_ffmpeg_binary() -> str:
    candidates = []

    current_converter = getattr(AudioSegment, "converter", None)
    if current_converter:
        candidates.append(str(current_converter))

    candidates.extend(["ffmpeg", "ffmpeg.exe", "avconv", "avconv.exe"])

    checked = set()

    for candidate in candidates:
        if not candidate or candidate in checked:
            continue
        checked.add(candidate)

        candidate_path = Path(candidate)
        if candidate_path.exists():
            resolved = str(candidate_path.resolve())
            AudioSegment.converter = resolved
            return resolved

        resolved = shutil.which(candidate)
        if resolved:
            AudioSegment.converter = resolved
            return resolved

    raise RuntimeError("FFmpeg was not found. Install ffmpeg and add it to PATH.")


def write_mp3_tags(
    output_path: str | Path,
    artist: str,
    title: str,
    year,
    genre: str,
):
    path = str(output_path)

    try:
        tags = EasyID3(path)
    except ID3NoHeaderError:
        audio = MP3(path)
        audio.add_tags()
        audio.save()
        tags = EasyID3(path)

    if str(artist).strip():
        tags["artist"] = [str(artist).strip()]
    if str(title).strip():
        tags["title"] = [str(title).strip()]
    if str(genre).strip():
        tags["genre"] = [str(genre).strip()]

    try:
        year_int = int(year)
    except Exception:
        year_int = 0

    if year_int > 0:
        tags["date"] = [str(year_int)]

    tags.save()


def export_waveform_to_mp3(
    waveform,
    sample_rate: int,
    output_folder: str | Path,
    artist: str,
    title: str,
    year,
    genre: str,
    position_mode: str,
    start_time: str,
    duration_sec: float,
    resolved_position: float | None = None,
    reference_track_duration_sec: float = 240.0,
    bitrate: str = "192k",
) -> str:
    resolve_ffmpeg_binary()

    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().numpy()

    waveform = np.asarray(waveform, dtype=np.float32)

    if waveform.ndim == 2 and waveform.shape[0] == 1:
        waveform = waveform[0]
    elif waveform.ndim == 2:
        waveform = waveform.mean(axis=0)
    elif waveform.ndim != 1:
        waveform = waveform.reshape(-1)

    waveform = np.clip(waveform, -1.0, 1.0)
    pcm = (waveform * 32767.0).astype(np.int16)

    audio = AudioSegment(
        data=pcm.tobytes(),
        sample_width=2,
        frame_rate=int(sample_rate),
        channels=1,
    )

    file_name = build_output_file_name(
        artist=artist,
        title=title,
        position_mode=position_mode,
        start_time=start_time,
        duration_sec=duration_sec,
        resolved_position=resolved_position,
        reference_track_duration_sec=reference_track_duration_sec,
    )
    output_path = ensure_unique_file_path(output_folder, file_name)

    tags = {}
    if str(artist).strip():
        tags["artist"] = str(artist).strip()
    if str(title).strip():
        tags["title"] = str(title).strip()
    if str(genre).strip():
        tags["genre"] = str(genre).strip()

    try:
        year_int = int(year)
    except Exception:
        year_int = 0

    if year_int > 0:
        tags["date"] = str(year_int)

    try:
        audio.export(output_path, format="mp3", bitrate=bitrate, tags=tags or None)
        write_mp3_tags(
            output_path=output_path,
            artist=artist,
            title=title,
            year=year,
            genre=genre,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "MP3 export failed because FFmpeg was not found. Install ffmpeg and add it to PATH."
        ) from exc
    except CouldntEncodeError as exc:
        raise RuntimeError(
            "MP3 export failed. FFmpeg is installed but encoding did not succeed."
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"MP3 export failed: {exc}") from exc

    return str(output_path)
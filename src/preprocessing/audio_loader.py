from contextlib import contextmanager
from pathlib import Path
import os
import random
import shutil
import sys
import threading

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from pydub import AudioSegment


@contextmanager
def suppress_native_stderr():
    try:
        stderr_fd = sys.stderr.fileno()
    except Exception:
        yield
        return

    try:
        sys.stderr.flush()
    except Exception:
        pass

    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        try:
            os.dup2(saved_fd, stderr_fd)
        finally:
            os.close(saved_fd)
            os.close(devnull_fd)


def _to_tensor(waveform: np.ndarray) -> torch.Tensor:
    array = np.asarray(waveform, dtype=np.float32)

    if array.ndim == 1:
        array = array[np.newaxis, :]
    elif array.ndim == 2:
        pass
    else:
        array = array.reshape(1, -1)

    return torch.from_numpy(array.astype(np.float32, copy=False))


def _ensure_mono_tensor(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def _ensure_mono_numpy(waveform: np.ndarray) -> np.ndarray:
    if waveform.ndim == 1:
        return waveform
    if waveform.ndim == 2:
        return waveform.mean(axis=0)
    return waveform.reshape(-1)


def _resample_numpy(waveform: np.ndarray, source_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if int(source_sample_rate) == int(target_sample_rate):
        return waveform.astype(np.float32, copy=False)

    if waveform.ndim == 1:
        resampled = librosa.resample(
            waveform.astype(np.float32, copy=False),
            orig_sr=int(source_sample_rate),
            target_sr=int(target_sample_rate),
        )
        return resampled.astype(np.float32, copy=False)

    channels = []
    for channel in waveform:
        resampled_channel = librosa.resample(
            channel.astype(np.float32, copy=False),
            orig_sr=int(source_sample_rate),
            target_sr=int(target_sample_rate),
        )
        channels.append(resampled_channel.astype(np.float32, copy=False))

    return np.stack(channels, axis=0).astype(np.float32, copy=False)


def _resolve_ffmpeg_binary() -> str:
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


def _load_with_librosa(file_path: str | Path, sample_rate: int, mono: bool):
    with suppress_native_stderr():
        waveform, _ = librosa.load(
            str(file_path),
            sr=int(sample_rate),
            mono=bool(mono),
        )

    if mono:
        waveform = _ensure_mono_numpy(np.asarray(waveform, dtype=np.float32))
    else:
        waveform = np.asarray(waveform, dtype=np.float32)

    return _to_tensor(waveform), "librosa"


def _load_with_soundfile(file_path: str | Path, sample_rate: int, mono: bool):
    with suppress_native_stderr():
        waveform, source_sample_rate = sf.read(str(file_path), always_2d=True, dtype="float32")

    waveform = waveform.T.astype(np.float32, copy=False)

    if mono:
        waveform = _ensure_mono_numpy(waveform)

    waveform = _resample_numpy(
        waveform=np.asarray(waveform, dtype=np.float32),
        source_sample_rate=int(source_sample_rate),
        target_sample_rate=int(sample_rate),
    )

    if mono:
        waveform = _ensure_mono_numpy(waveform)

    return _to_tensor(waveform), "soundfile"


def _load_with_pydub(file_path: str | Path, sample_rate: int, mono: bool):
    _resolve_ffmpeg_binary()

    with suppress_native_stderr():
        audio = AudioSegment.from_file(str(file_path))

    audio = audio.set_frame_rate(int(sample_rate))

    if mono:
        audio = audio.set_channels(1)

    sample_width = int(audio.sample_width)
    channels = int(audio.channels)

    samples = np.array(audio.get_array_of_samples())

    if channels > 1:
        samples = samples.reshape((-1, channels)).T.astype(np.float32, copy=False)
    else:
        samples = samples.astype(np.float32, copy=False)

    scale = float(1 << (8 * sample_width - 1))
    samples = samples / scale

    if mono:
        samples = _ensure_mono_numpy(samples)

    return _to_tensor(samples), "pydub"


def _load_with_torchaudio(file_path: str | Path, sample_rate: int, mono: bool):
    with suppress_native_stderr():
        waveform, source_sample_rate = torchaudio.load(str(file_path))

    if mono:
        waveform = _ensure_mono_tensor(waveform)

    if int(source_sample_rate) != int(sample_rate):
        waveform = torchaudio.functional.resample(
            waveform,
            int(source_sample_rate),
            int(sample_rate),
        )

    return waveform.float(), "torchaudio"


def _run_with_timeout(loader, timeout_sec: float, **kwargs):
    result = {}
    error = {}

    def target():
        try:
            result["value"] = loader(**kwargs)
        except Exception as exc:
            error["value"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=float(timeout_sec))

    if thread.is_alive():
        raise TimeoutError(f"{loader.__name__} timed out after {timeout_sec} seconds")

    if "value" in error:
        raise error["value"]

    return result["value"]


def load_audio_with_fallback(file_path: str | Path, sample_rate: int, mono: bool = True, timeout_sec: float = 20.0):
    errors = []

    loaders = [
        _load_with_librosa,
        _load_with_soundfile,
        _load_with_pydub,
        _load_with_torchaudio,
    ]

    for loader in loaders:
        try:
            waveform, backend = _run_with_timeout(
                loader,
                timeout_sec=timeout_sec,
                file_path=file_path,
                sample_rate=sample_rate,
                mono=mono,
            )
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.numel() == 0:
                raise RuntimeError(f"{backend} returned empty audio")
            return waveform.contiguous().float(), backend
        except Exception as exc:
            errors.append(f"{loader.__name__}: {exc}")

    error_text = "\n".join(errors)
    raise RuntimeError(f"All audio loading backends failed for file: {file_path}\n{error_text}")
from math import floor


def build_track_chunks(track_duration: float, chunk_length_sec: float, chunk_stride_sec: float) -> list[dict]:
    duration = float(track_duration)
    chunk_length = float(chunk_length_sec)
    chunk_stride = float(chunk_stride_sec)

    if duration <= 0 or chunk_length <= 0 or chunk_stride <= 0:
        return []

    if duration < chunk_length:
        return []

    chunk_count = floor((duration - chunk_length) / chunk_stride) + 1
    chunks = []

    for chunk_index in range(chunk_count):
        start_sec = round(chunk_index * chunk_stride, 6)
        end_sec = round(start_sec + chunk_length, 6)
        relative_position = 0.0 if duration <= 0 else max(0.0, min(1.0, start_sec / duration))

        chunks.append(
            {
                "chunk_index": int(chunk_index),
                "chunk_count": int(chunk_count),
                "chunk_start_sec": start_sec,
                "chunk_end_sec": end_sec,
                "relative_position": round(relative_position, 6),
            }
        )

    return chunks
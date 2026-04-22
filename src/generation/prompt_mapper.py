def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def parse_time_string_to_seconds(value: str) -> float:
    text = str(value).strip()
    if not text:
        return 0.0

    parts = text.split(":")
    try:
        if len(parts) == 1:
            return max(0.0, float(parts[0]))
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return max(0.0, minutes * 60.0 + seconds)
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return max(0.0, hours * 3600.0 + minutes * 60.0 + seconds)
    except Exception:
        return 0.0

    return 0.0


def section_to_relative_position(section: str) -> float:
    mapping = {
        "Intro": 0.05,
        "Early": 0.25,
        "Middle": 0.50,
        "Late": 0.75,
        "Outro": 0.95,
    }
    return float(mapping.get(str(section).strip(), 0.50))


def resolve_relative_position(
    position_mode: str,
    start_time: str,
    relative_position: float,
    section: str,
    reference_track_duration_sec: float = 240.0,
) -> float:
    mode = str(position_mode).strip()

    if mode == "Start Time":
        seconds = parse_time_string_to_seconds(start_time)
        if reference_track_duration_sec <= 0:
            return 0.0
        return clamp01(seconds / float(reference_track_duration_sec))

    if mode == "Relative Position":
        return clamp01(float(relative_position))

    if mode == "Section":
        return section_to_relative_position(section)

    return 0.5


def build_prompt_row(
    artist: str,
    title: str,
    year,
    genre: str,
    relative_position: float,
    mood: str = "",
    initial_key: str = "",
    bpm=0,
) -> dict:
    return {
        "artist": artist,
        "title": title,
        "year": year,
        "genre": genre,
        "relative_position": float(relative_position),
        "mood": mood,
        "initial_key": initial_key,
        "bpm": bpm,
    }
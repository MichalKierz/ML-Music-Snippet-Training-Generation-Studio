import random


MAX_SEED = 2**32 - 1


def make_seed() -> str:
    return str(random.randint(0, MAX_SEED))


def is_valid_seed(value: str) -> bool:
    text = str(value).strip()
    if not text:
        return False
    try:
        numeric = int(text)
    except Exception:
        return False
    return 0 <= numeric <= MAX_SEED


def format_duration_hms(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
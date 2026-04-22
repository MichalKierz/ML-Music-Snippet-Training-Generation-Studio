import json
from pathlib import Path


def _normalize_config_key(path: Path) -> str:
    stem = path.stem

    if stem.endswith("_defaults"):
        stem = stem[:-9]

    if stem.endswith("_config"):
        stem = stem[:-7]

    return stem


def load_config(path: str | Path) -> dict:
    config_path = Path(path)

    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_all_configs(config_dir: str | Path) -> dict:
    config_root = Path(config_dir)

    if not config_root.exists():
        raise FileNotFoundError(f"Config directory not found: {config_root}")

    configs = {}

    for path in sorted(config_root.glob("*.json")):
        key = _normalize_config_key(path)
        configs[key] = load_config(path)

    required_keys = {
        "app",
        "preprocess",
        "training",
        "generate",
        "token_model",
    }

    missing = sorted(required_keys - set(configs.keys()))
    if missing:
        raise RuntimeError(f"Missing required config files: {', '.join(missing)}")

    return configs
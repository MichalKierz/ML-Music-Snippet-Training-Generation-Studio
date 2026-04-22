import sys
from pathlib import Path


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def ensure_runtime_directories(base_dir: Path, app_config: dict) -> dict[str, Path]:
    runtime_dirs = {}
    default_directories = app_config.get("default_directories", {})

    for key, folder_name in default_directories.items():
        folder_path = (base_dir / folder_name).resolve()
        folder_path.mkdir(parents=True, exist_ok=True)
        runtime_dirs[key] = folder_path

    return runtime_dirs


def get_runtime_dir(runtime_dirs: dict[str, Path], key: str) -> Path:
    if key not in runtime_dirs:
        raise KeyError(f"Missing runtime directory key: {key}")
    return runtime_dirs[key]


def get_user_settings_path(base_dir: Path) -> Path:
    return base_dir / "user_settings.json"
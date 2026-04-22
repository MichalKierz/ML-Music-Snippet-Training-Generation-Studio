from dataclasses import dataclass
from pathlib import Path

from src.core.settings_manager import SettingsManager


@dataclass
class RuntimeContext:
    base_dir: Path
    configs: dict
    runtime_dirs: dict[str, Path]
    settings: SettingsManager
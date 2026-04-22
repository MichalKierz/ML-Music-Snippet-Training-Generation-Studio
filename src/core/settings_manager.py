import json
from copy import deepcopy
from pathlib import Path


def deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class SettingsManager:
    def __init__(self, path: Path, defaults: dict):
        self.path = path
        self.defaults = deepcopy(defaults)
        self.data = {}
        self.load()

    def load(self) -> dict:
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                stored = json.load(handle)
        else:
            stored = {}
        self.data = deep_merge(self.defaults, stored)
        self.save()
        return self.data

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(self.data, handle, indent=2, ensure_ascii=False)

    def get(self, section: str, key: str | None = None, default=None):
        section_data = self.data.get(section, {})
        if key is None:
            return section_data
        return section_data.get(key, default)

    def set(self, section: str, key: str, value):
        if section not in self.data or not isinstance(self.data[section], dict):
            self.data[section] = {}
        self.data[section][key] = value
        self.save()

    def set_section(self, section: str, values: dict):
        self.data[section] = values
        self.save()

    def update_section(self, section: str, values: dict):
        if section not in self.data or not isinstance(self.data[section], dict):
            self.data[section] = {}
        self.data[section].update(values)
        self.save()

    def reset(self):
        self.data = deepcopy(self.defaults)
        self.save()
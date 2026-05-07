"""Config loader. Pulls config.yaml + .env and exposes a typed object."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-(.*?))?\}")


def _resolve_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match) -> str:
            var, default = match.group(1), match.group(2) or ""
            return os.getenv(var, default)
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


class Config:
    """Light wrapper around a nested dict with attribute access."""

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, item: str) -> Any:
        if item.startswith("_"):
            raise AttributeError(item)
        if item not in self._data:
            raise AttributeError(f"Missing config key: {item}")
        v = self._data[item]
        return Config(v) if isinstance(v, dict) else v

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> dict:
        return self._data


def load_config(path: str | Path = "config.yaml") -> Config:
    load_dotenv(override=False)
    raw = Path(path).read_text()
    data = yaml.safe_load(raw)
    data = _resolve_env(data)
    return Config(data)

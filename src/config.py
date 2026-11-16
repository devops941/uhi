"""Configuration loading helpers.

A single ``Config`` object wraps ``config.yaml`` and gives dotted access to
nested keys so that no module has to hard-code thresholds, paths or weights.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class Config:
    """Thin, dict-backed configuration object with dotted lookup."""

    def __init__(self, data: dict[str, Any], path: Path | None = None):
        self._data = data
        self.path = path
        self.root = PROJECT_ROOT

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not cfg_path.exists():
            raise FileNotFoundError(f"config file not found: {cfg_path}")
        with open(cfg_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls(data, cfg_path)

    # ------------------------------------------------------------------
    # access
    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """Return a nested value using ``a.b.c`` notation."""
        node: Any = self._data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, key: str) -> Any:
        sentinel = object()
        value = self.get(key, sentinel)
        if value is sentinel:
            raise KeyError(key)
        return value

    def __contains__(self, key: str) -> bool:
        sentinel = object()
        return self.get(key, sentinel) is not sentinel

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def with_overrides(self, overrides: dict[str, Any]) -> Config:
        """Return a new Config with dotted-key overrides applied."""
        data = self.as_dict()
        for dotted, value in overrides.items():
            node = data
            parts = dotted.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return Config(data, self.path)

    # ------------------------------------------------------------------
    # path helpers
    # ------------------------------------------------------------------
    def path_for(self, key: str, create: bool = False) -> Path:
        """Resolve a ``paths.*`` entry against the project root."""
        raw = self.get(f"paths.{key}")
        if raw is None:
            raise KeyError(f"paths.{key} is not defined in config")
        resolved = (self.root / raw).resolve()
        if create:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved


def load_config(path: str | Path | None = None) -> Config:
    return Config.load(path)

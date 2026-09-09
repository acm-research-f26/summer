"""Config loading. Every threshold in the project comes from config/default.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


class Config(Mapping):
    """Read-only nested-dict wrapper with dotted-path access.

    >>> cfg = load_config()
    >>> cfg["figo.intramural_threshold_pct"]
    50.0
    """

    def __init__(self, data: dict[str, Any], path: Path | None = None):
        self._data = data
        self.path = path

    def __getitem__(self, key: str) -> Any:
        node: Any = self._data
        for part in key.split("."):
            if not isinstance(node, Mapping) or part not in node:
                raise KeyError(f"no config key {key!r} (failed at {part!r})")
            node = node[part]
        return node

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Config(path={self.path}, top_level={sorted(self._data)})"

    def resolve_path(self, key: str) -> Path:
        """Resolve a config value that names a repo-relative path."""
        return (REPO_ROOT / str(self[key])).resolve()


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p) as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"config at {p} did not parse to a mapping")
    return Config(data, path=p)

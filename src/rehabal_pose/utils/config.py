from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

__all__ = ["load_yaml", "resolve_path", "get", "interpolate_env"]

_ENV_PATTERN = re.compile(r"\$\{ENV:([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


def interpolate_env(value: str) -> str:
    def _replace(match: "re.Match[str]") -> str:
        var = match.group(1)
        default = match.group(2)
        if var in os.environ:
            return os.environ[var]
        if default is not None:
            return default
        raise KeyError(
            f"Environment variable {var!r} is not set and no default was provided"
        )

    return _ENV_PATTERN.sub(_replace, value)


def _interpolate_tree(obj: Any) -> Any:
    if isinstance(obj, str):
        return interpolate_env(obj)
    if isinstance(obj, dict):
        return {k: _interpolate_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate_tree(v) for v in obj]
    return obj


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"YAML config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(
            f"Top-level YAML document must be a mapping, got {type(data).__name__}"
        )
    return _interpolate_tree(data)


def resolve_path(p: str | os.PathLike[str]) -> Path:
    raw = os.fspath(p)
    expanded = os.path.expanduser(os.path.expandvars(raw))
    return Path(expanded).resolve()


def get(cfg: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted_key.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return default
    return node

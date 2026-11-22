"""Configuration loading utilities.

Provides YAML loading with environment-variable interpolation and small
path / nested-access helpers. No heavy dependencies are imported here so the
test suite can import this module freely.

Environment interpolation syntax (inside string values):
    ${ENV:VAR}            -> value of os.environ["VAR"]  (error if unset)
    ${ENV:VAR:default}    -> value of os.environ["VAR"] or "default" if unset
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

__all__ = ["load_yaml", "resolve_path", "get", "interpolate_env"]

# Matches ${ENV:VAR} or ${ENV:VAR:default}. The default may contain any
# character except the closing brace.
_ENV_PATTERN = re.compile(r"\$\{ENV:([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


def interpolate_env(value: str) -> str:
    """Interpolate ``${ENV:VAR}`` / ``${ENV:VAR:default}`` tokens in *value*.

    Parameters
    ----------
    value:
        A string possibly containing one or more env tokens.

    Returns
    -------
    str
        The interpolated string.

    Raises
    ------
    KeyError
        If a referenced environment variable is unset and no default was
        provided.
    """

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
    """Recursively interpolate env tokens in all string leaves of *obj*."""
    if isinstance(obj, str):
        return interpolate_env(obj)
    if isinstance(obj, dict):
        return {k: _interpolate_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate_tree(v) for v in obj]
    return obj


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML file into a dict with environment interpolation.

    Parameters
    ----------
    path:
        Path to a YAML file.

    Returns
    -------
    dict
        Parsed mapping (empty dict if the file is empty).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    TypeError
        If the top-level YAML document is not a mapping.
    """
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
    """Expand ``~`` and environment variables in *p* and return an absolute Path.

    This does not require the path to exist.
    """
    raw = os.fspath(p)
    expanded = os.path.expanduser(os.path.expandvars(raw))
    return Path(expanded).resolve()


def get(cfg: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Fetch a nested value from *cfg* using a dotted path.

    Example
    -------
    >>> get({"a": {"b": {"c": 1}}}, "a.b.c")
    1
    >>> get({"a": {}}, "a.b.c", default=42)
    42

    Parameters
    ----------
    cfg:
        The configuration mapping.
    dotted_key:
        Keys joined by ``.`` (e.g. ``"data.aihub219.root"``).
    default:
        Returned if any key in the path is missing or a non-mapping is
        encountered before the path is exhausted.
    """
    node: Any = cfg
    for part in dotted_key.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return default
    return node

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .config import load_yaml as read_yaml

__all__ = [
    "read_json",
    "write_json",
    "ensure_dir",
    "read_yaml",
    "atomic_write_text",
    "atomic_write_bytes",
]


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | os.PathLike[str]) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def atomic_write_text(
    path: str | os.PathLike[str], text: str, encoding: str = "utf-8"
) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    fd, tmp_name = tempfile.mkstemp(dir=str(p.parent), prefix=p.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as fh:
            fh.write(text)
        os.replace(tmp_name, p)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return p


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    fd, tmp_name = tempfile.mkstemp(dir=str(p.parent), prefix=p.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp_name, p)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return p


def write_json(
    path: str | os.PathLike[str],
    obj: Any,
    *,
    indent: int = 2,
    sort_keys: bool = False,
    atomic: bool = True,
) -> Path:
    text = json.dumps(obj, ensure_ascii=False, indent=indent, sort_keys=sort_keys)
    if atomic:
        return atomic_write_text(path, text)
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return p

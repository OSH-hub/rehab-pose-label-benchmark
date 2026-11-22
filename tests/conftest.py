"""Pytest configuration for the RehabAL-Pose test suite.

Ensures the src-layout package ``rehabal_pose`` is importable when the test
suite is run without an editable install. ``pyproject.toml`` already sets
``[tool.pytest.ini_options] pythonpath = ["src"]``; this file is a belt-and-
suspenders fallback so ``pytest`` works from any working directory and under
runners that ignore ``pythonpath``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repository root is the parent of this ``tests`` directory.
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"

if _SRC.is_dir():
    src_str = str(_SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

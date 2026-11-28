from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"

if _SRC.is_dir():
    src_str = str(_SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

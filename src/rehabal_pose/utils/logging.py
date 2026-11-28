from __future__ import annotations

import logging
import sys

__all__ = ["get_logger"]

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    has_stream_handler = any(
        isinstance(h, logging.StreamHandler) for h in logger.handlers
    )
    if not has_stream_handler:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(fmt=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)
        )
        logger.addHandler(handler)
    return logger

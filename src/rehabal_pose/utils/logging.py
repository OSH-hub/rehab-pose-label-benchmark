"""Logging helpers.

A single ``get_logger`` factory that returns a configured stdlib logger with
exactly one stream handler (no duplicate handlers on repeated calls).
"""

from __future__ import annotations

import logging
import sys

__all__ = ["get_logger"]

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger named *name*.

    The logger gets a single :class:`logging.StreamHandler` writing to stdout.
    Calling this repeatedly with the same name does not add duplicate handlers.

    Parameters
    ----------
    name:
        Logger name (typically ``__name__``).
    level:
        Logging level (default :data:`logging.INFO`).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Do not propagate to the root logger to avoid duplicated lines if the
    # application also configures the root logger.
    logger.propagate = False

    # Only attach our handler once.
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

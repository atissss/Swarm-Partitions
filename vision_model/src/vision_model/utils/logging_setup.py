"""Centralized logging configuration.

Every package should use `logging.getLogger(__name__)` and never `print`.
Call `configure_logging()` once, from a CLI entry point, to set format/level
for the whole run.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure the root logger once for the process.

    Args:
        level: Logging level, e.g. `logging.INFO` or `"DEBUG"`. Subsequent
            calls after the first are no-ops so library code can call this
            defensively without clobbering a CLI's chosen level.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring root logging if needed.

    Args:
        name: Typically `__name__` of the calling module.
    """
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)

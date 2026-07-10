"""Shared CLI error-handling wrapper.

Every `cli.*` entry point wraps its `main()` body with this so a
`VisionModelError` (misconfiguration, missing files, a bad geo-tag, ...)
prints a clean one-line error and exits non-zero, instead of a raw Python
traceback — while still letting genuinely unexpected exceptions surface
with their full traceback for debugging.
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable
from typing import TypeVar

from vision_model.utils.exceptions import VisionModelError
from vision_model.utils.logging_setup import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., None])


def cli_entrypoint(func: F) -> F:  # noqa: UP047 - TypeVar kept for tooling/readability
    """Decorator: catch `VisionModelError` and exit(1) with a clean message.

    Args:
        func: A CLI `main(cfg)` function (typically already wrapped by
            `@hydra.main`).

    Returns:
        The wrapped function.
    """

    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> None:
        try:
            func(*args, **kwargs)
        except VisionModelError as exc:
            logger.error("%s: %s", type(exc).__name__, exc)
            sys.exit(1)

    return wrapper  # type: ignore[return-value]

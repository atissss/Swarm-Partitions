from __future__ import annotations

import pytest

from vision_model.cli._common import cli_entrypoint
from vision_model.utils.exceptions import ConfigError, DatasetError


class TestCliEntrypoint:
    def test_passes_through_return_value_and_args(self) -> None:
        calls = []

        @cli_entrypoint
        def fn(a, b, *, c):
            calls.append((a, b, c))

        fn(1, 2, c=3)
        assert calls == [(1, 2, 3)]

    def test_vision_model_error_exits_instead_of_raising(self) -> None:
        @cli_entrypoint
        def fn():
            raise ConfigError("bad config")

        with pytest.raises(SystemExit) as exc_info:
            fn()
        assert exc_info.value.code == 1

    def test_subclass_of_vision_model_error_is_caught(self) -> None:
        @cli_entrypoint
        def fn():
            raise DatasetError("bad dataset")

        with pytest.raises(SystemExit):
            fn()

    def test_unrelated_exception_propagates(self) -> None:
        @cli_entrypoint
        def fn():
            raise ValueError("something else entirely")

        with pytest.raises(ValueError):
            fn()

    def test_no_error_does_not_exit(self) -> None:
        @cli_entrypoint
        def fn():
            return "unreachable via decorator, but shouldn't raise"

        fn()  # should simply not raise

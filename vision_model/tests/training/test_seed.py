from __future__ import annotations

import random

import numpy as np

from vision_model.training.seed import set_seed


class TestSetSeed:
    def test_python_random_is_reproducible(self) -> None:
        set_seed(123)
        a = [random.random() for _ in range(5)]
        set_seed(123)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_numpy_random_is_reproducible(self) -> None:
        set_seed(7)
        a = np.random.rand(5)
        set_seed(7)
        b = np.random.rand(5)
        assert np.array_equal(a, b)

    def test_different_seeds_diverge(self) -> None:
        set_seed(1)
        a = [random.random() for _ in range(5)]
        set_seed(2)
        b = [random.random() for _ in range(5)]
        assert a != b

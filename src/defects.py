"""Defect and residual utilities for generated Q prefixes."""

from __future__ import annotations

import numpy as np


def first_differences(values: np.ndarray) -> np.ndarray:
    """Return consecutive differences."""
    return np.diff(values.astype(np.int64))


def centered_defects(values: np.ndarray) -> np.ndarray:
    """Return deviations from the running mean as a simple defect proxy."""
    if values.size == 0:
        return np.array([], dtype=float)
    indices = np.arange(1, values.size + 1, dtype=float)
    trend = np.cumsum(values) / indices
    return values.astype(float) - trend


def dyadic_blocks(values: np.ndarray, min_power: int = 6, max_power: int | None = None) -> dict[int, np.ndarray]:
    """Split a prefix into dyadic blocks [2^k, 2^(k+1))."""
    if max_power is None:
        max_power = int(np.floor(np.log2(max(values.size, 1)))) - 1
    blocks: dict[int, np.ndarray] = {}
    for power in range(min_power, max_power + 1):
        start = 2**power
        stop = min(2 ** (power + 1), values.size)
        if start < stop:
            blocks[power] = values[start:stop]
    return blocks


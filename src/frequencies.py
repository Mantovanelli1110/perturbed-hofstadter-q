"""Frequency distribution helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def frequency_table(values: np.ndarray) -> pd.DataFrame:
    """Return value counts sorted by value."""
    unique, counts = np.unique(values, return_counts=True)
    return pd.DataFrame({"value": unique, "count": counts, "frequency": counts / values.size})


def block_frequency_tables(blocks: dict[int, np.ndarray]) -> dict[int, pd.DataFrame]:
    """Return frequency tables for dyadic blocks."""
    return {power: frequency_table(block) for power, block in blocks.items()}


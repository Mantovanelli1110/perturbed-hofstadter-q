"""Fourier diagnostics for dyadic profiles."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq


def normalized_spectrum(values: np.ndarray) -> pd.DataFrame:
    """Return normalized one-sided Fourier power spectrum."""
    if values.size < 2:
        return pd.DataFrame({"frequency": [], "power": []})
    centered = values.astype(float) - np.mean(values)
    amplitudes = np.abs(rfft(centered))
    power = amplitudes**2
    total = power.sum()
    if total > 0:
        power = power / total
    return pd.DataFrame({"frequency": rfftfreq(values.size), "power": power})


def dominant_frequency(values: np.ndarray) -> tuple[float, float]:
    """Return the nonzero frequency with largest normalized power."""
    spectrum = normalized_spectrum(values)
    if spectrum.empty or len(spectrum) <= 1:
        return 0.0, 0.0
    row = spectrum.iloc[1:].sort_values("power", ascending=False).iloc[0]
    return float(row["frequency"]), float(row["power"])


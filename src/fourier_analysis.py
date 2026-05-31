"""Fourier diagnostics for dyadic defect profiles.

This module implements the Fourier analysis described in the paper.

Given a generated prefix Q(1), ..., Q(N), it computes the dyadic defect

    R(n) = Q(2n) - 2 Q(n),

extracts dyadic blocks

    R_k(j) = R(2^k + j),  0 <= j < 2^k,

standardizes each block by subtracting its empirical mean and dividing by its
empirical standard deviation, and computes low-frequency Fourier power
fractions.

No windowing is applied.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def dyadic_defect(q_values: np.ndarray) -> np.ndarray:
    """Return R(n)=Q(2n)-2Q(n) for all n with 2n <= N.

    The input array stores Q(1), ..., Q(N) in 0-indexed form:
    q_values[i] = Q(i+1).

    The output array stores R(1), ..., R(floor(N/2)) in 0-indexed form.
    """
    q = np.asarray(q_values, dtype=np.int64)
    max_n = q.size // 2

    if max_n == 0:
        return np.array([], dtype=np.int64)

    q_n = q[:max_n]
    q_2n = q[1 : 2 * max_n : 2]

    return q_2n - 2 * q_n


def defect_block(r_values: np.ndarray, k: int) -> np.ndarray:
    """Return R_k(j)=R(2^k+j), 0 <= j < 2^k.

    The input array stores R(1), R(2), ... in 0-indexed form.
    """
    if k < 0:
        raise ValueError("k must be nonnegative.")

    start_n = 2**k
    stop_n = 2 ** (k + 1)

    start_idx = start_n - 1
    stop_idx = stop_n - 1

    if stop_idx > r_values.size:
        raise ValueError(
            f"Need R(n) up to n={stop_n - 1} for k={k}, "
            f"but only have R(1),...,R({r_values.size})."
        )

    return np.asarray(r_values[start_idx:stop_idx], dtype=np.int64)


def standardize(values: np.ndarray) -> np.ndarray:
    """Center and divide by empirical standard deviation with denominator N."""
    arr = np.asarray(values, dtype=np.float64)

    if arr.size == 0:
        raise ValueError("Cannot standardize an empty array.")

    centered = arr - arr.mean()
    std = np.sqrt(np.mean(centered**2))

    if std == 0:
        raise ValueError("Cannot standardize a constant array.")

    return centered / std


def full_power_spectrum(standardized_values: np.ndarray) -> np.ndarray:
    """Return P(l)=|Rhat(l)|^2 using the DFT normalization 1/N.

    This matches the convention

        Rhat(l) = (1/N) sum_j R(j) exp(-2 pi i l j / N).
    """
    arr = np.asarray(standardized_values, dtype=np.float64)
    n = arr.size

    coeffs = np.fft.fft(arr) / n
    return np.abs(coeffs) ** 2


def low_frequency_fraction(power: np.ndarray, L: int) -> float:
    """Return rho(L), excluding the zero mode.

    rho(L) is the fraction of nonzero Fourier power contained in the first L
    positive and negative modes.
    """
    p = np.asarray(power, dtype=np.float64)
    n = p.size

    if n < 2:
        return 0.0
    if L < 1:
        raise ValueError("L must be positive.")
    if 2 * L >= n:
        raise ValueError("Require 2*L < block length.")

    denominator = p[1:].sum()
    if denominator == 0:
        return 0.0

    numerator = p[1 : L + 1].sum() + p[n - L : n].sum()
    return float(numerator / denominator)


def parity_demean_block(block: np.ndarray) -> np.ndarray:
    """Subtract separate means on even and odd sites j mod 2."""
    arr = np.asarray(block, dtype=np.float64).copy()

    arr[0::2] -= arr[0::2].mean()
    arr[1::2] -= arr[1::2].mean()

    return arr


def parity_subprofiles(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return even-site and odd-site subprofiles of a dyadic block.

    For R_k(j), this returns R_k(0), R_k(2), ... and R_k(1), R_k(3), ...
    """
    arr = np.asarray(block)
    return arr[0::2], arr[1::2]


def analyze_fourier(
    q_values: np.ndarray,
    min_k: int = 12,
    max_k: int = 20,
    cutoffs: tuple[int, ...] = (4, 8, 16, 32),
) -> pd.DataFrame:
    """Compute low-frequency power fractions for dyadic defect blocks."""
    r_values = dyadic_defect(q_values)
    rows: list[dict[str, float | int]] = []

    for k in range(min_k, max_k + 1):
        block = defect_block(r_values, k)
        standardized = standardize(block)
        power = full_power_spectrum(standardized)

        row: dict[str, float | int] = {
            "k": k,
            "block_size": 2**k,
        }

        for L in cutoffs:
            row[f"rho_{L}"] = low_frequency_fraction(power, L)

        rows.append(row)

    return pd.DataFrame(rows)


def analyze_parity_checks(
    q_values: np.ndarray,
    ks: tuple[int, ...] = (12, 14, 16, 18, 20),
    cutoffs: tuple[int, ...] = (8, 16),
) -> pd.DataFrame:
    """Compute parity robustness checks for the Fourier spectra."""
    r_values = dyadic_defect(q_values)
    rows: list[dict[str, float | int]] = []

    for k in ks:
        block = defect_block(r_values, k)

        standardized_all = standardize(block)
        power_all = full_power_spectrum(standardized_all)

        parity_demeaned = standardize(parity_demean_block(block))
        power_parity_demeaned = full_power_spectrum(parity_demeaned)

        even_profile, odd_profile = parity_subprofiles(block)
        even_power = full_power_spectrum(standardize(even_profile))
        odd_power = full_power_spectrum(standardize(odd_profile))

        row: dict[str, float | int] = {
            "k": k,
            "block_size": 2**k,
        }

        for L in cutoffs:
            row[f"all_L{L}"] = low_frequency_fraction(power_all, L)
            row[f"parity_demeaned_L{L}"] = low_frequency_fraction(
                power_parity_demeaned, L
            )
            row[f"even_L{L}"] = low_frequency_fraction(even_power, L)
            row[f"odd_L{L}"] = low_frequency_fraction(odd_power, L)

        rows.append(row)

    return pd.DataFrame(rows)


def save_percent_table(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame of fractions as percentages for easier comparison."""
    out = df.copy()

    for col in out.columns:
        if col.startswith("rho_") or "_L" in col:
            out[col] = 100.0 * out[col].astype(float)

    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Fourier diagnostics for dyadic defect profiles."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/Q_4194304.csv"),
        help="CSV file containing Q(1),...,Q(N) as one integer per row.",
    )
    parser.add_argument("--min-k", type=int, default=12)
    parser.add_argument("--max-k", type=int, default=20)
    parser.add_argument(
        "--fourier-output",
        type=Path,
        default=Path("results/low_frequency_power.csv"),
    )
    parser.add_argument(
        "--parity-output",
        type=Path,
        default=Path("results/parity_fourier_checks.csv"),
    )
    args = parser.parse_args()

    q_values = np.loadtxt(args.input, dtype=np.int64, delimiter=",")

    low_freq = analyze_fourier(
        q_values,
        min_k=args.min_k,
        max_k=args.max_k,
        cutoffs=(4, 8, 16, 32),
    )
    save_percent_table(low_freq, args.fourier_output)

    parity = analyze_parity_checks(
        q_values,
        ks=tuple(k for k in (12, 14, 16, 18, 20) if args.min_k <= k <= args.max_k),
        cutoffs=(8, 16),
    )
    save_percent_table(parity, args.parity_output)


if __name__ == "__main__":
    main()


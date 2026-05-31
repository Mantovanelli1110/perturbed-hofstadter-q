"""Frequency distribution helpers for the perturbed Hofstadter-Q recursion.

The generated array is assumed to store Q(1), ..., Q(N) in 0-indexed form:
values[i] = Q(i+1).

Since all values of Q are odd in the canonical orbit, the frequency function
used in the paper is

    F(m) = #{n >= 1 : Q(n) = 2m - 1}.

This module computes F(m), cumulative counts C(m), dyadic frequency blocks,
and verification helpers for the dyadic frequency law.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def frequency_array(q_values: np.ndarray, max_m: int | None = None) -> np.ndarray:
    """Return F(1), ..., F(max_m).

    Parameters
    ----------
    q_values:
        Array storing Q(1), ..., Q(N).
    max_m:
        Largest m for which F(m) should be returned.  If omitted, max_m is
        inferred from the largest odd value in q_values.

    Returns
    -------
    np.ndarray
        Integer array f with f[m-1] = F(m).
    """
    q = np.asarray(q_values, dtype=np.int64)

    if q.size == 0:
        return np.array([], dtype=np.int64)

    if np.any(q <= 0):
        raise ValueError("Q-values must be positive.")
    if np.any(q % 2 == 0):
        raise ValueError("All Q-values are expected to be odd.")

    observed_m = (q + 1) // 2

    if max_m is None:
        max_m = int(observed_m.max())

    if max_m < 1:
        return np.array([], dtype=np.int64)

    counts = np.bincount(observed_m, minlength=max_m + 1)
    return counts[1 : max_m + 1].astype(np.int64)


def frequency_table(q_values: np.ndarray, max_m: int | None = None) -> pd.DataFrame:
    """Return a table with m, value=2m-1, F(m), and cumulative C(m)."""
    f = frequency_array(q_values, max_m=max_m)
    m = np.arange(1, f.size + 1, dtype=np.int64)
    values = 2 * m - 1
    cumulative = np.cumsum(f)

    return pd.DataFrame(
        {
            "m": m,
            "value": values,
            "F": f,
            "C": cumulative,
        }
    )


def running_max_frequency(frequencies: np.ndarray) -> np.ndarray:
    """Return M(m)=max_{1 <= j <= m} F(j)."""
    f = np.asarray(frequencies, dtype=np.int64)
    if f.size == 0:
        return f.copy()
    return np.maximum.accumulate(f)


def dyadic_frequency_block(frequencies: np.ndarray, k: int) -> np.ndarray:
    """Return F(s) for s in the dyadic value block [2^k, 2^(k+1)).

    The input array stores F(1), F(2), ... in 0-indexed form.
    """
    if k < 0:
        raise ValueError("k must be nonnegative.")

    f = np.asarray(frequencies, dtype=np.int64)

    start_s = 2**k
    stop_s = 2 ** (k + 1)

    start_idx = start_s - 1
    stop_idx = stop_s - 1

    if stop_idx > f.size:
        raise ValueError(
            f"Need F(s) up to s={stop_s - 1} for k={k}, "
            f"but only have F(1),...,F({f.size})."
        )

    return f[start_idx:stop_idx]


def dyadic_frequency_blocks(
    frequencies: np.ndarray,
    min_k: int = 0,
    max_k: int | None = None,
    require_complete: bool = True,
) -> dict[int, np.ndarray]:
    """Return dyadic blocks of F on [2^k, 2^(k+1))."""
    f = np.asarray(frequencies, dtype=np.int64)

    if f.size == 0:
        return {}

    if max_k is None:
        max_k = int(np.floor(np.log2(f.size + 1))) - 1

    blocks: dict[int, np.ndarray] = {}

    for k in range(min_k, max_k + 1):
        start_idx = 2**k - 1
        stop_idx = 2 ** (k + 1) - 1

        if stop_idx > f.size:
            if require_complete:
                continue
            stop_idx = f.size

        if start_idx < f.size:
            blocks[k] = f[start_idx:stop_idx]

    return blocks


def expected_dyadic_block(k: int) -> np.ndarray:
    """Return the expected multiset values 3+nu_2(j), j=1,...,2^k."""
    if k < 0:
        raise ValueError("k must be nonnegative.")

    j = np.arange(1, 2**k + 1, dtype=np.int64)

    # nu_2(j) is the exponent of the largest power of 2 dividing j.
    # For positive integers, this is log2(j & -j).
    lowbit = j & -j
    nu2 = np.log2(lowbit).astype(np.int64)

    return 3 + nu2


def check_dyadic_frequency_law(frequencies: np.ndarray, max_k: int) -> pd.DataFrame:
    """Check the dyadic frequency law up to max_k.

    Returns a DataFrame with one row per k and a Boolean column ``ok``.
    """
    rows: list[dict[str, int | bool]] = []

    for k in range(0, max_k + 1):
        observed = dyadic_frequency_block(frequencies, k)
        expected = expected_dyadic_block(k)

        ok = np.array_equal(np.sort(observed), np.sort(expected))

        rows.append(
            {
                "k": k,
                "block_size": 2**k,
                "observed_mass": int(observed.sum()),
                "expected_mass": int(expected.sum()),
                "ok": bool(ok),
            }
        )

    return pd.DataFrame(rows)


def cumulative_formula(m: np.ndarray | int) -> np.ndarray:
    """Return the predicted C(m)=4m-k-s_2(r+1).

    Here m=2^k+r with 0 <= r < 2^k.
    """
    arr = np.asarray(m, dtype=np.int64)

    if np.any(arr < 1):
        raise ValueError("m must be positive.")

    k = np.floor(np.log2(arr)).astype(np.int64)
    r = arr - (1 << k)

    # Python int.bit_count is exact and avoids depending on NumPy versions.
    s2 = np.array([int(x).bit_count() for x in (r + 1)], dtype=np.int64)

    return 4 * arr - k - s2


def check_cumulative_formula(frequencies: np.ndarray, max_m: int | None = None) -> pd.DataFrame:
    """Check C(m)=4m-k-s_2(r+1) for m up to max_m."""
    f = np.asarray(frequencies, dtype=np.int64)

    if max_m is None:
        max_m = f.size

    if max_m < 1:
        return pd.DataFrame({"m": [], "C_observed": [], "C_expected": [], "ok": []})

    if max_m > f.size:
        raise ValueError(f"Only have frequencies up to m={f.size}.")

    m = np.arange(1, max_m + 1, dtype=np.int64)
    c_observed = np.cumsum(f[:max_m])
    c_expected = cumulative_formula(m)

    return pd.DataFrame(
        {
            "m": m,
            "C_observed": c_observed,
            "C_expected": c_expected,
            "ok": c_observed == c_expected,
        }
    )


def save_frequency_table(q_values: np.ndarray, path: Path, max_m: int | None = None) -> None:
    """Save the frequency table as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frequency_table(q_values, max_m=max_m).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute frequency data for the perturbed Hofstadter-Q sequence."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/Q_10000000.csv"),
        help="CSV file containing Q(1),...,Q(N) as one integer per row.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/frequency_table.csv"),
    )
    parser.add_argument("--max-m", type=int, default=None)
    parser.add_argument(
        "--check-output",
        type=Path,
        default=Path("results/dyadic_frequency_checks.csv"),
    )
    parser.add_argument("--max-k", type=int, default=20)
    args = parser.parse_args()

    q_values = np.loadtxt(args.input, dtype=np.int64, delimiter=",")

    table = frequency_table(q_values, max_m=args.max_m)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)

    f = table["F"].to_numpy(dtype=np.int64)

    max_k = min(args.max_k, int(np.floor(np.log2(f.size + 1))) - 1)
    checks = check_dyadic_frequency_law(f, max_k=max_k)
    args.check_output.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(args.check_output, index=False)


if __name__ == "__main__":
    main()

"""Defect, fluctuation, and dyadic-block utilities for perturbed Hofstadter-Q data.

The generated array is assumed to store Q(1), ..., Q(N) in 0-indexed form:
values[i] = Q(i+1).

The main defect sequence used in the paper is

    R(n) = Q(2n) - 2 Q(n).

This module also provides the centered fluctuation

    E(n) = Q(n) - n/2

and utilities for extracting dyadic blocks.
"""

from __future__ import annotations

import numpy as np


def first_differences(values: np.ndarray) -> np.ndarray:
    """Return first differences Q(n+1)-Q(n)."""
    q = np.asarray(values, dtype=np.int64)
    return np.diff(q)


def centered_fluctuation(values: np.ndarray) -> np.ndarray:
    """Return E(n)=Q(n)-n/2 for n=1,...,N."""
    q = np.asarray(values, dtype=np.int64)
    n = np.arange(1, q.size + 1, dtype=np.float64)
    return q.astype(np.float64) - n / 2.0


def dyadic_defect(values: np.ndarray) -> np.ndarray:
    """Return R(n)=Q(2n)-2Q(n) for all n with 2n <= N.

    If values has length N, the returned array has length floor(N/2), with
    result[i] = R(i+1).
    """
    q = np.asarray(values, dtype=np.int64)
    max_n = q.size // 2
    if max_n == 0:
        return np.array([], dtype=np.int64)

    q_n = q[:max_n]                 # Q(1), ..., Q(max_n)
    q_2n = q[1 : 2 * max_n : 2]     # Q(2), Q(4), ..., Q(2 max_n)

    return q_2n - 2 * q_n


def dyadic_block(values: np.ndarray, k: int, one_indexed: bool = True) -> np.ndarray:
    """Return the block [2^k, 2^(k+1)) from a sequence-like array.

    Parameters
    ----------
    values:
        Array storing a sequence in 0-indexed Python form.
    k:
        Dyadic block exponent.
    one_indexed:
        If True, values[i] is interpreted as a(n) with n=i+1, and the returned
        block is a(2^k), ..., a(2^(k+1)-1).  This is the convention used for Q,
        R, and F in the paper.

        If False, values is interpreted as indexed from 0, and the returned
        slice is values[2^k : 2^(k+1)].
    """
    if k < 0:
        raise ValueError("k must be nonnegative.")

    arr = np.asarray(values)
    start = 2**k
    stop = 2 ** (k + 1)

    if one_indexed:
        start -= 1
        stop -= 1

    if start >= arr.size:
        return arr[:0].copy()

    return arr[start : min(stop, arr.size)]


def dyadic_blocks(
    values: np.ndarray,
    min_power: int = 6,
    max_power: int | None = None,
    one_indexed: bool = True,
    require_complete: bool = True,
) -> dict[int, np.ndarray]:
    """Split a sequence into dyadic blocks [2^k, 2^(k+1)).

    By default, this uses the mathematical 1-indexed convention:
    values[i] = a(i+1).  Thus block k contains a(2^k), ..., a(2^(k+1)-1).

    If require_complete=True, only complete dyadic blocks are returned.
    """
    arr = np.asarray(values)

    if min_power < 0:
        raise ValueError("min_power must be nonnegative.")

    if arr.size == 0:
        return {}

    if max_power is None:
        if one_indexed:
            # Need indices 2^k,...,2^(k+1)-1 to be available, so 2^(k+1)-1 <= N.
            max_power = int(np.floor(np.log2(arr.size + 1))) - 1
        else:
            # Need Python indices 2^k,...,2^(k+1)-1 to be available, so 2^(k+1) <= N.
            max_power = int(np.floor(np.log2(arr.size))) - 1

    blocks: dict[int, np.ndarray] = {}

    for k in range(min_power, max_power + 1):
        block = dyadic_block(arr, k, one_indexed=one_indexed)

        expected_len = 2**k
        if require_complete and block.size != expected_len:
            continue

        if block.size > 0:
            blocks[k] = block

    return blocks


def standardized(values: np.ndarray, ddof: int = 0) -> np.ndarray:
    """Return a centered and variance-normalized floating-point copy."""
    arr = np.asarray(values, dtype=np.float64)

    if arr.size == 0:
        return arr.copy()

    mean = arr.mean()
    std = arr.std(ddof=ddof)

    if std == 0:
        raise ValueError("Cannot standardize an array with zero standard deviation.")

    return (arr - mean) / std


def parity_subprofiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a one-indexed profile into even-site and odd-site subprofiles.

    For a block profile R_k(j)=R(2^k+j), j=0,...,2^k-1, this returns

        even = R_k(0), R_k(2), R_k(4), ...
        odd  = R_k(1), R_k(3), R_k(5), ...

    corresponding to the parity split used in the Fourier robustness checks.
    """
    arr = np.asarray(values)
    return arr[0::2], arr[1::2]

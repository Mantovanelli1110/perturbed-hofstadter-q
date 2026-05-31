"""Generate finite prefixes of the Hofstadter-Q sequence.

The canonical recurrence is

    Q(1) = Q(2) = 1
    Q(n) = Q(n - Q(n - 1)) + Q(n - Q(n - 2))

This module keeps perturbations explicit: callers can provide initial seeds or
post-process generated values in their own experiment code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def generate_q(n: int, q1: int = 1, q2: int = 1) -> np.ndarray:
    """Return Q(1), ..., Q(n) as a 1-indexed recurrence stored in 0-indexed form."""
    if n < 1:
        return np.array([], dtype=np.int64)
    if q1 < 1 or q2 < 1:
        raise ValueError("Seeds must be positive integers.")

    q = np.zeros(n, dtype=np.int64)
    q[0] = q1
    if n == 1:
        return q
    q[1] = q2

    for idx in range(2, n):
        left = idx + 1 - q[idx - 1]
        right = idx + 1 - q[idx - 2]
        if left < 1 or right < 1:
            raise ValueError(f"Invalid recurrence index at n={idx + 1}.")
        q[idx] = q[left - 1] + q[right - 1]
    return q


def save_sequence(values: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, values, fmt="%d", delimiter=",")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Hofstadter-Q prefix.")
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=Path("data/Q_4096.csv"))
    args = parser.parse_args()
    save_sequence(generate_q(args.n), args.output)


if __name__ == "__main__":
    main()


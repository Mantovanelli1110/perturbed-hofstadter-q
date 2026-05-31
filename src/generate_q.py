"""Generate finite prefixes of the perturbed Hofstadter-Q sequence.

The recurrence studied in the paper is

    Q(1) = Q(2) = 1,
    Q(n) = Q(n - Q(n - 1)) + Q(n - Q(n - 2)) + (-1)^n   for n >= 3.

The returned NumPy array stores Q(1), ..., Q(N) in 0-indexed form:
values[i] = Q(i+1).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def generate_q(n: int, q1: int = 1, q2: int = 1) -> np.ndarray:
    """Return Q(1), ..., Q(n) for the perturbed Hofstadter-Q recursion.

    Parameters
    ----------
    n:
        Number of terms to generate.
    q1, q2:
        Initial values Q(1) and Q(2).  The canonical orbit uses q1=q2=1.

    Returns
    -------
    np.ndarray
        Array of length n with values[i] = Q(i+1).

    Raises
    ------
    ValueError
        If a recursive argument is nonpositive or exceeds the computed range.
    """
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
        n_math = idx + 1

        left = n_math - q[idx - 1]
        right = n_math - q[idx - 2]

        if left < 1 or right < 1:
            raise ValueError(
                f"Invalid recurrence index at n={n_math}: "
                f"left={left}, right={right}."
            )
        if left > n_math - 1 or right > n_math - 1:
            raise ValueError(
                f"Forward reference at n={n_math}: "
                f"left={left}, right={right}."
            )

        perturbation = 1 if n_math % 2 == 0 else -1
        q[idx] = q[left - 1] + q[right - 1] + perturbation

    return q


def save_sequence(values: np.ndarray, path: Path) -> None:
    """Save a generated sequence prefix as a one-column CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, values, fmt="%d", delimiter=",")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a prefix of the perturbed Hofstadter-Q recursion."
    )
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--q1", type=int, default=1)
    parser.add_argument("--q2", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("data/Q_4096.csv"))
    args = parser.parse_args()

    values = generate_q(args.n, q1=args.q1, q2=args.q2)
    save_sequence(values, args.output)


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()


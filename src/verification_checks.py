"""Lightweight verification checks for generated Q prefixes."""

from __future__ import annotations

import argparse

import numpy as np

from generate_q import generate_q


def check_positive(values: np.ndarray) -> bool:
    return bool(np.all(values > 0))


def check_recurrence(values: np.ndarray) -> bool:
    if values.size <= 2:
        return True
    for idx in range(2, values.size):
        left = idx + 1 - values[idx - 1]
        right = idx + 1 - values[idx - 2]
        if left < 1 or right < 1:
            return False
        expected = values[left - 1] + values[right - 1]
        if values[idx] != expected:
            return False
    return True


def verification_summary(n: int) -> str:
    values = generate_q(n)
    lines = [
        f"n={n}",
        f"positive={check_positive(values)}",
        f"recurrence={check_recurrence(values)}",
        f"min={values.min() if values.size else 'NA'}",
        f"max={values.max() if values.size else 'NA'}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run verification checks.")
    parser.add_argument("--n", type=int, default=4096)
    args = parser.parse_args()
    print(verification_summary(args.n))


if __name__ == "__main__":
    main()


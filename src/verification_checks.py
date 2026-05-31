"""Verification checks for generated prefixes of the perturbed Hofstadter-Q recursion.

The recurrence studied in the paper is

    Q(1) = Q(2) = 1,
    Q(n) = Q(n-Q(n-1)) + Q(n-Q(n-2)) + (-1)^n,  n >= 3.

The generated array stores Q(1), ..., Q(N) in 0-indexed form:
values[i] = Q(i+1).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from generate_q import generate_q


def perturbation(n_math: int) -> int:
    """Return (-1)^n for a mathematical index n."""
    return 1 if n_math % 2 == 0 else -1


def check_positive(values: np.ndarray) -> bool:
    """Check Q(n)>0 for all computed n."""
    q = np.asarray(values, dtype=np.int64)
    return bool(np.all(q > 0))


def check_recursive_arguments(values: np.ndarray) -> bool:
    """Check positivity and backwardness of all recursive arguments."""
    q = np.asarray(values, dtype=np.int64)

    for idx in range(2, q.size):
        n_math = idx + 1
        left = n_math - q[idx - 1]
        right = n_math - q[idx - 2]

        if left < 1 or right < 1:
            return False
        if left >= n_math or right >= n_math:
            return False

    return True


def check_recurrence(values: np.ndarray) -> bool:
    """Check the perturbed Hofstadter-Q recurrence."""
    q = np.asarray(values, dtype=np.int64)

    if q.size <= 2:
        return True

    if q[0] != 1 or q[1] != 1:
        return False

    for idx in range(2, q.size):
        n_math = idx + 1
        left = n_math - q[idx - 1]
        right = n_math - q[idx - 2]

        if left < 1 or right < 1:
            return False
        if left >= n_math or right >= n_math:
            return False

        expected = q[left - 1] + q[right - 1] + perturbation(n_math)

        if q[idx] != expected:
            return False

    return True


def check_parity(values: np.ndarray) -> bool:
    """Check Q(n) is odd for all computed n."""
    q = np.asarray(values, dtype=np.int64)
    return bool(np.all(q % 2 == 1))


def check_first_differences(values: np.ndarray) -> bool:
    """Check Q(n+1)-Q(n) lies in {-2,0,2} throughout the prefix."""
    q = np.asarray(values, dtype=np.int64)

    if q.size < 2:
        return True

    diffs = np.diff(q)
    return bool(np.all(np.isin(diffs, np.array([-2, 0, 2], dtype=np.int64))))


def clock_process(values: np.ndarray) -> np.ndarray:
    """Return t_1(n)=n-Q(n-1) for n=2,...,N+1.

    The returned array t has t[i] = t_1(i+2).
    """
    q = np.asarray(values, dtype=np.int64)
    n = np.arange(2, q.size + 2, dtype=np.int64)
    return n - q


def check_clock_reconstruction(values: np.ndarray) -> bool:
    """Check Q(m)=m+1-t_1(m+1)."""
    q = np.asarray(values, dtype=np.int64)

    if q.size == 0:
        return True

    t = clock_process(q)
    m = np.arange(1, q.size + 1, dtype=np.int64)

    reconstructed = m + 1 - t[: q.size]
    return bool(np.array_equal(q, reconstructed))


def dyadic_defect(values: np.ndarray) -> np.ndarray:
    """Return R(n)=Q(2n)-2Q(n) for all n with 2n <= N."""
    q = np.asarray(values, dtype=np.int64)
    max_n = q.size // 2

    if max_n == 0:
        return np.array([], dtype=np.int64)

    q_n = q[:max_n]
    q_2n = q[1 : 2 * max_n : 2]

    return q_2n - 2 * q_n


def check_defect_identity(values: np.ndarray) -> bool:
    """Check R(n)=2t_1(n+1)-t_1(2n+1)-1."""
    q = np.asarray(values, dtype=np.int64)
    max_n = q.size // 2

    if max_n == 0:
        return True

    r_direct = dyadic_defect(q)
    t = clock_process(q)

    # t array has t[i] = t_1(i+2).
    # t_1(n+1) is t[(n+1)-2] = t[n-1].
    # t_1(2n+1) is t[(2n+1)-2] = t[2n-1].
    n = np.arange(1, max_n + 1, dtype=np.int64)
    r_clock = 2 * t[n - 1] - t[2 * n - 1] - 1

    return bool(np.array_equal(r_direct, r_clock))


def frequency_array(values: np.ndarray, max_m: int | None = None) -> np.ndarray:
    """Return F(1),...,F(max_m), where F(m)=#{n:Q(n)=2m-1}."""
    q = np.asarray(values, dtype=np.int64)

    if q.size == 0:
        return np.array([], dtype=np.int64)

    if np.any(q % 2 == 0):
        raise ValueError("All Q-values must be odd to compute F(m).")

    observed_m = (q + 1) // 2

    if max_m is None:
        max_m = int(observed_m.max())

    counts = np.bincount(observed_m, minlength=max_m + 1)
    return counts[1 : max_m + 1].astype(np.int64)


def expected_dyadic_block(k: int) -> np.ndarray:
    """Return expected multiset 3+nu_2(j), j=1,...,2^k."""
    j = np.arange(1, 2**k + 1, dtype=np.int64)
    lowbit = j & -j
    nu2 = np.log2(lowbit).astype(np.int64)
    return 3 + nu2


def check_dyadic_frequency_law(values: np.ndarray, max_k: int) -> bool:
    """Check the dyadic frequency law through complete blocks up to max_k."""
    f = frequency_array(values)

    for k in range(max_k + 1):
        start = 2**k - 1
        stop = 2 ** (k + 1) - 1

        if stop > f.size:
            return False

        observed = f[start:stop]
        expected = expected_dyadic_block(k)

        if not np.array_equal(np.sort(observed), np.sort(expected)):
            return False

    return True


def verification_summary(n: int, max_k: int | None = None) -> str:
    """Generate Q(1),...,Q(n) and return a text verification summary."""
    values = generate_q(n)

    if max_k is None:
        # Conservative default: only test dyadic frequency blocks that fit well
        # inside the observed value range.
        f = frequency_array(values)
        max_k = max(0, int(np.floor(np.log2(f.size + 1))) - 1)

    checks = {
        "positive_values": check_positive(values),
        "recursive_arguments": check_recursive_arguments(values),
        "perturbed_recurrence": check_recurrence(values),
        "parity_Q_odd": check_parity(values),
        "first_differences_in_-2_0_2": check_first_differences(values),
        "clock_reconstruction": check_clock_reconstruction(values),
        "defect_identity": check_defect_identity(values),
        "dyadic_frequency_law": check_dyadic_frequency_law(values, max_k=max_k),
    }

    lines = [
        "Verification summary",
        "",
        f"N = {n}",
        f"max_k_frequency_check = {max_k}",
        f"min_Q = {values.min() if values.size else 'NA'}",
        f"max_Q = {values.max() if values.size else 'NA'}",
        "",
        "Checks:",
    ]

    for name, ok in checks.items():
        status = "OK" if ok else "FAILED"
        lines.append(f"[{status}] {name}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run verification checks for the perturbed Hofstadter-Q recursion."
    )
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--max-k", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for saving the verification summary.",
    )
    args = parser.parse_args()

    summary = verification_summary(args.n, max_k=args.max_k)

    if args.output is None:
        print(summary)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(summary + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

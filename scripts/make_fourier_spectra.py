"""Generate fourier_spectra_dyadic_profiles.png.

This figure plots Fourier spectra of dyadic defect profiles.

For each k in {4,...,10}, we compute

    R(n) = Q(2n) - 2 Q(n),

extract the dyadic block

    R_k(j) = R(2^k + j),   0 <= j < 2^k,

center and standardize the block, compute the one-sided Fourier power
spectrum, and normalize the displayed power by its maximum value.

The displayed Fourier modes are the first 60 nonnegative modes.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from defects import dyadic_block, dyadic_defect, standardized
from generate_q import generate_q


FIGURES_DIR = Path("figures")


def normalized_one_sided_power(profile: np.ndarray) -> np.ndarray:
    """Return one-sided Fourier power normalized by its maximum.

    The input profile is centered and standardized before transformation.
    """
    profile_std = standardized(profile)

    coeffs = np.fft.rfft(profile_std)
    power = np.abs(coeffs) ** 2

    if power.max() > 0:
        power = power / power.max()

    return power


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ks = list(range(4, 11))
    max_mode = 60

    # Need R(n) through the block k=10, i.e. n < 2^11.
    # Since R(n)=Q(2n)-2Q(n), generating Q up to 2^12 is sufficient.
    # We generate a slightly larger prefix for safety.
    q = generate_q(2**13)
    r = dyadic_defect(q)

    fig, ax = plt.subplots(figsize=(12, 6))

    for k in ks:
        block = dyadic_block(r, k, one_indexed=True)
        power = normalized_one_sided_power(block)

        modes = np.arange(power.size)
        cutoff = min(max_mode + 1, power.size)

        ax.plot(modes[:cutoff], power[:cutoff], linewidth=1.2, label=rf"$k={k}$")

    ax.set_title("Fourier spectra of dyadic fluctuation profiles")
    ax.set_xlabel("Fourier mode")
    ax.set_ylabel("normalized power")
    ax.set_xlim(0, max_mode)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=2, frameon=False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fourier_spectra_dyadic_profiles.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

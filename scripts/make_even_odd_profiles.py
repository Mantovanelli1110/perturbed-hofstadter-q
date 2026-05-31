"""Generate even_odd_profiles_dyadic_blocks.png.

This figure shows parity-separated dyadic defect profiles for representative
dyadic blocks.  For each k in {14,16,18,20}, we compute

    R(n) = Q(2n) - 2 Q(n),

extract the dyadic block

    R_k(j) = R(2^k + j),   0 <= j < 2^k,

split it into even and odd subprofiles

    R_k^(0)(m) = R(2^k + 2m),
    R_k^(1)(m) = R(2^k + 2m + 1),

then center and standardize each parity subprofile separately before plotting
against the normalized coordinate m/(2^(k-1)-1).
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from defects import dyadic_block, dyadic_defect, parity_subprofiles, standardized
from generate_q import generate_q


FIGURES_DIR = Path("figures")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Need R(n) through the block k=20, i.e. n < 2^21.
    # Since R(n)=Q(2n)-2Q(n), generate Q up to 2^22.
    q = generate_q(2**22)
    r = dyadic_defect(q)

    ks = [14, 16, 18, 20]

    fig, ax = plt.subplots(figsize=(12, 6))

    for k in ks:
        block = dyadic_block(r, k, one_indexed=True)

        even_profile, odd_profile = parity_subprofiles(block)

        even_std = standardized(even_profile)
        odd_std = standardized(odd_profile)

        m_even = np.arange(even_std.size)
        m_odd = np.arange(odd_std.size)

        x_even = m_even / (even_std.size - 1)
        x_odd = m_odd / (odd_std.size - 1)

        ax.plot(x_even, even_std, linewidth=1.0, label=rf"$k={k}$, even")
        ax.plot(
            x_odd,
            odd_std,
            linewidth=1.0,
            linestyle="--",
            label=rf"$k={k}$, odd",
        )

    ax.set_title("Parity-separated dyadic defect profiles")
    ax.set_xlabel("Normalized position in parity subprofile")
    ax.set_ylabel("Standardized defect")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", ncol=2)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "even_odd_profiles_dyadic_blocks.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

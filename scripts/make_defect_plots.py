"""Generate the four defect-related figures for the perturbed Hofstadter Q paper.

This script produces:

    figures/perturbed_hofstadter_first_differences_clean.png
    figures/perturbed_hofstadter_safety_margin_clean.png
    figures/renormalized_defects.png
    figures/dyadically_normalized_R_logscale.png

It assumes that:
- src/generate_q.py generates the perturbed recursion
      Q(n)=Q(n-Q(n-1))+Q(n-Q(n-2))+(-1)^n
- src/defects.py contains the helper functions already prepared.

The first three figures use a prefix of length 4096, while the log-scale defect
plot uses a larger prefix so that the defect sequence is available up to n=2^15.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from defects import centered_fluctuation, dyadic_block, dyadic_defect, first_differences, standardized
from generate_q import generate_q


FIGURES_DIR = Path("figures")


def add_dyadic_guides(ax: plt.Axes, n_max: int, *, alpha: float = 0.22) -> None:
    """Add light vertical guide lines at n = 2^k."""
    k = 0
    while 2**k <= n_max:
        ax.axvline(2**k, linewidth=0.7, alpha=alpha)
        k += 1


def add_dyadic_guides_log2(ax: plt.Axes, k_min: int, k_max: int, *, alpha: float = 0.28) -> None:
    """Add light vertical guide lines at log2(n)=k."""
    for k in range(k_min, k_max + 1):
        ax.axvline(k, linewidth=0.7, alpha=alpha, linestyle=":")


def plot_first_differences(q: np.ndarray) -> None:
    """Reproduce perturbed_hofstadter_first_differences_clean.png."""
    d = first_differences(q)
    n = np.arange(1, d.size + 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(n, d, s=8, label=r"$D(n)=Q(n+1)-Q(n)$")
    ax.axhline(0.0, linewidth=1.2, alpha=0.6)

    add_dyadic_guides(ax, int(n[-1]))

    ax.set_title("First differences of the perturbed Hofstadter recursion")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$D(n)$")
    ax.legend(loc="upper left")

    ax.text(
        0.5,
        -0.16,
        r"Light vertical guide lines mark dyadic positions $n=2^k$.",
        transform=ax.transAxes,
        ha="center",
        va="top",
    )

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "perturbed_hofstadter_first_differences_clean.png", dpi=200)
    plt.close(fig)


def plot_safety_margin(q: np.ndarray) -> None:
    """Reproduce perturbed_hofstadter_safety_margin_clean.png.

    The plotted quantity is
        S(n)=n-max{Q(n-1),Q(n-2)}.
    """
    n = np.arange(3, q.size + 1)
    s = n - np.maximum(q[1:-1], q[:-2])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(n, s, s=8, label=r"$S(n)=n-\max\{Q(n-1),Q(n-2)\}$")
    ax.plot(n, n / 2.0, linestyle="--", linewidth=1.5, label=r"$n/2$")

    add_dyadic_guides(ax, int(n[-1]))

    ax.set_title("Safety margin of the perturbed Hofstadter recursion")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$S(n)$")
    ax.legend(loc="upper left")

    ax.text(
        0.5,
        -0.16,
        r"Light vertical guide lines mark dyadic positions $n=2^k$.",
        transform=ax.transAxes,
        ha="center",
        va="top",
    )

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "perturbed_hofstadter_safety_margin_clean.png", dpi=200)
    plt.close(fig)


def build_centered_profiles(q: np.ndarray, ks: list[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Build centered fluctuation profiles E(2^k m) on normalized x-grids.

    For each k, define
        E(n)=Q(n)-n/2,
    and sample the profile
        m -> E(2^k m),   1 <= m <= floor(N/2^k).

    The returned dictionary maps k to (x, profile), where
        x = (m-1)/(M_k-1) in [0,1].
    """
    e = centered_fluctuation(q)
    n_total = q.size

    profiles: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k in ks:
        step = 2**k
        m_max = n_total // step
        m = np.arange(1, m_max + 1)
        sampled = e[step * m - 1]
        if m_max > 1:
            x = (m - 1) / (m_max - 1)
        else:
            x = np.array([0.0], dtype=float)
        profiles[k] = (x, sampled)
    return profiles


def plot_renormalized_defects(q: np.ndarray) -> None:
    """Reproduce renormalized_defects.png.

    This figure uses centered fluctuations
        E(n)=Q(n)-n/2
    sampled along dyadic subsequences n=2^k m.  The top panel overlays the
    profiles against the normalized coordinate
        x=(m-1)/(M_k-1).
    The bottom row compares each k-profile against a reference profile.
    """
    ks = list(range(0, 6))
    profiles = build_centered_profiles(q, ks)

    # Reference profile for the small comparison panels.
    # Using k=1 gives the visual behavior seen in the screenshot:
    # k=0 differs; k>=1 nearly collapse.
    ref_k = 1
    x_ref, y_ref = profiles[ref_k]

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 6, height_ratios=[3.2, 1.6], hspace=0.42, wspace=0.28)

    # Top large panel.
    ax_main = fig.add_subplot(gs[0, :])
    markers = ["o", "s", "^", "D", "v", "*"]

    for marker, k in zip(markers, ks):
        x, y = profiles[k]
        ax_main.scatter(x, y, s=18, marker=marker, alpha=0.8, label=rf"$k={k}$")

    ax_main.axhline(0.0, linewidth=1.2, alpha=0.6)
    ax_main.set_title("Dyadic rescaling collapse of centered defects")
    ax_main.set_xlabel(r"normalized position $x=(m-1)/(M_k-1)$")
    ax_main.set_ylabel(r"$Q(2^k m)-2^{k-1}m$")
    ax_main.legend(title=r"scale $k$", loc="upper left", ncol=2, framealpha=0.9)

    # Bottom row: compare each profile to the reference profile.
    for j, k in enumerate(ks):
        ax = fig.add_subplot(gs[1, j])
        x, y = profiles[k]

        # Interpolate the reference profile to the current x-grid.
        y_ref_interp = np.interp(x, x_ref, y_ref)

        ax.scatter(x, y, s=14, alpha=0.8)
        ax.scatter(x, y_ref_interp, s=12, alpha=0.8)
        ax.axhline(0.0, linewidth=1.0, alpha=0.6)

        ax.set_title(rf"$k={k}$")
        ax.set_xlabel(r"$x$")

        if j == 0:
            ax.set_ylabel("profile")
        else:
            ax.set_yticklabels([])

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "renormalized_defects.png", dpi=200)
    plt.close(fig)


def plot_dyadically_normalized_R_logscale() -> None:
    """Reproduce dyadically_normalized_R_logscale.png.

    We generate Q up to 2^16 so that the dyadic defect
        R(n)=Q(2n)-2Q(n)
    is available for n <= 2^15.  For each dyadic block [2^k,2^(k+1)),
    k=4,...,14, we center and standardize the block independently and plot it
    against log2(n).
    """
    n_q = 2**16
    q = generate_q(n_q)
    r = dyadic_defect(q)  # available for n=1,...,2^15

    fig, ax = plt.subplots(figsize=(12, 6))

    k_min = 4
    k_max = 14

    for k in range(k_min, k_max + 1):
        block = dyadic_block(r, k, one_indexed=True)
        block_std = standardized(block)

        n_vals = np.arange(2**k, 2 ** (k + 1))
        x = np.log2(n_vals)

        ax.plot(x, block_std, linewidth=0.8)

    add_dyadic_guides_log2(ax, k_min, k_max + 1)

    ax.set_title(r"Dyadically normalized defect sequence on the $\log_2 n$ scale")
    ax.set_xlabel(r"$\log_2 n$")
    ax.set_ylabel(r"normalized $R(n)$")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dyadically_normalized_R_logscale.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Prefix for the first three figures.
    q_4096 = generate_q(4096)

    plot_first_differences(q_4096)
    plot_safety_margin(q_4096)
    plot_renormalized_defects(q_4096)
    plot_dyadically_normalized_R_logscale()


if __name__ == "__main__":
    main()

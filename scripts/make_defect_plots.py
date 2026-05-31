"""Generate defect-related figures for the perturbed Hofstadter-Q paper.

This script produces:

    figures/perturbed_hofstadter_first_differences_clean.png
    figures/perturbed_hofstadter_safety_margin_clean.png
    figures/renormalized_defects.png
    figures/dyadically_normalized_R_logscale.png

Assumptions
-----------
- src/generate_q.py generates the perturbed recursion
      Q(n)=Q(n-Q(n-1))+Q(n-Q(n-2))+(-1)^n
- src/defects.py provides:
      first_differences
      centered_fluctuation
      dyadic_defect
      dyadic_block
      standardized
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from defects import (
    centered_fluctuation,
    dyadic_block,
    dyadic_defect,
    first_differences,
    standardized,
)
from generate_q import generate_q


FIGURES_DIR = Path("figures")


def add_dyadic_guides(ax: plt.Axes, n_max: int, *, alpha: float = 0.22) -> None:
    """Add light vertical guide lines at n = 2^k."""
    k = 0
    while 2**k <= n_max:
        ax.axvline(2**k, linewidth=0.7, alpha=alpha)
        k += 1


def add_dyadic_guides_log2(
    ax: plt.Axes, k_min: int, k_max: int, *, alpha: float = 0.28
) -> None:
    """Add light vertical guide lines at log2(n) = k."""
    for k in range(k_min, k_max + 1):
        ax.axvline(k, linewidth=0.7, alpha=alpha, linestyle=":", color="tab:blue")


def plot_first_differences(q: np.ndarray) -> None:
    """Generate perturbed_hofstadter_first_differences_clean.png."""
    d = first_differences(q)
    n = np.arange(1, d.size + 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(n, d, s=8, label=r"$D(n)=Q(n+1)-Q(n)$")
    ax.axhline(0.0, linewidth=1.2, alpha=0.6)

    # Use q.size here so the final guide line at n=4096 is included.
    add_dyadic_guides(ax, q.size)
    ax.set_xlim(0, q.size + 200)

    ax.set_title("First differences of the perturbed Hofstadter recursion")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$D(n)$")
    ax.legend(loc="upper left", frameon=False)

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
    """Generate perturbed_hofstadter_safety_margin_clean.png.

    The plotted quantity is
        S(n)=n-max{Q(n-1),Q(n-2)}.
    """
    n = np.arange(3, q.size + 1)
    s = n - np.maximum(q[1:-1], q[:-2])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(n, s, s=8, label=r"$S(n)=n-\max\{Q(n-1),Q(n-2)\}$")
    ax.plot(n, n / 2.0, linestyle="--", linewidth=1.5, label=r"$n/2$")

    add_dyadic_guides(ax, int(n[-1]))
    ax.set_xlim(0, q.size + 200)

    ax.set_title("Safety margin of the perturbed Hofstadter recursion")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$S(n)$")
    ax.legend(loc="upper left", frameon=False)

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


def build_centered_profiles(
    q: np.ndarray, ks: list[int]
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Build dyadically sampled centered-fluctuation profiles.

    We use
        E(n)=Q(n)-n/2,
    and for each k sample
        m -> E(2^k m)=Q(2^k m)-2^{k-1}m.
    """
    e = centered_fluctuation(q)
    profiles: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    n_total = q.size

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
    """Generate renormalized_defects.png."""
    ks = list(range(0, 6))
    profiles = build_centered_profiles(q, ks)

    # Reference profile used for the lower comparison panels.
    ref_k = 1
    x_ref, y_ref = profiles[ref_k]

    with plt.rc_context(
        {
            "font.size": 14,
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "legend.fontsize": 14,
            "legend.title_fontsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    ):
        fig = plt.figure(figsize=(12, 8.1))
        gs = fig.add_gridspec(
            2,
            6,
            height_ratios=[3.55, 1.55],
            hspace=0.34,
            wspace=0.28,
        )

        # --- top overlay panel ---
        ax_main = fig.add_subplot(gs[0, :])
        markers = ["o", "s", "^", "D", "v", "*"]

        for marker, k in zip(markers, ks):
            x, y = profiles[k]
            ax_main.scatter(x, y, s=24, marker=marker, alpha=0.8, label=rf"$k={k}$")

        ax_main.axhline(0.0, linewidth=1.2, alpha=0.6)
        ax_main.grid(True, alpha=0.25)

        ax_main.set_title("Dyadic rescaling collapse of centered defects", pad=10)
        ax_main.set_xlabel(r"normalized position $x=(m-1)/(M_k-1)$", labelpad=8)
        ax_main.set_ylabel(r"$Q(2^k m)-2^{k-1}m$")
        ax_main.legend(title=r"scale $k$", loc="upper left", ncol=2, framealpha=0.9)

        # --- y-limits for lower panels ---
        # k=1,...,5 share a common symmetric scale
        shared_absmax = 0.0
        for k in ks[1:]:
            x, y = profiles[k]
            y_ref_interp = np.interp(x, x_ref, y_ref)
            shared_absmax = max(
                shared_absmax,
                np.max(np.abs(y)),
                np.max(np.abs(y_ref_interp)),
            )
        shared_ylim = (-1.06 * shared_absmax, 1.06 * shared_absmax)

        # k=0 gets its own symmetric scale
        x0, y0 = profiles[0]
        y0_ref_interp = np.interp(x0, x_ref, y_ref)
        absmax0 = max(np.max(np.abs(y0)), np.max(np.abs(y0_ref_interp)))
        ylim0 = (-1.06 * absmax0, 1.06 * absmax0)

        # --- bottom comparison panels ---
        for j, k in enumerate(ks):
            ax = fig.add_subplot(gs[1, j])
            x, y = profiles[k]
            y_ref_interp = np.interp(x, x_ref, y_ref)

            ax.scatter(x, y, s=18, alpha=0.8)
            ax.scatter(x, y_ref_interp, s=16, alpha=0.8)
            ax.axhline(0.0, linewidth=1.0, alpha=0.6)
            ax.grid(True, alpha=0.25)

            ax.set_title(rf"$k={k}$", fontsize=16, pad=10)
            ax.set_xlabel(r"$x$", fontsize=14, labelpad=4)

            if j == 0:
                ax.set_ylabel("profile", fontsize=14)
                ax.set_ylim(*ylim0)
            else:
                ax.set_yticklabels([])
                ax.set_ylim(*shared_ylim)

        fig.subplots_adjust(top=0.93, bottom=0.08, left=0.08, right=0.99)
        fig.savefig(FIGURES_DIR / "renormalized_defects.png", dpi=200)
        plt.close(fig)


def plot_dyadically_normalized_R_logscale() -> None:
    """Generate dyadically_normalized_R_logscale.png.

    We generate Q up to 2^16 so that
        R(n)=Q(2n)-2Q(n)
    is available up to n=2^15.  For each dyadic block [2^k,2^(k+1)),
    k=4,...,14, we center and standardize the block independently and plot it
    against log2(n).

    All curves are plotted in blue, matching the final agreed version.
    """
    n_q = 2**16
    q = generate_q(n_q)
    r = dyadic_defect(q)

    fig, ax = plt.subplots(figsize=(12, 6))

    k_min = 4
    k_max = 14

    for k in range(k_min, k_max + 1):
        block = dyadic_block(r, k, one_indexed=True)
        block_std = standardized(block)

        n_vals = np.arange(2**k, 2 ** (k + 1))
        x = np.log2(n_vals)

        ax.plot(x, block_std, linewidth=0.8, color="tab:blue")

    add_dyadic_guides_log2(ax, k_min, k_max + 1)

    ax.set_title(r"Dyadically normalized defect sequence on the $\log_2 n$ scale")
    ax.set_xlabel(r"$\log_2 n$")
    ax.set_ylabel(r"normalized $R(n)$")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dyadically_normalized_R_logscale.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Prefix used for the first three figures.
    q_4096 = generate_q(4096)

    plot_first_differences(q_4096)
    plot_safety_margin(q_4096)
    plot_renormalized_defects(q_4096)
    plot_dyadically_normalized_R_logscale()


if __name__ == "__main__":
    main()

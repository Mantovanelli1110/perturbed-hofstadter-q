"""Generate frequency-related figures for the perturbed Hofstadter-Q paper.

This script produces:

    figures/frequency_distribution.png
    figures/dyadic_frequency_blocks_B6_B9.png
    figures/normalized_peak_locations.png

The frequency function is

    F(m) = #{n >= 1 : Q(n) = 2m - 1}.

The script assumes that src/generate_q.py generates the perturbed recursion

    Q(n)=Q(n-Q(n-1))+Q(n-Q(n-2))+(-1)^n.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from generate_q import generate_q


FIGURES_DIR = Path("figures")


def frequency_array(q_values: np.ndarray, max_m: int | None = None) -> np.ndarray:
    """Return F(1),...,F(max_m), where F(m)=#{n: Q(n)=2m-1}."""
    q = np.asarray(q_values, dtype=np.int64)

    if q.size == 0:
        return np.array([], dtype=np.int64)

    if np.any(q % 2 == 0):
        raise ValueError("All Q-values are expected to be odd.")

    observed_m = (q + 1) // 2

    if max_m is None:
        max_m = int(observed_m.max())

    counts = np.bincount(observed_m, minlength=max_m + 1)
    return counts[1 : max_m + 1].astype(np.int64)


def add_dyadic_guides(ax: plt.Axes, x_max: int, *, alpha: float = 0.22) -> None:
    """Add light vertical guide lines at x=2^k."""
    k = 0
    while 2**k <= x_max:
        ax.axvline(2**k, linewidth=0.7, alpha=alpha)
        k += 1


def dyadic_frequency_block(f: np.ndarray, k: int) -> np.ndarray:
    """Return F(m) for m in [2^k,2^(k+1))."""
    start = 2**k - 1
    stop = 2 ** (k + 1) - 1

    if stop > f.size:
        raise ValueError(
            f"Need F(m) up to m={2 ** (k + 1) - 1}, but only have {f.size}."
        )

    return f[start:stop]


def dominant_peak_location(f: np.ndarray, k: int) -> int:
    """Return the m-location of a dominant peak in the dyadic block B_k.

    The block is m in [2^k,2^(k+1)).  If several maxima occur, this returns
    the first one.
    """
    block = dyadic_frequency_block(f, k)
    local_index = int(np.argmax(block))
    return 2**k + local_index


def plot_frequency_distribution(f: np.ndarray) -> None:
    """Generate frequency_distribution.png."""
    max_m = 4000

    if f.size < max_m:
        raise ValueError(f"Need F(m) up to m={max_m}, but only have {f.size} values.")

    m = np.arange(1, max_m + 1)
    f_plot = f[:max_m]
    running_max = np.maximum.accumulate(f_plot)

    title_fs = 18
    label_fs = 16
    tick_fs = 12
    upper_legend_fs = 13
    lower_legend_fs = 11
    small_fs = 11

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.18},
    )

    # Upper panel: frequency values.
    ax1.scatter(m, f_plot, s=9, alpha=0.75, label=r"$F(m)$")
    add_dyadic_guides(ax1, max_m)
    ax1.grid(True, alpha=0.22)
    ax1.set_title(r"Value frequencies $F(m)$", fontsize=title_fs, pad=8)
    ax1.set_ylabel(r"$F(m)$", fontsize=label_fs)
    ax1.tick_params(labelsize=tick_fs)
    ax1.set_ylim(0, 15)
    ax1.legend(loc="upper right", frameon=False, fontsize=upper_legend_fs)

    # Lower panel: running maximum.
    ax2.plot(
        m,
        running_max,
        linewidth=2.0,
        label=r"$M(m)=\max_{1\leq j\leq m} F(j)$",
    )
    add_dyadic_guides(ax2, max_m)
    ax2.grid(True, alpha=0.22)
    ax2.set_xlabel(r"$m$", fontsize=label_fs)
    ax2.set_ylabel(r"$M(m)$", fontsize=label_fs)
    ax2.tick_params(labelsize=tick_fs)
    ax2.set_xlim(0, max_m)
    ax2.set_ylim(0, 15)
    ax2.legend(loc="upper left", frameon=False, fontsize=lower_legend_fs)

    ax2.text(
        0.98,
        0.09,
        r"light vertical lines: $m=2^k$",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=small_fs,
    )

    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.09, top=0.94, hspace=0.18)
    fig.savefig(FIGURES_DIR / "frequency_distribution.png", dpi=200)
    plt.close(fig)


def plot_dyadic_frequency_blocks(f: np.ndarray) -> None:
    """Generate dyadic_frequency_blocks_B6_B9.png."""
    ks = [6, 7, 8, 9]
    markers = ["o", "s", "^", "D"]

    title_fs = 18
    label_fs = 16
    tick_fs = 12
    legend_fs = 13

    fig, ax = plt.subplots(figsize=(12, 6))

    for k, marker in zip(ks, markers):
        block = dyadic_frequency_block(f, k)
        x = np.arange(block.size) / (block.size - 1)

        ax.scatter(
            x,
            block,
            s=28,
            marker=marker,
            alpha=0.55,
            label=rf"$B_{k}$",
        )

    ax.set_title(
        "Normalized dyadic blocks of the frequency sequence",
        fontsize=title_fs,
        pad=8,
    )
    ax.set_xlabel("normalized position in block", fontsize=label_fs)
    ax.set_ylabel(r"$F(m)$", fontsize=label_fs)
    ax.tick_params(labelsize=tick_fs)

    # Match the original visual range: top tick at 14.
    ax.set_ylim(2.7, 15.0)
    ax.set_yticks([4, 6, 8, 10, 12, 14])

    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=legend_fs)

    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.11, top=0.93)
    fig.savefig(FIGURES_DIR / "dyadic_frequency_blocks_B6_B9.png", dpi=200)
    plt.close(fig)


def plot_normalized_peak_locations(f: np.ndarray) -> None:
    """Generate normalized_peak_locations.png."""
    ks = np.arange(5, 13)
    peak_m = np.array([dominant_peak_location(f, int(k)) for k in ks], dtype=float)
    ratios = peak_m / (2.0**ks)

    title_fs = 18
    label_fs = 16
    tick_fs = 12
    legend_fs = 13

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(ks, ratios, linewidth=1.3, alpha=0.8)
    ax.scatter(ks, ratios, s=85, alpha=0.8, label="empirical peak location")
    ax.axhline(4.0 / 3.0, linestyle="--", linewidth=1.8, label=r"$4/3$")

    ax.set_title("Normalized peak locations", fontsize=title_fs, pad=8)
    ax.set_xlabel(r"dyadic block index $k$", fontsize=label_fs)
    ax.set_ylabel(r"$m_k/2^k$", fontsize=label_fs)
    ax.tick_params(labelsize=tick_fs)

    ax.set_xlim(4.65, 12.35)
    ax.set_ylim(0, 1.5)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="lower left", frameon=False, fontsize=legend_fs)

    # Inset zoom near 4/3, tuned to match the original figure.
    inset = ax.inset_axes([0.60, 0.33, 0.38, 0.35])
    inset.plot(ks, ratios, linewidth=1.1, alpha=0.8)
    inset.scatter(ks, ratios, s=42, alpha=0.8)
    inset.axhline(4.0 / 3.0, linestyle="--", linewidth=1.2)
    inset.set_title("zoom near 4/3", fontsize=12)
    inset.grid(True, alpha=0.22)
    inset.set_xlim(4.7, 12.4)
    inset.set_xticks([5, 7, 9, 11])
    inset.set_ylim(1.328, 1.348)
    inset.set_yticks([1.330, 1.335, 1.340, 1.345])
    inset.tick_params(labelsize=10)

    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.1, top=0.93)
    fig.savefig(FIGURES_DIR / "normalized_peak_locations.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Need frequencies safely beyond m=4000 and blocks through B_12.
    # Q up to 100000 is ample for these plots and still fast.
    q = generate_q(100_000)
    f = frequency_array(q)

    plot_frequency_distribution(f)
    plot_dyadic_frequency_blocks(f)
    plot_normalized_peak_locations(f)


if __name__ == "__main__":
    main()

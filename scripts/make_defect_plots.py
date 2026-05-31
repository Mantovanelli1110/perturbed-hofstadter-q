"""Generate defect-related plots."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from defects import centered_defects, first_differences
from generate_q import generate_q


def main() -> None:
    values = generate_q(4096)
    figures = Path("figures")
    figures.mkdir(parents=True, exist_ok=True)

    diffs = first_differences(values)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(2, len(values) + 1), diffs, linewidth=0.8)
    ax.set_xlabel("n")
    ax.set_ylabel("Q(n) - Q(n-1)")
    ax.set_title("First differences")
    fig.tight_layout()
    fig.savefig(figures / "perturbed_hofstadter_first_differences_clean.png", dpi=200)
    plt.close(fig)

    n = np.arange(3, len(values) + 1)
    left_indices = n - values[1:-1]
    right_indices = n - values[:-2]
    safety_margin = np.minimum(left_indices, right_indices)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(n, safety_margin, linewidth=0.8)
    ax.set_xlabel("n")
    ax.set_ylabel("minimum recurrence index")
    ax.set_title("Recurrence safety margin")
    fig.tight_layout()
    fig.savefig(figures / "perturbed_hofstadter_safety_margin_clean.png", dpi=200)
    plt.close(fig)

    defects = centered_defects(values)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(values) + 1), defects, linewidth=0.8)
    ax.set_xlabel("n")
    ax.set_ylabel("centered defect")
    ax.set_title("Renormalized defects")
    fig.tight_layout()
    fig.savefig(figures / "renormalized_defects.png", dpi=200)
    plt.close(fig)

    scale = np.sqrt(np.arange(1, len(defects) + 1))
    normalized = np.abs(defects) / scale
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(range(1, len(values) + 1), normalized + 1e-12, linewidth=0.8)
    ax.set_xlabel("n")
    ax.set_ylabel("|R(n)| / sqrt(n)")
    ax.set_title("Dyadically normalized R profile")
    fig.tight_layout()
    fig.savefig(figures / "dyadically_normalized_R_logscale.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

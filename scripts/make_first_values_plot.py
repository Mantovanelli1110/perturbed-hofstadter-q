"""Generate firstvalues_4096.png for the perturbed Hofstadter-Q recursion."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from generate_q import generate_q


def main() -> None:
    n_max = 4096
    values = generate_q(n_max)

    n = np.arange(1, n_max + 1)

    output = Path("figures/firstvalues_4096.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Main sequence.
    ax.scatter(n, values, s=5, label=r"$Q(n)$")

    # Reference line n/2.
    ax.plot(n, n / 2.0, linestyle="--", linewidth=1.5, label=r"$n/2$")

    # Dyadic guide lines n = 2^k.
    k = 0
    while 2**k <= n_max:
        ax.axvline(2**k, linewidth=0.7, alpha=0.25)
        k += 1

    ax.set_title("Initial behavior of the perturbed Hofstadter recursion")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$Q(n)$")
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
    fig.savefig(output, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

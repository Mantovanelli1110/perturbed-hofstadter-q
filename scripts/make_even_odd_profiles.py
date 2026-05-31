"""Generate even/odd dyadic profile comparisons."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from defects import dyadic_blocks
from generate_q import generate_q


def main() -> None:
    values = generate_q(4096)
    blocks = dyadic_blocks(values, min_power=6, max_power=9)
    output = Path("figures/even_odd_profiles_dyadic_blocks.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    for power, block in blocks.items():
        even_mean = block[::2].mean()
        odd_mean = block[1::2].mean()
        ax.scatter([power - 0.05, power + 0.05], [even_mean, odd_mean], label=f"B{power}")
    ax.set_xlabel("dyadic block power")
    ax.set_ylabel("mean Q value")
    ax.set_title("Even/odd dyadic block profiles")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()


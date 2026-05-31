"""Generate frequency distribution figures."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from defects import dyadic_blocks
from frequencies import frequency_table
from generate_q import generate_q


def main() -> None:
    values = generate_q(4096)
    figures = Path("figures")
    figures.mkdir(parents=True, exist_ok=True)

    table = frequency_table(values)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(table["value"], table["count"], width=1.0)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    ax.set_title("Frequency distribution")
    fig.tight_layout()
    fig.savefig(figures / "frequency_distribution.png", dpi=200)
    plt.close(fig)

    blocks = dyadic_blocks(values, min_power=6, max_power=9)
    fig, ax = plt.subplots(figsize=(8, 4))
    for power, block in blocks.items():
        block_table = frequency_table(block)
        ax.plot(block_table["value"], block_table["frequency"], label=f"B{power}")
    ax.set_xlabel("value")
    ax.set_ylabel("frequency")
    ax.set_title("Dyadic frequency blocks")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "dyadic_frequency_blocks_B6_B9.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()


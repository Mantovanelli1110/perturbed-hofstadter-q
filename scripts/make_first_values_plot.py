"""Generate the first-values plot."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from generate_q import generate_q


def main() -> None:
    values = generate_q(4096)
    output = Path("figures/firstvalues_4096.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(values) + 1), values, linewidth=0.8)
    ax.set_xlabel("n")
    ax.set_ylabel("Q(n)")
    ax.set_title("First 4096 Hofstadter-Q values")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()


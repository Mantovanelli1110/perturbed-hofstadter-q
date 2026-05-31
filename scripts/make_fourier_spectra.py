"""Generate Fourier spectra for dyadic profiles."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from defects import dyadic_blocks
from fourier_analysis import dominant_frequency, normalized_spectrum
from generate_q import generate_q


def main() -> None:
    values = generate_q(4096)
    blocks = dyadic_blocks(values, min_power=6, max_power=9)
    figures = Path("figures")
    figures.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    peaks = []
    for power, block in blocks.items():
        spectrum = normalized_spectrum(block)
        ax.plot(spectrum["frequency"], spectrum["power"], label=f"B{power}")
        peaks.append((power, *dominant_frequency(block)))
    ax.set_xlabel("normalized frequency")
    ax.set_ylabel("normalized power")
    ax.set_title("Fourier spectra of dyadic profiles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "fourier_spectra_dyadic_profiles.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([p[0] for p in peaks], [p[1] for p in peaks], marker="o")
    ax.set_xlabel("dyadic block power")
    ax.set_ylabel("dominant frequency")
    ax.set_title("Normalized peak locations")
    fig.tight_layout()
    fig.savefig(figures / "normalized_peak_locations.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()


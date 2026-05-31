"""Run all figure-generation scripts."""

from __future__ import annotations

import importlib


SCRIPTS = [
    "make_first_values_plot",
    "make_frequency_plots",
    "make_defect_plots",
    "make_fourier_spectra",
    "make_even_odd_profiles",
]


def main() -> None:
    for module_name in SCRIPTS:
        module = importlib.import_module(module_name)
        module.main()


if __name__ == "__main__":
    main()


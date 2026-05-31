"""Run all figure-generation scripts.

Run from the repository root with

    python scripts/make_all_figures.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "make_first_values_plot",
    "make_frequency_plots",
    "make_defect_plots",
    "make_fourier_spectra",
    "make_even_odd_profiles",
]


def main() -> None:
    sys.path.insert(0, str(SCRIPTS_DIR))

    for module_name in SCRIPTS:
        print(f"Running {module_name}.py ...")
        module = importlib.import_module(module_name)

        if not hasattr(module, "main"):
            raise AttributeError(f"{module_name}.py does not define a main() function.")

        module.main()

    print("All figures generated successfully.")


if __name__ == "__main__":
    main()

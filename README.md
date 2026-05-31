# perturbed-hofstadter-q

Research code and reproducibility scaffolding for experiments around perturbed
Hofstadter-Q sequences, dyadic defect profiles, frequency distributions, and
Fourier diagnostics.

## Repository Layout

```text
perturbed-hofstadter-q/
├── src/        # Core generation, defect, frequency, Fourier, and checks code
├── scripts/    # Figure and result generation entry points
├── figures/    # Generated figures
├── data/       # Optional generated or sampled data
├── results/    # CSV and text summaries
└── paper/      # Optional manuscript assets
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

With conda:

```bash
conda env create -f environment.yml
conda activate perturbed-hofstadter-q
```

## Usage

Generate all figures and result summaries:

```bash
python scripts/make_all_figures.py
```

Run basic verification checks:

```bash
python src/verification_checks.py --n 4096
```

## Notes

The files in `figures/`, `data/`, `results/`, and `paper/` are intended to be
generated or added as the project matures. Large generated datasets should be
compressed and accompanied by checksums in `data/checksums.txt`.


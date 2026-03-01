# src/

Standalone scripts that correspond to specific parts of the paper.
These are also embedded in the notebook, but kept here as separate files for convenience.

| Script | What it does |
|---|---|
| `feature_importance_v2.py` | Computes feature importances across all models with real standard deviation for RandomForest |
| `figure2_generation.py` | Generates Figure 2: CNN normalized mean activation profiles |
| `stats_and_latex.py` | Runs the paired t-tests and Cohen's d comparisons, outputs LaTeX table |

All three scripts expect the baseline models and scaler to already exist in `Models_SOTA_Hybrid/` on Google Drive, so run Experiment 1 first.

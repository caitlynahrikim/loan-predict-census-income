# User Guide

What each part of `capstone-project-code.ipynb` does. See [`SETUP.md`](SETUP.md) first for
environment setup.

```
Raw CSV → Part 1: Load → Part 2: Define problem → Part 3: EDA → Part 4: Prepare data
                                                                        │
                                          ┌─────────────────────────────┴───────────────┐
                                          ▼                                               ▼
                              Part 5: Logistic Regression                    Part 6: Neural Network
                                          └─────────────────────┬─────────────────────────┘
                                                                 ▼
                                                      Part 7: Compare & reflect
```

**Part 1 — Load:** reads `censusData.csv` into a DataFrame (32,561 × 15).

**Part 2 — Define the problem:** markdown only; documents the label (`income_binary`),
candidate features, and business rationale.

**Part 3 — EDA:** checks class balance, summary stats, dtypes, missingness, and a
correlation matrix against the label (used to drop `fnlwgt`). Ends with a written summary
of data-quality issues and ethical/fairness considerations.

**Part 4 — Data prep** (most worth reading closely if adapting this to a new dataset):
1. Drop `fnlwgt` (a sampling weight, not a person-level attribute).
2. Fill categorical nulls (`workclass`, `occupation`, `native-country`) with `"Unknown"`.
3. Drop rows with missing numeric values (`age`, `hours-per-week`).
4. Bucket `native-country` to top 8 values + `"Other"`.
5. One-hot encode remaining categoricals; reassemble into `df_final` and `feature_list`.

**Part 5 — Logistic regression:** split data (80/20, `random_state=1234`); train a baseline
via `train_test_LR`; tune `C` with 5-fold `GridSearchCV`; refit the final model; evaluate
with accuracy, F1, a confusion matrix, and a coefficient plot.

**Part 6 — Neural network:** (re-run Parts 1–5 first if using a fresh kernel). Scale
features with `StandardScaler`; build a `Sequential` model (3 hidden layers, 64→32→16,
ReLU; sigmoid output); compile with SGD (`lr=0.001`) and binary cross-entropy; train 100
epochs with a 20% validation split and a custom logging callback; plot loss/accuracy curves
to check for overfitting; evaluate on the test set (threshold predictions at 0.5).

**Part 7 — Compare:** builds a summary table of both models' accuracy/F1, then reflects on
whether the neural network's complexity was justified and what to try next.

## Reproducing Outside the Notebook

```bash
python3 scripts/repro_viz.py
```

Reruns the data prep + logistic regression steps and regenerates the supporting plots. See
[`API.md`](API.md) for details.

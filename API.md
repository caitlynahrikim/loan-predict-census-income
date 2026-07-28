# API Reference

This is a notebook-based analysis, not an installable package — this doc references the
reusable functions/objects in `capstone-project-code.ipynb` and `scripts/repro_viz.py`.

## Data Schema

`censusData.csv` — 32,561 rows × 15 columns.

| Column | Type | Notes |
|---|---|---|
| `age`, `hours-per-week` | numeric | rows with nulls dropped |
| `workclass`, `occupation`, `native-country` | categorical | nulls filled `"Unknown"` |
| `fnlwgt` | numeric | dropped (sampling weight, not a person attribute) |
| `education`, `education-num`, `marital-status`, `relationship`, `race`, `sex_selfID` | — | no missing values |
| `capital-gain`, `capital-loss` | numeric | |
| `income_binary` | label | `"<=50K"`/`">50K"` → mapped to `1`/`0` |

After prep: `df_final` has 32,075 rows × 69 columns (68 features + label).

## Key Functions

**`train_test_LR(X_train, y_train, X_test, y_test, c=1)`**
Fits a `LogisticRegression(C=c)` and returns `(loss, acc, f1)` on the test set.
```python
loss, acc, f1 = train_test_LR(X_train, y_train, X_test, y_test, c=1)
```

**`GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)`**
Selects `C` from `{0.01, 0.1, 1, 10, 100}` via 5-fold CV. Best value:
`grid_search.best_params_['C']`.

**`ProgBarLoggerNEpochs(num_epochs, every_n=50)`**
Keras callback that logs metrics every `every_n` epochs instead of every epoch.
```python
history = nn_model.fit(
    X_train_scaled, y_train, epochs=100, validation_split=0.2, verbose=0,
    callbacks=[ProgBarLoggerNEpochs(num_epochs=100, every_n=5)],
)
```

**Neural network architecture (`nn_model`):**

| Layer | Units | Activation |
|---|---|---|
| Input | 68 | — |
| Dense × 3 | 64 → 32 → 16 | ReLU |
| Output | 1 | Sigmoid |

Compiled with `SGD(learning_rate=0.001)`, `BinaryCrossentropy(from_logits=False)`,
metric `accuracy`.

## `scripts/repro_viz.py`

Standalone reproduction of the data prep + logistic regression pipeline.

```bash
python3 scripts/repro_viz.py
```

Prints best `C`, final test accuracy, and F1 to stdout; writes `class_distribution.png`,
`confusion_matrix.png`, and `lr_coefficients.png` to `docs/images/`. Does not cover the
neural network, which currently only exists in the notebook.

## Adapting This Pipeline

1. Load your data with a binary label column.
2. Fill categorical nulls with a placeholder; drop rows with missing numeric values.
3. Drop any non-attribute feature (this dataset's equivalent was `fnlwgt`).
4. Bucket high-cardinality categoricals (top N + `"Other"`) before one-hot encoding.
5. Split with a fixed `random_state` for reproducibility.
6. Start with `train_test_LR` as a baseline before reaching for more complex models — this
   project's core finding was that added complexity wasn't worth it.

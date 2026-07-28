# Setup Guide

## Prerequisites

- Python 3.9–3.11 (for TensorFlow/Keras compatibility)
- `pip`, `venv`, Git

## 1. Clone and Create a Virtual Environment

```bash
git clone https://github.com/caitlynahrikim/loan-predict-census-income.git
cd loan-predict-census-income
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

The dataset (`censusData.csv`) is already included — no separate download needed.

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Installs pandas, numpy, matplotlib, seaborn, scikit-learn, tensorflow, and jupyter.
`tensorflow` is the heaviest install — if you only need the logistic regression portion,
you can drop it from `requirements.txt`.

Verify:
```bash
python3 -c "import pandas, sklearn, tensorflow; print('OK')"
```

## 3. Train the Models

```bash
jupyter notebook capstone-project-code.ipynb
```

Run cells **top to bottom**, in order:

1. **Parts 1–4** — load data, EDA, data prep (produces the feature matrix both models use).
2. **Part 5** — logistic regression: baseline fit, `GridSearchCV` tuning, final fit (fast).
3. **Part 6** — neural network: scaling, architecture, training (~2–3 min for 100 epochs).
   If starting a fresh kernel session, use **Kernel → Restart & Run All** first — variables
   don't persist between sessions.
4. **Part 7** — side-by-side model comparison.

Non-interactive alternative:
```bash
jupyter nbconvert --to notebook --execute capstone-project-code.ipynb --output executed.ipynb
```

## 4. Evaluate Performance

Metrics print inline as each model trains: accuracy/F1/confusion matrix for logistic
regression (Part 5), accuracy/F1 plus training curves for the neural network (Part 6), and
a combined comparison table (Part 7).

To reproduce just the logistic regression results and plots without Jupyter:
```bash
python3 scripts/repro_viz.py
```

> Exact metrics can vary slightly (~0.5%) by scikit-learn version. The train/test split is
> seeded (`random_state=1234`), so the split itself is reproducible.

## Troubleshooting

| Issue | Fix |
|---|---|
| `ConvergenceWarning` from `LogisticRegression` | Expected — model still trains fine. Increase `max_iter` to silence it. |
| `NameError` in Part 6 | Kernel restarted or cells run out of order — use **Kernel → Restart & Run All**. |
| `ModuleNotFoundError: tensorflow` | `pip install tensorflow` or re-run `pip install -r requirements.txt`. |
| `OneHotEncoder` `sparse` error | Newer scikit-learn renamed `sparse` → `sparse_output`. |

Next: [`USER_GUIDE.md`](USER_GUIDE.md) for a walkthrough, or [`API.md`](API.md) for function
reference.

# Loan Predict: Census Income Classification

Predicting whether an individual's annual income exceeds $50,000 using 1994 U.S. Census
data, to support automated eligibility screening for nonprofit lending programs.

## Project Overview

Frames income prediction as binary classification: given demographic and employment
attributes, predict whether income is `<=50K` or `>50K`/year. The motivating scenario is a
fintech company (EquiLend) building a model for a CDFI to screen loan/financial-education
applicants — one whose predictions the CDFI can explain to applicants.

Dataset: `censusData.csv` 32,561 records, numeric features (age, education-num,
capital-gain/loss, hours-per-week) and categorical features (workclass, education,
marital-status, occupation, relationship, race, sex, native-country), label
`income_binary`.

## Objectives and Goals

- Build a classifier predicting whether income exceeds $50,000/year.
- Compare an interpretable model (logistic regression) against a neural network to see if
  added complexity is justified.
- Favor outputs a nonprofit can explain to applicants.
- Surface fairness/ethical risks from historical bias baked into 1994-era data.

## Methodology

1. **EDA** — checked class balance (imbalanced toward `<=50K`), missing values, and
   correlation with the label; `fnlwgt` showed almost no relationship and was dropped.
2. **Data prep** — filled categorical nulls (`workclass`, `occupation`, `native-country`)
   with `"Unknown"`; dropped rows with missing numeric values (`age`, `hours-per-week`);
   bucketed `native-country` to top 8 + `"Other"`; one-hot encoded categoricals; scaled
   features for the neural network only.
3. **Modeling** — logistic regression tuned via 5-fold `GridSearchCV` over `C`; a
   3-hidden-layer (64→32→16, ReLU) Keras neural network with sigmoid output, SGD optimizer,
   trained 100 epochs.
4. **Evaluation** — accuracy and F1 on a held-out 20% test set, plus a confusion matrix and
   coefficient inspection for the logistic regression model.

## Results and Key Findings

| Metric   | Logistic Regression | Neural Network |
|----------|---------------------|-----------------|
| Accuracy | 0.852               | 0.853           |
| F1 Score | 0.905               | 0.906           |

- The two models performed almost identically — the neural network's added complexity
  wasn't justified by its marginal gain.
- **Logistic regression was selected**: comparable performance, but interpretable
  coefficients matter for a nonprofit that must explain eligibility decisions.
- The model misses more `>50K` cases than `<=50K` cases — a safer failure mode for a
  screening tool (more likely to over-flag than wrongly reject).
- The neural network showed mild overfitting after ~epoch 20 (validation loss plateaued
  while training loss kept dropping).
- **Fairness concern:** sex was one of the strongest predictors toward `<=50K`, meaning the
  model learned historical income disparities rather than correcting for them.
  `native-country`, `marital-status`, and `relationship` are also likely proxies for
  protected characteristics.

## Installation and Quick Start

```bash
git clone https://github.com/caitlynahrikim/loan-predict-census-income.git
cd loan-predict-census-income
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook capstone-project-code.ipynb
```

Full instructions, including training and evaluation steps: [`setup.md`](setup.md).

## Documentation

| Document | Description |
|---|---|
| [`setup.md`](setup.md) | Local setup, dependencies, running the notebook. |
| [`user-guide.md`](user-guide.md) | What each notebook section does. |
| [`api-doc.md`](api-doc.md) | Reference for reusable functions and data schema. |

## Potential Next Steps

- Address class imbalance directly (oversampling or class-weighted loss).
- Audit fairness across subgroups (sex, race, native-country error rates).
- Retrain on more recent data — 1994 labor-market conditions are dated.
- Use early stopping for the neural network (~20 epochs matches 100-epoch performance).
- Try additional models (decision trees, KNN) to confirm LR is the right trade-off.

## Individual Contributions

Completed individually as a course capstone. All EDA, data prep, modeling, tuning,
evaluation, and analysis were done by the author.

## AI Use Attestation

Claude was used to brainstorm data-cleaning approaches, help design the coefficient
visualization, and debug cleaning code. All decisions, code, and analysis were reviewed and
verified by the author. Full reflection is in the notebook's final section.

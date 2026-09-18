# ML Capstone Project

Course: 23CSE301 Machine Learning (B.Tech. CSE, III year), Academic Year 2026-27

End-to-end machine learning pipelines for the three capstone tracks: data loading,
EDA, cleaning and feature engineering, model training and comparison,
hyperparameter tuning and result visualisation. Review 1 (full regression track
plus classification Part A) is complete; classification Part B and the clustering
track are scheduled for Review 2.

| Track | Dataset | Status |
|---|---|---|
| Regression | 200+ Financial Indicators of US Stocks (2018 slice) | Complete: all 10 algorithms, final comparison, top-2 cross-validation |
| Classification | NASA Kepler Objects of Interest (KOI) | Part A complete: all 5 algorithms; Part B in Review 2 |
| Clustering | Assigned for Review 2 | Not started |

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .streamlit/config.toml       # theme for the Streamlit app
├── data/                        # raw CSVs (not committed), see data/README.md
├── notebooks/
│   ├── regression.ipynb
│   ├── classification.ipynb
│   ├── clustering.ipynb
│   └── figures/                 # saved plots, one folder per model plus Comparison/
├── models/                      # saved best models (.pkl), built by app/export_models.py
└── app/
    ├── app.py                   # Streamlit interface
    └── export_models.py         # rebuilds models/ with the notebooks' exact preprocessing
```

## Regression track

**Problem statement.** Given a US company's annual financial statement data
(income statement, balance sheet, cash flow and derived ratios), predict the
percentage change in its share price over the following year.

**Dataset.** [200+ Financial Indicators of US Stocks (2014-2018)](https://www.kaggle.com/datasets/cnic92/200-financial-indicators-of-us-stocks-20142018)
on Kaggle (cnic92). We use the 2018 file: 4,392 companies and 225 columns. The
target is `2019 PRICE VAR [%]`, the price variation during 2019. The `class`
column is excluded as a feature because it is derived from the target. Place the
file at `data/2018_Financial_Data.csv` (it is not committed).

**Pipeline.** Drop near-empty columns, keep 41 curated non-redundant indicators,
winsorise at the 1st/99th percentile, stratified 80/20 split (`random_state=42`),
one-hot encode `Sector`, median imputation and scaling fitted on the training
split only, and one engineered feature
(`Growth Adjusted Margin = Revenue Growth x Net Profit Margin`).

**Results.** Held-out test set, identical split for every model, ranked by R2.

| Model | R2 | RMSE | MAE |
|---|---|---|---|
| Random Forest Regressor | 0.111 | 45.53 | 31.96 |
| Gradient Boosting Regressor | 0.100 | 45.80 | 31.90 |
| ElasticNet Regression | 0.040 | 47.30 | 33.32 |
| Ridge Regression | 0.039 | 47.33 | 33.36 |
| SVR | 0.039 | 47.33 | 31.99 |
| Lasso Regression | 0.038 | 47.34 | 33.36 |
| Decision Tree Regressor | 0.037 | 47.38 | 33.30 |
| KNN Regressor | 0.036 | 47.40 | 32.84 |
| Linear Regression | 0.034 | 47.45 | 33.47 |
| Polynomial Regression | 0.016 | 47.89 | 33.15 |

5-fold cross-validated R2 for the top two models: Random Forest 0.061,
Gradient Boosting 0.074. Every model lands at a low R2, which is the honest
finding for this problem: annual fundamentals explain only a small share of
next-year price moves, and the tree ensembles capture the most of what is there.

## Classification track

**Problem statement.** Classify a Kepler Object of Interest as CONFIRMED,
CANDIDATE or FALSE POSITIVE from its transit signal and host-star measurements.

**Dataset.** [NASA Kepler Objects of Interest](https://data.nasa.gov/dataset/kepler-objects-of-interest-koi)
cumulative table: 9,564 objects and 153 columns. The notebook downloads the CSV
automatically from the Exoplanet Archive and saves it to `data/kepler_koi.csv`
(not committed). The target is `koi_disposition`; the disposition score and the
false-positive flags are excluded from the features to avoid leakage.

**Pipeline.** 99 numeric features plus an engineered `transit_depth_per_hour`,
stratified 80/20 split, median imputation and standard scaling inside each model
pipeline (fitted on training folds only), weighted precision/recall/F1 and
one-vs-rest ROC-AUC.

**Results, Part A.** Held-out test set, ranked by weighted F1.

| Model | Accuracy | Weighted F1 | ROC-AUC (OvR) |
|---|---|---|---|
| Support Vector Classifier | 0.822 | 0.820 | 0.944 |
| Decision Tree (tuned) | 0.821 | 0.819 | 0.923 |
| Logistic Regression | 0.820 | 0.813 | 0.937 |
| Decision Tree (baseline) | 0.793 | 0.794 | 0.838 |
| K-Nearest Neighbors (tuned) | 0.785 | 0.782 | 0.913 |
| K-Nearest Neighbors (baseline) | 0.774 | 0.771 | 0.901 |
| Gaussian Naive Bayes | 0.614 | 0.619 | 0.877 |

Part B (Random Forest, AdaBoost, Gradient Boosting, Bagging and MLP classifiers)
and the consolidated 10-algorithm table are Review 2 deliverables.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put the datasets in `data/` (see `data/README.md`), then open the notebooks and
run every cell top to bottom:

```bash
jupyter notebook
```

## Interactive app (Streamlit)

A web interface serves every Review 1 model: all ten regressors for next-year
stock returns and all seven classifier runs for Kepler objects. Each page has a
model selector (the best model on the held-out split is chosen by default) and
an "All models" view that scores the same inputs with every model at once.

```bash
python app/export_models.py   # rebuilds models/ using the notebooks' exact preprocessing
streamlit run app/app.py
```

`app/export_models.py` needs both datasets in `data/`. It refits every model
with the hyperparameters the notebooks' grid searches picked, checks that each
one reproduces the notebook's held-out score, and saves them together with
everything the app draws from: training medians and percentiles, per-class
typical values, each model's confusion matrix, per-class scores and test-set
predictions, sector statistics and a sample of training objects. The app loads
these from `models/`, so it runs without the raw datasets. The two bundles
total about 15 MB and are committed so the deployed app can load them.

Pages:

- **Exoplanet classifier** (space-themed banner): pick a classifier, enter a transit
  signal and host-star measurements, get the predicted disposition with class
  probabilities, then see the object among 2,500 known Kepler objects, a radar
  comparison with typical objects of each class, what every classifier says about
  the same object, and the selected model's confusion matrix and per-class scores.
- **Stock return predictor**: pick a regressor, enter a company's fundamentals and
  sector, get the predicted next-year change on a dial, then see where it lands
  among the training companies, how much each input moves the prediction
  (one-at-a-time sensitivity), what every regressor predicts, typical returns by
  sector, and the selected model's actual-vs-predicted on the held-out set.
- **About the models**: datasets, preprocessing, the full comparison charts and tables
  for both tracks, and run instructions.

The sidebar has a dark/light toggle. It switches Streamlit's theme at runtime
for the whole app (the setting is process-wide, so every open tab follows the
last toggle). Fonts are loaded from jsDelivr; without internet the app falls
back to the system font and still works.

### Deployment (Streamlit Community Cloud)

The two model bundles in `models/` are committed (they are the only `.pkl`
files not ignored), so the app deploys straight from this repository without
the raw datasets. On [share.streamlit.io](https://share.streamlit.io) choose
this repository, branch `main`, main file `app/app.py`, and Python 3.13 under
advanced settings. `requirements.txt` pins scikit-learn to 1.9.0, the exact
version the models were saved with, so they load without version warnings.

## Reproducibility notes

- `random_state=42` everywhere, with a consistent stratified 80/20 split within each track.
- All scalers, encoders and imputers are fitted on the training split only.
- Every notebook runs top to bottom without errors; figures are saved under `notebooks/figures/`.

## AI Assistance Disclosure

Generative AI/Codex was used for code scaffolding and development assistance.
Dataset interpretation, analytical observations, feature-engineering
justification, and review/viva explanations must be completed and verified by
the student team.

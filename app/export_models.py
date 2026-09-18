"""Export the two best capstone models plus the preprocessing artifacts the
Streamlit app needs. This re-creates the exact notebook preprocessing so the
app predicts with the same models reported in the notebooks.

Run from the repo root:
    python app/export_models.py

Needs data/2018_Financial_Data.csv and data/kepler_koi.csv (see data/README.md).
Writes models/regression_random_forest.pkl and models/classification_svc.pkl.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

REG_FEATURES = [
    'Revenue', 'Revenue Growth', 'Gross Profit', 'Operating Income', 'EBITDA', 'EBIT', 'Net Income', 'EPS',
    'Gross Margin', 'EBITDA Margin', 'operatingProfitMargin', 'netProfitMargin',
    'Total assets', 'Total liabilities', 'Total shareholders equity', 'Total current assets',
    'Total current liabilities', 'Total debt', 'Cash and cash equivalents',
    'currentRatio', 'quickRatio', 'cashRatio',
    'debtEquityRatio', 'debtRatio', 'interestCoverage',
    'assetTurnover', 'inventoryTurnover', 'returnOnAssets', 'returnOnEquity', 'ROIC',
    'Operating Cash Flow', 'Free Cash Flow', 'Capital Expenditure',
    'PE ratio', 'PB ratio', 'EV to Sales', 'Price to Sales Ratio', 'Dividend Yield',
    'EPS Growth', 'Net Income Growth', 'Asset Growth',
]
REG_KEY_FEATURES = [
    'EPS', 'Total assets', 'returnOnEquity', 'PB ratio', 'Total current assets',
    'Total shareholders equity', 'Total liabilities', 'returnOnAssets', 'Price to Sales Ratio',
    'Asset Growth', 'Revenue Growth', 'Net Income Growth', 'ROIC', 'netProfitMargin',
]
CLS_KEY_FEATURES = [
    'koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_model_snr', 'koi_impact',
    'koi_teq', 'koi_steff', 'koi_slogg', 'koi_srad', 'koi_num_transits', 'koi_kepmag',
]


def export_regression():
    df = pd.read_csv(DATA / "2018_Financial_Data.csv")
    df_clean = df.drop(columns=df.columns[df.isnull().mean() > 0.9].tolist())
    target = "2019 PRICE VAR [%]"
    data = df_clean[["Unnamed: 0", "Sector"] + REG_FEATURES + [target]].copy()

    #winsorise target and features at the 1st/99th percentile, exactly as the notebook
    t_lo, t_hi = data[target].quantile([0.01, 0.99])
    data[target] = data[target].clip(t_lo, t_hi)
    clip_bounds = {}
    for c in REG_FEATURES:
        lo, hi = data[c].quantile([0.01, 0.99])
        data[c] = data[c].clip(lo, hi)
        clip_bounds[c] = (float(lo), float(hi))

    data["target_bin"] = pd.qcut(data[target], q=5, labels=False, duplicates="drop")
    X = data[["Sector"] + REG_FEATURES]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=data["target_bin"])

    sector_train = X_train["Sector"].copy()
    X_train = pd.get_dummies(X_train, columns=["Sector"], drop_first=True)
    X_test = pd.get_dummies(X_test, columns=["Sector"], drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    #percentile grid of each raw indicator on the training split, used by the app's sensitivity chart
    grid = np.linspace(0, 1, 101)
    feature_quantiles = {f: X_train[f].quantile(grid).to_numpy(dtype=float).tolist() for f in REG_FEATURES}
    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)
    X_train["Growth Adjusted Margin"] = X_train["Revenue Growth"] * X_train["netProfitMargin"]
    X_test["Growth Adjusted Margin"] = X_test["Revenue Growth"] * X_test["netProfitMargin"]

    #tuned random forest, the best model in the regression comparison table
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_split=2,
                               random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)
    pred = rf.predict(X_test)
    metrics = {"R2": float(r2_score(y_test, pred)),
               "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
               "MAE": float(mean_absolute_error(y_test, pred))}
    importance = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    by_sector = (pd.DataFrame({"sector": sector_train.to_numpy(), "y": y_train.to_numpy()})
                 .groupby("sector")["y"].agg(["median", "mean", "count"]))

    bundle = {
        "model": rf,
        "raw_features": REG_FEATURES,
        "key_features": [f for f in REG_KEY_FEATURES if f in REG_FEATURES],
        "model_columns": list(X_train.columns),
        "sector_categories": sorted(df_clean["Sector"].dropna().unique().tolist()),
        "sector_dummy_columns": [c for c in X_train.columns if c.startswith("Sector_")],
        "clip_bounds": clip_bounds,
        "train_medians": {f: float(train_medians[f]) for f in REG_FEATURES},
        "target": target,
        "target_clip": (float(t_lo), float(t_hi)),
        "target_stats": {"median": float(y_train.median()),
                         "q25": float(y_train.quantile(0.25)),
                         "q75": float(y_train.quantile(0.75)),
                         "p10": float(y_train.quantile(0.10)),
                         "p90": float(y_train.quantile(0.90))},
        "metrics": metrics,
        "y_train": y_train.to_numpy(dtype=float),
        "y_test": y_test.to_numpy(dtype=float),
        "y_pred_test": pred.astype(float),
        "feature_quantiles": feature_quantiles,
        "sector_stats": {s: {"median": float(r["median"]), "mean": float(r["mean"]), "count": int(r["count"])}
                         for s, r in by_sector.iterrows()},
        "sector_counts": {str(k): int(v) for k, v in df_clean["Sector"].value_counts().items()},
        "feature_importance": {k: float(v) for k, v in importance.head(15).items()},
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    joblib.dump(bundle, MODELS / "regression_random_forest.pkl")
    return metrics


def export_classification():
    df = pd.read_csv(DATA / "kepler_koi.csv")
    wdf = df.drop_duplicates().copy()
    if {"koi_depth", "koi_duration"}.issubset(wdf.columns):
        wdf["transit_depth_per_hour"] = wdf["koi_depth"] / wdf["koi_duration"].replace(0, np.nan)
    target = "koi_disposition"
    mdf = wdf.dropna(subset=[target]).copy()
    excluded = [target, "kepid", "koi_pdisposition", "koi_score",
                "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]
    excluded = [c for c in excluded if c in mdf.columns]
    numeric = mdf.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric if c not in excluded]
    features = [c for c in features if not mdf[c].isna().all() and mdf[c].nunique(dropna=True) > 1]

    X = mdf[features].copy()
    y = mdf[target].astype(str).copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    preprocessor = ColumnTransformer(
        transformers=[("numeric", Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                                                  ("scaler", StandardScaler())]), features)],
        remainder="drop", verbose_feature_names_out=False)
    #tuned svc, the best model in the classification comparison table
    pipe = Pipeline(steps=[("preprocess", preprocessor),
                           ("model", SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                                         random_state=RANDOM_STATE))]).fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)
    classes = [str(c) for c in pipe.classes_]
    y_bin = label_binarize(y_test, classes=classes)
    metrics = {"Accuracy": float(accuracy_score(y_test, pred)),
               "F1_Weighted": float(f1_score(y_test, pred, average="weighted")),
               "ROC_AUC_OvR": float(roc_auc_score(y_bin, proba, multi_class="ovr", average="weighted"))}
    report = classification_report(y_test, pred, labels=classes, output_dict=True)
    per_class = {c: {"precision": float(report[c]["precision"]), "recall": float(report[c]["recall"]),
                     "f1": float(report[c]["f1-score"]), "support": int(report[c]["support"])} for c in classes}
    confusion = confusion_matrix(y_test, pred, labels=classes).astype(int).tolist()

    #typical values per class and the percentile grid of each key feature, for the app's radar chart
    key_features = [f for f in CLS_KEY_FEATURES if f in features]
    grid = np.linspace(0, 1, 101)
    class_medians = {c: {f: float(X_train.loc[y_train == c, f].median()) for f in key_features} for c in classes}
    feature_quantiles = {f: X_train[f].quantile(grid).to_numpy(dtype=float).tolist() for f in key_features}

    #small sample of training objects for the app's scatter chart (log axes need positives)
    scatter_cols = [c for c in ["koi_period", "koi_depth", "koi_prad", "koi_model_snr"] if c in features]
    scatter_sample = X_train[scatter_cols].assign(koi_disposition=y_train.values).dropna()
    scatter_sample = scatter_sample[(scatter_sample["koi_period"] > 0) & (scatter_sample["koi_depth"] > 0)]
    scatter_sample = scatter_sample.sample(n=min(2500, len(scatter_sample)), random_state=RANDOM_STATE).reset_index(drop=True)

    bundle = {
        "pipeline": pipe,
        "feature_columns": features,
        "key_features": key_features,
        "classes": classes,
        "feature_medians": {f: float(X_train[f].median()) for f in features},
        "metrics": metrics,
        "per_class": per_class,
        "confusion": confusion,
        "class_medians": class_medians,
        "feature_quantiles": feature_quantiles,
        "scatter_sample": scatter_sample,
        "class_counts": {str(k): int(v) for k, v in y_train.value_counts().items()},
        "class_counts_all": {str(k): int(v) for k, v in y.value_counts().items()},
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    joblib.dump(bundle, MODELS / "classification_svc.pkl")
    return metrics


if __name__ == "__main__":
    reg = export_regression()
    print("regression  (tuned Random Forest):", {k: round(v, 4) for k, v in reg.items()})
    cls = export_classification()
    print("classification (tuned SVC):       ", {k: round(v, 4) for k, v in cls.items()})
    for p in sorted(MODELS.glob("*.pkl")):
        print(f"saved {p.relative_to(ROOT)}  ({p.stat().st_size / 1e6:.1f} MB)")

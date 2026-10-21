"""Export every notebook model plus the preprocessing artifacts the Streamlit app
needs. This re-creates the exact notebook preprocessing and hyperparameters so
the app predicts with the same models reported in the notebooks.

Run from the repo root:
    python app/export_models.py

Needs data/2018_Financial_Data.csv and data/kepler_koi.csv (see data/README.md).
Writes models/regression_models.pkl, models/classification_models.pkl and models/clustering_kmeans.pkl.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from scipy.cluster.hierarchy import cut_tree, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, calinski_harabasz_score, classification_report, confusion_matrix,
                             davies_bouldin_score, f1_score, mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, label_binarize
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

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

#test-set scores the notebooks report; the export checks that every refitted model reproduces them
NOTEBOOK_REGRESSION = {
    "Random Forest": 0.1106, "Gradient Boosting": 0.0999, "ElasticNet": 0.0399, "Ridge": 0.0390, "SVR": 0.0390,
    "Lasso": 0.0385, "Decision Tree": 0.0367, "KNN": 0.0359, "Linear": 0.0339, "Polynomial": 0.0158,
}
NOTEBOOK_CLASSIFICATION = {
    "Random Forest (tuned)": 0.8593, "Random Forest (baseline)": 0.8600,
    "Gradient Boosting (tuned)": 0.8532, "Gradient Boosting (baseline)": 0.8612,
    "AdaBoost (tuned)": 0.8264, "AdaBoost (baseline)": 0.8022,
    "SVC": 0.8197, "Decision Tree (tuned)": 0.8190, "Logistic Regression": 0.8132, "Decision Tree (baseline)": 0.7938,
    "KNN (tuned)": 0.7822, "KNN (baseline)": 0.7705, "Gaussian Naive Bayes": 0.6186,
    "MLP (tuned)": 0.8124, "MLP (baseline)": 0.8043,
}


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
    features_final = REG_FEATURES + ["Growth Adjusted Margin"]

    def scaled(est):
        #the notebook scaled the numeric columns only and left the sector dummies as 0/1
        return Pipeline([("scale", ColumnTransformer([("num", StandardScaler(), features_final)], remainder="passthrough",
                                                     verbose_feature_names_out=False)), ("model", est)])

    #every regressor with the hyperparameters the notebook's grid searches picked
    specs = [
        ("Random Forest", RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_split=2, random_state=RANDOM_STATE, n_jobs=-1),
         "100 trees, max depth 8", "tree"),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE),
         "200 trees, depth 2, learning rate 0.05", "tree"),
        ("ElasticNet", scaled(ElasticNet(alpha=0.1, l1_ratio=0.9, random_state=RANDOM_STATE, max_iter=10000)), "alpha 0.1, l1 ratio 0.9", "linear"),
        ("Ridge", scaled(Ridge(alpha=50, random_state=RANDOM_STATE)), "alpha 50", "linear"),
        ("SVR", Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf", C=10, gamma="scale"))]), "RBF kernel, C 10", "none"),
        ("Lasso", scaled(Lasso(alpha=0.1, random_state=RANDOM_STATE, max_iter=5000)), "alpha 0.1", "linear"),
        ("Decision Tree", DecisionTreeRegressor(max_depth=3, min_samples_leaf=4, min_samples_split=2, random_state=RANDOM_STATE),
         "max depth 3, min leaf 4", "tree"),
        ("KNN", Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsRegressor(n_neighbors=31, weights="distance", metric="manhattan"))]),
         "k 31, manhattan, distance weights", "none"),
        ("Linear", scaled(LinearRegression()), "ordinary least squares", "linear"),
        ("Polynomial", Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)), ("scaler", StandardScaler()),
                                 ("ridge", Ridge(alpha=10000, random_state=RANDOM_STATE))]), "degree 2, ridge alpha 10000", "none"),
    ]
    models = {}
    for name, est, params, kind in specs:
        est.fit(X_train, y_train)
        pred = est.predict(X_test)
        metrics = {"R2": float(r2_score(y_test, pred)),
                   "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                   "MAE": float(mean_absolute_error(y_test, pred))}
        if kind == "tree":
            weights = pd.Series(est.feature_importances_, index=X_train.columns)
            weight_label = "Feature importance, share of the total"
        elif kind == "linear":
            #coefficients on standardised features are comparable, the 0/1 sector dummies are not, so they are left out
            coef = pd.Series(est.named_steps["model"].coef_, index=list(est.named_steps["scale"].get_feature_names_out()))
            weights = coef[features_final].abs()
            weights = weights / weights.sum()
            weight_label = "Standardised coefficient size, share of the total"
        else:
            weights, weight_label = None, None
        models[name] = {
            "model": est, "metrics": metrics, "params": params, "kind": kind,
            "y_pred_test": pred.astype(float),
            "weights": None if weights is None else {k: float(v) for k, v in weights.sort_values(ascending=False).head(15).items()},
            "weight_label": weight_label,
        }
        flag = "" if abs(metrics["R2"] - NOTEBOOK_REGRESSION[name]) < 0.0005 else "   <-- differs from the notebook"
        print(f"  {name:18s} R2={metrics['R2']:.4f}  RMSE={metrics['RMSE']:.2f}  MAE={metrics['MAE']:.2f}{flag}")

    order = sorted(models, key=lambda n: -models[n]["metrics"]["R2"])
    by_sector = (pd.DataFrame({"sector": sector_train.to_numpy(), "y": y_train.to_numpy()})
                 .groupby("sector")["y"].agg(["median", "mean", "count"]))
    bundle = {
        "models": models,
        "model_order": order,
        "best": order[0],
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
        "y_train": y_train.to_numpy(dtype=float),
        "y_test": y_test.to_numpy(dtype=float),
        "feature_quantiles": feature_quantiles,
        "sector_stats": {s: {"median": float(r["median"]), "mean": float(r["mean"]), "count": int(r["count"])}
                         for s, r in by_sector.iterrows()},
        "sector_counts": {str(k): int(v) for k, v in df_clean["Sector"].value_counts().items()},
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    joblib.dump(bundle, MODELS / "regression_models.pkl", compress=3)
    return {n: models[n]["metrics"]["R2"] for n in order}


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
    classes = sorted(y.unique().tolist())
    y_bin = label_binarize(y_test, classes=classes)

    #every classifier in the notebook (part a, plus part b as sections land) with the settings it used or its grid search picked
    specs = [
        ("Random Forest (tuned)", RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=5, random_state=RANDOM_STATE),
         "300 trees, max depth 20, min split 5"),
        ("Random Forest (baseline)", RandomForestClassifier(random_state=RANDOM_STATE), "default 100 trees, unlimited depth"),
        ("AdaBoost (tuned)", AdaBoostClassifier(n_estimators=200, learning_rate=1.0, random_state=RANDOM_STATE),
         "200 estimators, learning rate 1.0"),
        ("AdaBoost (baseline)", AdaBoostClassifier(random_state=RANDOM_STATE), "default 50 estimators, learning rate 1.0"),
        ("Gradient Boosting (tuned)", GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=2,
                                                                 random_state=RANDOM_STATE),
         "150 stages, learning rate 0.1, max depth 2"),
        ("Gradient Boosting (baseline)", GradientBoostingClassifier(random_state=RANDOM_STATE),
         "default 100 stages, learning rate 0.1, max depth 3"),
        ("SVC", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=RANDOM_STATE), "RBF kernel, C 10"),
        ("Decision Tree (tuned)", DecisionTreeClassifier(criterion="gini", max_depth=8, min_samples_leaf=1, min_samples_split=2,
                                                         random_state=RANDOM_STATE), "gini, max depth 8"),
        ("Logistic Regression", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE), "L2 penalty, 2000 iterations"),
        ("Decision Tree (baseline)", DecisionTreeClassifier(random_state=RANDOM_STATE), "unlimited depth"),
        ("KNN (tuned)", KNeighborsClassifier(n_neighbors=5, metric="manhattan"), "k 5, manhattan"),
        ("KNN (baseline)", KNeighborsClassifier(n_neighbors=5, metric="minkowski"), "k 5, minkowski"),
        ("Gaussian Naive Bayes", GaussianNB(), "default settings"),
        ("MLP (tuned)", MLPClassifier(hidden_layer_sizes=(128,), activation="relu", alpha=0.1, solver="adam", early_stopping=True,
                                      validation_fraction=0.1, n_iter_no_change=15, max_iter=400, random_state=RANDOM_STATE),
         "128 relu units, alpha 0.1, early stopping"),
        ("MLP (baseline)", MLPClassifier(max_iter=500, random_state=RANDOM_STATE), "100 relu units, defaults"),
    ]
    models = {}
    for name, est, params in specs:
        pipe = Pipeline(steps=[("preprocess", clone(preprocessor)), ("model", est)]).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)
        metrics = {"Accuracy": float(accuracy_score(y_test, pred)),
                   "F1_Weighted": float(f1_score(y_test, pred, average="weighted")),
                   "ROC_AUC_OvR": float(roc_auc_score(y_bin, proba, multi_class="ovr", average="weighted"))}
        report = classification_report(y_test, pred, labels=classes, output_dict=True, zero_division=0)
        models[name] = {
            "pipeline": pipe, "metrics": metrics, "params": params,
            "per_class": {c: {"precision": float(report[c]["precision"]), "recall": float(report[c]["recall"]),
                              "f1": float(report[c]["f1-score"]), "support": int(report[c]["support"])} for c in classes},
            "confusion": confusion_matrix(y_test, pred, labels=classes).astype(int).tolist(),
        }
        flag = "" if abs(metrics["F1_Weighted"] - NOTEBOOK_CLASSIFICATION[name]) < 0.0005 else "   <-- differs from the notebook"
        print(f"  {name:26s} acc={metrics['Accuracy']:.4f}  f1={metrics['F1_Weighted']:.4f}  auc={metrics['ROC_AUC_OvR']:.4f}{flag}")
    order = sorted(models, key=lambda n: -models[n]["metrics"]["F1_Weighted"])

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
        "models": models,
        "model_order": order,
        "best": order[0],
        "feature_columns": features,
        "key_features": key_features,
        "classes": classes,
        "feature_medians": {f: float(X_train[f].median()) for f in features},
        "class_medians": class_medians,
        "feature_quantiles": feature_quantiles,
        "scatter_sample": scatter_sample,
        "class_counts": {str(k): int(v) for k, v in y_train.value_counts().items()},
        "class_counts_all": {str(k): int(v) for k, v in y.value_counts().items()},
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    joblib.dump(bundle, MODELS / "classification_models.pkl", compress=3)
    return {n: models[n]["metrics"]["F1_Weighted"] for n in order}


#the clustering notebook's feature set and the k = 2 scores it reports
CLU_FEATURES = {
    'pl_orbper': 'Orbital period (days)', 'pl_trandurh': 'Transit duration (hours)', 'pl_trandep': 'Transit depth (ppm)',
    'pl_rade': 'Planet radius (Earth radii)', 'pl_insol': 'Insolation flux (Earth = 1)', 'pl_eqt': 'Equilibrium temperature (K)',
    'st_tmag': 'TESS magnitude', 'st_dist': 'Star distance (pc)', 'st_teff': 'Star temperature (K)',
    'st_logg': 'Star surface gravity (log g)', 'st_rad': 'Star radius (Solar radii)',
}
CLU_LOG_FEATURES = ['pl_orbper', 'pl_trandurh', 'pl_trandep', 'pl_rade', 'pl_insol', 'st_dist', 'st_rad']
NOTEBOOK_CLUSTERING = {
    "K-Means": {"Silhouette": 0.2752, "Davies-Bouldin": 1.5024, "Calinski-Harabasz": 2969.0},
    "Agglomerative": {"Silhouette": 0.1943, "Davies-Bouldin": 1.8368, "Calinski-Harabasz": 1777.4},
}
TOI_SNAPSHOT = "19 Oct 2026"
TOI_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+toi&format=csv"


def export_clustering():
    path = DATA / "tess_toi.csv"
    if not path.exists():
        import urllib.request
        urllib.request.urlretrieve(TOI_URL, path)
    df = pd.read_csv(path, low_memory=False)
    features = list(CLU_FEATURES)

    #same preprocessing as clustering.ipynb section b: log10, clip 1/99, median impute, standardise
    X_raw = df[features].copy()
    X_prep = X_raw.copy()
    for col in CLU_LOG_FEATURES:
        X_prep[col] = np.log10(X_prep[col].where(X_prep[col] > 0))
    clip_bounds = {}
    for col in features:
        lo, hi = X_prep[col].quantile([0.01, 0.99])
        X_prep[col] = X_prep[col].clip(lo, hi)
        clip_bounds[col] = (float(lo), float(hi))
    imputer = SimpleImputer(strategy="median").fit(X_prep)
    X_imputed = pd.DataFrame(imputer.transform(X_prep), columns=features, index=X_prep.index)
    scaler = StandardScaler().fit(X_imputed)
    X = pd.DataFrame(scaler.transform(X_imputed), columns=features, index=X_prep.index)

    #k-means for every k in the notebook's sweep, so the app can show any of them
    kmeans, sweep = {}, []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        kmeans[k] = km
        sweep.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(silhouette_score(X, km.labels_)),
                      "davies_bouldin": float(davies_bouldin_score(X, km.labels_)),
                      "calinski_harabasz": float(calinski_harabasz_score(X, km.labels_))})
    sweep = pd.DataFrame(sweep)
    best = sweep.loc[sweep["k"] == 2].iloc[0]
    reported = {"Silhouette": best["silhouette"], "Davies-Bouldin": best["davies_bouldin"], "Calinski-Harabasz": best["calinski_harabasz"]}
    for name, val in reported.items():
        ref = NOTEBOOK_CLUSTERING["K-Means"][name]
        flag = "" if abs(val - ref) < 0.001 * max(1, ref) else "   <-- differs from the notebook"
        print(f"  K-Means k=2 {name:18s} {val:.4f}{flag}")

    #agglomerative hierarchical clustering, section d of the notebook: complete linkage cut at k = 2, plus the linkage sweep behind that choice
    agg_labels = AgglomerativeClustering(n_clusters=2, linkage="complete", metric="euclidean").fit_predict(X)
    agg_metrics = {"Silhouette": float(silhouette_score(X, agg_labels)), "Davies-Bouldin": float(davies_bouldin_score(X, agg_labels)),
                   "Calinski-Harabasz": float(calinski_harabasz_score(X, agg_labels))}
    for name, val in agg_metrics.items():
        ref = NOTEBOOK_CLUSTERING["Agglomerative"][name]
        flag = "" if abs(val - ref) < 0.001 * max(1, ref) else "   <-- differs from the notebook"
        print(f"  Agglomerative complete k=2 {name:18s} {val:.4f}{flag}")
    Xn = X.to_numpy()
    agg_means = np.vstack([Xn[agg_labels == c].mean(axis=0) for c in range(2)])
    #name the two groups by their planet-radius mean, so the order cannot flip between runs
    agg_small = int(np.argmin(agg_means[:, features.index("pl_rade")]))
    agg_names = {agg_small: "Smaller planets on longer orbits around cool, nearby stars",
                 1 - agg_small: "Hot, large-planet signals around distant, larger stars"}
    agg_sweep = []
    for method in ("ward", "complete", "average"):
        tree = linkage(Xn, method=method, metric="euclidean")
        for k in range(2, 7):
            lab = cut_tree(tree, n_clusters=k).ravel()
            agg_sweep.append({"linkage": method, "k": k, "smallest_share": float(np.bincount(lab).min() / len(lab) * 100),
                              "silhouette": float(silhouette_score(X, lab)), "davies_bouldin": float(davies_bouldin_score(X, lab)),
                              "calinski_harabasz": float(calinski_harabasz_score(X, lab))})
    agg_sweep = pd.DataFrame(agg_sweep)

    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(X)
    group_map = {"CP": "planet", "KP": "planet", "PC": "candidate", "APC": "candidate", "FP": "not a planet", "FA": "not a planet"}
    #cluster 0 at k = 2 is the small-planet group in the notebook; name the clusters by their planet-radius centroid so the order cannot flip
    centers2 = kmeans[2].cluster_centers_[:, features.index("pl_rade")]
    small = int(np.argmin(centers2))
    names2 = {small: "Small planets around nearby, compact stars", 1 - small: "Jupiter-sized signals around distant, larger stars"}

    bundle = {
        "features": features, "feature_labels": CLU_FEATURES, "log_features": CLU_LOG_FEATURES,
        "clip_bounds": clip_bounds, "imputer": imputer, "scaler": scaler,
        "X_scaled": X.to_numpy(dtype=np.float32), "X_raw": X_raw.to_numpy(dtype=np.float32),
        "X_pca": pca.transform(X).astype(np.float32), "pca": pca,
        "explained_variance": [float(v) for v in pca.explained_variance_ratio_],
        "toi": df["toi"].astype(str).tolist(), "disposition": df["tfopwg_disp"].fillna("missing").tolist(),
        "disp_group": df["tfopwg_disp"].map(group_map).fillna("unknown").tolist(),
        "kmeans": kmeans, "sweep": sweep, "best_k": 2, "cluster_names": {2: names2},
        "feature_medians": {f: float(X_raw[f].median()) for f in features},
        "agglomerative": {"linkage": "complete", "k": 2, "labels": agg_labels.astype(np.int8), "metrics": agg_metrics,
                          "sizes": np.bincount(agg_labels).tolist(), "means": agg_means.astype(np.float32),
                          "cluster_names": agg_names, "sweep": agg_sweep},
        "n_rows": int(len(df)), "downloaded": TOI_SNAPSHOT,
    }
    joblib.dump(bundle, MODELS / "clustering_kmeans.pkl", compress=3)
    return {"K-Means": reported, "Agglomerative": agg_metrics}


if __name__ == "__main__":
    print("regression (test R2, ranked):")
    export_regression()
    print("classification (test weighted F1, ranked):")
    export_classification()
    print("clustering (K-Means, reported model k = 2):")
    export_clustering()
    for p in sorted(MODELS.glob("*.pkl")):
        print(f"saved {p.relative_to(ROOT)}  ({p.stat().st_size / 1e6:.1f} MB)")

# ML Capstone Project

Course: 23CSE301 Machine Learning

This repository contains the Machine Learning Capstone work for regression,
classification, and clustering tracks. The current stage focuses only on the
classification track.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
├── notebooks/
│   ├── regression.ipynb
│   ├── classification.ipynb
│   └── clustering.ipynb
├── models/
└── app/
```

## Classification Dataset

Dataset: NASA Kepler Objects of Interest (KOI)

Official source: https://data.nasa.gov/dataset/kepler-objects-of-interest-koi

Programmatic CSV source used by the notebook:
https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv

The NASA Open Data Portal describes KOI as a catalog of observed Kepler targets
flagged as possible exoplanet detections, while also noting that some entries
may be false positives caused by other transient detections.

Do not commit the raw CSV file to git. Place the downloaded file at:

```text
data/kepler_koi.csv
```

The classification notebook can also download the CSV automatically if internet
access is available.

## Current Classification Work

Implemented for Review 1:

- Gaussian Naive Bayes
- Decision Tree Classifier
- Decision Tree hyperparameter tuning with `GridSearchCV`
- Multiclass evaluation using weighted metrics and OvR ROC-AUC

Remaining Review 1 Part-A classifiers:

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Classifier

The regression track will be implemented separately. The clustering track is
reserved for a later review step.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Open `notebooks/classification.ipynb` and run the cells from top to bottom.

## AI Assistance Disclosure

Generative AI/Codex was used for code scaffolding and development assistance.
Dataset interpretation, analytical observations, feature-engineering
justification, and review/viva explanations must be completed and verified by
the student team.

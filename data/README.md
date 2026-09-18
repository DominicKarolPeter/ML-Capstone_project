# Data Directory

Raw datasets live here and are intentionally ignored by git so the repository
stays small. The notebooks and `app/export_models.py` expect the files at the
paths below.

## Regression: 200+ Financial Indicators of US Stocks (2018 slice)

Source: https://www.kaggle.com/datasets/cnic92/200-financial-indicators-of-us-stocks-20142018

Download `2018_Financial_Data.csv` from Kaggle and place it at:

```text
data/2018_Financial_Data.csv
```

Expected shape: 4,392 rows x 225 columns, with `Sector`, `2019 PRICE VAR [%]`
and `Class` as the last three columns.

## Classification: NASA Kepler Objects of Interest (KOI)

Official page: https://data.nasa.gov/dataset/kepler-objects-of-interest-koi

The classification notebook downloads the cumulative KOI table automatically
from the Exoplanet Archive when internet access is available, and saves it to:

```text
data/kepler_koi.csv
```

Programmatic source used by the notebook:

```text
https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv
```

Expected shape: about 9,564 rows x 153 columns (NASA updates the live table
over time, so the exact count can drift).

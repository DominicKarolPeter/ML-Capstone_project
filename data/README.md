# Data Directory

Place the NASA Kepler Objects of Interest CSV here as:

```text
data/kepler_koi.csv
```

Official NASA Open Data Portal page:
https://data.nasa.gov/dataset/kepler-objects-of-interest-koi

Programmatic CSV source used in the notebook:

```text
https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv
```

The raw CSV is intentionally ignored by git so the repository stays small. The
notebook will use this local file when present and can download it when internet
access is available.

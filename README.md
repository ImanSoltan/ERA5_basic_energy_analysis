# ERA5 PV Energy Analysis Project

## Project Overview
This project downloads, preprocesses, and analyzes ERA5 reanalysis data for solar PV energy analysis, focusing on the Bonn, Germany region for the first half of 2024.

## Directory Structure
```
ERA5_PV_Analysis/
├── data/
│   ├── raw/                # Raw downloaded ERA5 data (NetCDF)
│   ├── processed/          # Processed/merged/cleaned data
│   └── metadata/           # Metadata, e.g., coordinates
├── notebooks/              # Jupyter notebooks for EDA and analysis
├── scripts/
│   ├── download/           # Download scripts
│   ├── preprocess/         # Preprocessing scripts
│   ├── analysis/           # Analysis scripts (anomaly, events, clustering)
│   └── utils/              # Utility functions
├── results/                # Figures, reports, models
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── PROJECT_PLAN.md         # Project plan
```

## Setup
1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set up your `.cdsapirc` file in your home directory with your CDS API key.

## Data Download
Run the download script to fetch ERA5 data for Bonn:
```bash
python scripts/download/download_era5.py
```

## Data Preprocessing
Run the preprocessing script to merge, clean, and derive variables:
```bash
python scripts/preprocess/preprocess_data.py
```

## Exploratory Data Analysis (EDA)
Open and run the EDA notebook:
```
notebooks/02_Exploratory_Data_Analysis.ipynb
```
This notebook provides summary statistics, time series, and distribution plots for key variables.

## Core Analyses
- **Anomaly Detection:** `notebooks/03_Anomaly_Detection.ipynb`
- **Extreme Events:** `notebooks/04_Extreme_Events.ipynb`
- **Weather Pattern Clustering:** `notebooks/05_Weather_Clustering.ipynb`

Each notebook demonstrates how to use the corresponding helper functions in `scripts/analysis/`.

## Extending the Project
- Add new analysis scripts to `scripts/analysis/`.
- Add new notebooks for further exploration or prediction.

## Issues & Lessons Learned
- Ensure variable names match the CDS API exactly.
- Use the provided bounding box for Bonn: `[50.75, 7.0, 50.70, 7.15]`.
- Download and preprocessing scripts skip files that already exist.

## License
MIT 
# ERA5 Basic Energy Analysis

## Project Overview
This project provides a pipeline for downloading, preprocessing, and analyzing ERA5 reanalysis data for energy analysis. The pipeline focuses on the Bonn, Germany region but can be adapted for other locations. It handles the full workflow from data acquisition to generating analysis-ready datasets.

## Key Features
- **Automated Download**: Fetches ERA5 data for specified variables, years, and months
- **Efficient Processing**: Handles large datasets with memory-efficient operations
- **Time Filtering**: Keeps only relevant time steps (00:00, 06:00, 12:00, 18:00 UTC)
- **Site-Specific Analysis**: Extracts time series for specific locations
- **Reproducible**: Complete workflow from raw data to analysis

## Directory Structure
```
ERA5_basic_energy_analysis/
├── data/
│   ├── raw/                # Raw downloaded ERA5 data (GRIB/NetCDF)
│   └── processed/         # Processed and merged data
│       ├── era5_merged_*.nc      # Final merged dataset
│       └── site_timeseries_*.nc  # Extracted site time series
├── notebooks/              # Jupyter notebooks for analysis
│   └── 01_Data_Download_and_Preprocessing.ipynb
├── scripts/
│   ├── download/          # Data download utilities
│   │   └── download_era5.py
│   └── preprocess/        # Data processing scripts
│       ├── preprocess_data.py
│       └── merge_variables.py
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Getting Started

### Prerequisites
- Python 3.8+
- CDS API key (for downloading ERA5 data)
- Required Python packages (install via `pip install -r requirements.txt`)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ERA5_basic_energy_analysis.git
   cd ERA5_basic_energy_analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your CDS API key:
   - Create a file at `~/.cdsapirc` with your credentials:
     ```
     url: https://cds.climate.copernicus.eu/api/v2
     key: YOUR_UID:YOUR_API_KEY
     ```

## Usage

### 1. Download Data
Run the download script with custom parameters:
```bash
python scripts/download/download_era5.py --years 2023 --months 1 2 3 --variables 2m_temperature surface_solar_radiation_downwards
```

### 2. Preprocess and Merge Data
Process the downloaded data and merge variables:
```bash
python -m scripts.preprocess.preprocess_data --input-dir data/raw --output-dir data/processed
```

### 3. Explore Data
Use the Jupyter notebook for interactive analysis:
```bash
jupyter notebook notebooks/01_Data_Download_and_Preprocessing.ipynb
```

## Output Files
After processing, you'll find these files in `data/processed/`:
- `era5_merged_Bonn.nc`: Complete dataset with all variables
- `site_timeseries_Bonn.nc`: Time series data for the specified location

## Customization
- **Change Location**: Modify the `area` parameter in the download script
- **Add Variables**: Update the `variables` list in the download script
- **Adjust Time Range**: Modify the `years` and `months` parameters

## Troubleshooting
- **Missing Data**: Check CDS API for data availability
- **Processing Errors**: Ensure sufficient disk space and memory
- **Time Zone Issues**: All times are in UTC

## License
MIT

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.
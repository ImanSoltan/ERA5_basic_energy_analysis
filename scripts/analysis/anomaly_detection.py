import xarray as xr
import numpy as np
import pandas as pd
from scipy.stats import zscore
from statsmodels.tsa.seasonal import STL

def detect_zscore_anomalies(series, threshold=3.0):
    zscores = zscore(series.dropna())
    anomalies = np.abs(zscores) > threshold
    return anomalies

def detect_iqr_anomalies(series, factor=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return (series < lower) | (series > upper)

def detect_stl_anomalies(series, period=24, threshold=3.0):
    stl = STL(series.dropna(), period=period)
    res = stl.fit()
    resid = res.resid
    zscores = zscore(resid)
    anomalies = np.abs(zscores) > threshold
    return pd.Series(anomalies, index=series.dropna().index)

# Example usage in a notebook:
# ds = xr.open_dataset('data/processed/era5_processed_Bonn_2024_months_1-6.nc')
# df = ds.to_dataframe().reset_index()
# anomalies = detect_zscore_anomalies(df['surface_solar_radiation_downwards_w_m2']) 
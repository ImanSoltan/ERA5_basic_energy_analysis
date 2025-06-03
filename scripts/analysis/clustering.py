import xarray as xr
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def cluster_weather_patterns(df, features, n_clusters=4, random_state=42):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features].dropna())
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    return labels, centroids

# Example usage in a notebook:
# ds = xr.open_dataset('data/processed/era5_processed_Bonn_2024_months_1-6.nc')
# df = ds.to_dataframe().reset_index()
# features = ['surface_solar_radiation_downwards_w_m2', '2m_temperature_c', '10m_wind_speed', 'total_cloud_cover']
# labels, centroids = cluster_weather_patterns(df, features, n_clusters=4) 
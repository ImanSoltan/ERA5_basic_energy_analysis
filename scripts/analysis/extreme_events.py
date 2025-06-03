import xarray as xr
import pandas as pd
import numpy as np

def find_prolonged_low_irradiance(series, threshold, min_duration=3):
    below = series < threshold
    events = []
    start = None
    for i, val in enumerate(below):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= min_duration:
                events.append((start, i-1))
            start = None
    if start is not None and len(below) - start >= min_duration:
        events.append((start, len(below)-1))
    return events

def find_heatwaves(series, threshold, min_duration=3):
    above = series > threshold
    events = []
    start = None
    for i, val in enumerate(above):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= min_duration:
                events.append((start, i-1))
            start = None
    if start is not None and len(above) - start >= min_duration:
        events.append((start, len(above)-1))
    return events

def find_high_cloud_cover(series, threshold=0.8, min_duration=3):
    above = series > threshold
    events = []
    start = None
    for i, val in enumerate(above):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= min_duration:
                events.append((start, i-1))
            start = None
    if start is not None and len(above) - start >= min_duration:
        events.append((start, len(above)-1))
    return events

# Example usage in a notebook:
# ds = xr.open_dataset('data/processed/era5_processed_Bonn_2024_months_1-6.nc')
# df = ds.to_dataframe().reset_index()
# low_irr_events = find_prolonged_low_irradiance(df['surface_solar_radiation_downwards_w_m2'], threshold=50) 
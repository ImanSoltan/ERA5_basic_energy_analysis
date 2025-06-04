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

def detect_clear_sky_anomalies(ds, threshold=0.3, resample_freq=None):
    """
    Detect anomalies by comparing actual solar radiation to clear-sky model.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing 'ssrd' and 'ssrdc' variables
    threshold : float, default=0.3
        Threshold for anomaly detection (0-1), lower values detect more anomalies
    resample_freq : str, optional
        If provided, resample the data to this frequency (e.g., 'D' for daily, 'H' for hourly)
        This can make the plot more manageable for long time series
        
    Returns:
    --------
    tuple: (anomalies, figure)
        anomalies: xarray.DataArray of boolean values indicating anomalies
        figure: plotly figure object showing the anomalies
    """
    if 'ssrd' not in ds or 'ssrdc' not in ds:
        raise ValueError("Dataset must contain both 'ssrd' and 'ssrdc' variables")
    
    # Select first spatial point if multiple exist
    if 'latitude' in ds.dims and 'longitude' in ds.dims:
        ds_processed = ds.isel(latitude=0, longitude=0, drop=True)
    else:
        ds_processed = ds.copy()
    
    # Resample if requested
    if resample_freq is not None:
        ds_processed = ds_processed.resample(time=resample_freq).mean()
    
    # Calculate clear-sky index
    csi = ds_processed['ssrd'] / ds_processed['ssrdc']
    
    # Find anomalies (when actual is significantly less than clear-sky)
    anomalies = csi < (1 - threshold)
    
    # Create plot
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Add actual radiation
    fig.add_trace(go.Scatter(
        x=ds_processed.time.values,
        y=ds_processed['ssrd'].values,
        name='Actual Radiation',
        line=dict(color='blue', width=1),
        opacity=0.8
    ))
    
    # Add clear-sky radiation
    fig.add_trace(go.Scatter(
        x=ds_processed.time.values,
        y=ds_processed['ssrdc'].values,
        name='Clear-Sky Radiation',
        line=dict(color='green', dash='dash', width=1),
        opacity=0.8
    ))
    
    # Add anomalies
    anomaly_mask = anomalies.values
    if np.any(anomaly_mask):
        fig.add_trace(go.Scatter(
            x=ds_processed.time.values[anomaly_mask],
            y=ds_processed['ssrd'].values[anomaly_mask],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=6, opacity=0.8, line=dict(width=1, color='DarkRed'))
        ))
    else:
        print("No anomalies detected with the current threshold.")
    
    # Add range slider and buttons
    fig.update_layout(
        title=dict(
            text=f'Clear-Sky Radiation Anomalies (Threshold: {threshold})',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Time',
        yaxis_title='Radiation (W/m²)',
        showlegend=True,
        hovermode='x unified',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=6, label='6m', step='month', stepmode='backward'),
                    dict(count=1, label='YTD', step='year', stepmode='todate'),
                    dict(count=1, label='1y', step='year', stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(visible=True),
            type='date'
        ),
        height=600
    )
    
    # Add annotation with anomaly statistics
    anomaly_percent = np.mean(anomaly_mask) * 100
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref='paper',
        yref='paper',
        text=f'Anomalies: {np.sum(anomaly_mask):,} points ({anomaly_percent:.1f}%)',
        showarrow=False,
        bgcolor='white',
        bordercolor='black',
        borderwidth=1
    )
    
    return anomalies, fig

# Example usage in a notebook:
# ds = xr.open_dataset('data/processed/era5_processed_Bonn_2024_months_1-6.nc')
# anomalies, fig = detect_clear_sky_anomalies(ds, threshold=0.3)
# fig.show()
# anomalies = detect_zscore_anomalies(df['surface_solar_radiation_downwards_w_m2']) 
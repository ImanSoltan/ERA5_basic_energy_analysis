import xarray as xr
import numpy as np
import pandas as pd
from scipy.stats import zscore
from statsmodels.tsa.seasonal import STL
import plotly.graph_objects as go

def detect_zscore_anomalies(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Detect anomalies in a time series using the Z-score method.

    Anomalies are identified where the Z-score exceeds the specified threshold.
    NaN values are dropped before calculating Z-scores.

    Parameters:
    -----------
    series : pd.Series
        Input time series.
    threshold : float, default=3.0
        Z-score threshold for identifying anomalies.

    Returns:
    --------
    pd.Series
        Boolean series with the same index as the input `series`,
        where True indicates an anomaly. Original NaN positions are False.
    """
    series_dropped = series.dropna()
    if series_dropped.empty:
        return pd.Series(False, index=series.index, dtype=bool)
    
    zscores_values = zscore(series_dropped.values)
    anomalies_values = np.abs(zscores_values) > threshold
    return pd.Series(anomalies_values, index=series_dropped.index).reindex(series.index, fill_value=False)

def detect_iqr_anomalies(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """
    Detect anomalies in a time series using the Interquartile Range (IQR) method.

    Anomalies are identified as points falling below Q1 - factor*IQR or
    above Q3 + factor*IQR.

    Parameters:
    -----------
    series : pd.Series
        Input time series.
    factor : float, default=1.5
        Factor to multiply the IQR by for determining outlier bounds.

    Returns:
    --------
    pd.Series
        Boolean series with the same index as the input `series`,
        where True indicates an anomaly. Original NaN positions are False.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or pd.isna(q1) or pd.isna(q3):
        # Handle cases where IQR cannot be computed (e.g., too many NaNs)
        return pd.Series(False, index=series.index, dtype=bool)

    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    anomalies = (series < lower_bound) | (series > upper_bound)
    return anomalies.fillna(False) # Ensure NaNs from original series become False

def detect_stl_anomalies(series: pd.Series, period: int = 24, threshold: float = 3.0) -> pd.Series:
    """
    Detect anomalies in a time series using STL decomposition.

    Anomalies are identified based on the z-scores of the residual component.

    Parameters:
    -----------
    series : pd.Series
        Input time series with a DatetimeIndex.
    period : int, default=24
        Length of the seasonal decomposition period.
        For hourly data, a period of 24 is typically used to capture daily seasonality.
        For daily data with weekly seasonality, a period of 7 might be used.
    threshold : float, default=3.0
        Z-score threshold for identifying anomalies in the residuals.

    Returns:
    --------
    pd.Series
        Boolean series with the same index as the input `series`,
        where True indicates an anomaly. Original NaN positions are False.
    """
    series_dropped = series.dropna()
    if series_dropped.empty or len(series_dropped) <= 2 * period:
        # STL requires the series length to be greater than 2 * period.
        # Also handle empty series after dropna.
        # print(f"Warning: Series length ({len(series_dropped)}) is not sufficient for STL decomposition "
        #       f"with period={period}. Returning no anomalies.")
        return pd.Series(False, index=series.index, dtype=bool)

    stl = STL(series_dropped, period=period, robust=True)
    res = stl.fit()
    resid = res.resid
    
    # Z-score calculation can fail if resid is constant (e.g., all zeros)
    if resid.std() == 0:
        # print(f"Warning: Residuals standard deviation is zero. Cannot compute Z-scores.")
        # If residuals are all zero (perfect fit or no variability), no anomalies by this method.
        anomalies_values = np.zeros(len(resid), dtype=bool)
    else:
        zscores_values = zscore(resid)
        anomalies_values = np.abs(zscores_values) > threshold
        
    return pd.Series(anomalies_values, index=series_dropped.index).reindex(series.index, fill_value=False)

def detect_clear_sky_anomalies(ds: xr.Dataset, threshold: float = 0.3, resample_freq: str = None, conversion_factor_to_wm2: float = 3600.0) -> tuple[xr.DataArray, go.Figure]:
    """
    Detect anomalies by comparing actual solar radiation to clear-sky model.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing 'ssrd' (surface solar radiation) and 'ssrdc' (clear-sky solar radiation) variables.
    threshold : float, default=0.3
        Threshold for anomaly detection (0-1). Anomalies occur when actual radiation is less than
        (1 - threshold) * clear-sky radiation. Lower values detect more anomalies.
    resample_freq : str, optional
        If provided, resample the data to this frequency (e.g., 'D' for daily, 'H' for hourly)
        before calculating anomalies and plotting. This can make the plot more manageable.
    conversion_factor_to_wm2 : float, default=3600.0
        Factor to convert radiation variables (ssrd, ssrdc) to W/m² for plotting.
        If variables are in J/m² accumulated per hour, use 3600.0.
        If variables are already in W/m², set to 1.0 or None.
        The Clear Sky Index calculation itself is independent of this factor if ssrd and ssrdc have consistent units.

    Returns:
    --------
    tuple[xr.DataArray, go.Figure]
        anomalies: xarray.DataArray of boolean values indicating anomalies, aligned with ds_processed time.
        figure: plotly.graph_objects.Figure object showing the anomalies.
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
    
    # Calculate clear-sky index (ratio is unit-independent if ssrd and ssrdc are consistent)
    csi = ds_processed['ssrd'] / ds_processed['ssrdc']
    
    # Find anomalies (when actual is significantly less than clear-sky)
    # Ensure csi is not NaN or inf before comparison to avoid warnings/errors with boolean indexing
    valid_csi = csi.where(np.isfinite(csi), other=np.nan)
    anomalies = valid_csi < (1 - threshold)

    # Prepare data for plotting (convert to W/m² if factor is provided)
    ssrd_plot = ds_processed['ssrd'].values
    ssrdc_plot = ds_processed['ssrdc'].values
    if conversion_factor_to_wm2 is not None and conversion_factor_to_wm2 != 0:
        ssrd_plot = ssrd_plot / conversion_factor_to_wm2
        ssrdc_plot = ssrdc_plot / conversion_factor_to_wm2
    
    # Create plot
    fig = go.Figure()
    
    # Add actual radiation
    fig.add_trace(go.Scatter(
        x=ds_processed.time.values,
        y=ssrd_plot,
        name='Actual Radiation',
        line=dict(color='blue', width=1),
        opacity=0.8
    ))
    
    # Add clear-sky radiation
    fig.add_trace(go.Scatter(
        x=ds_processed.time.values,
        y=ssrdc_plot,
        name='Clear-Sky Radiation',
        line=dict(color='green', dash='dash', width=1),
        opacity=0.8
    ))
    
    # Add anomalies
    anomaly_mask = anomalies.values
    if np.any(anomaly_mask):
        fig.add_trace(go.Scatter(
            x=ds_processed.time.values[anomaly_mask],
            y=ssrd_plot[anomaly_mask],
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
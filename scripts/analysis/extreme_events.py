import xarray as xr
import pandas as pd
import numpy as np

def find_prolonged_low_irradiance(series, threshold=80.0, min_duration=4, time_index=None, 
                               day_start_hour=6, day_end_hour=18):
    """
    Identify periods of prolonged low solar irradiance that could impact PV system performance.
    
    Parameters:
    -----------
    series : pandas.Series or xarray.DataArray
        Time series of surface solar radiation downwards (W/m²)
    threshold : float, default=80.0
        Irradiance threshold below which conditions are considered low (W/m²)
    min_duration : int, default=4
        Minimum number of consecutive hours to be considered significant
    time_index : pandas.DatetimeIndex, optional
        If provided, used to filter daytime hours
    day_start_hour : int, default=6
        Start of production day (0-23)
    day_end_hour : int, default=18
        End of production day (0-23)
        
    Returns:
    --------
    list of tuples
        Each tuple contains (start_index, end_index) of low irradiance events
        during production hours
    
    Example:
    --------
    >>> events = find_prolonged_low_irradiance(
    ...     df['ssrd'], 
    ...     threshold=80,
    ...     min_duration=4,
    ...     time_index=df.index,
    ...     day_start_hour=6,
    ...     day_end_hour=18
    ... )
    """
    # Create a mask for low irradiance
    low_irradiance = series < threshold
    
    # If time_index is provided, only consider daytime hours
    if time_index is not None:
        # Handle different types of time indices
        if hasattr(time_index, 'hour'):  # Already has hour attribute (DatetimeIndex)
            hours = pd.Series(time_index.hour, index=time_index)
        elif hasattr(time_index, 'dt') and hasattr(time_index.dt, 'hour'):  # Series with dt accessor
            hours = time_index.dt.hour
        elif hasattr(time_index, 'time'):  # xarray time coordinate
            hours = pd.to_datetime(time_index.values).hour
        else:
            raise ValueError("time_index must be a DatetimeIndex, a Series with datetime values, "
                         "or an xarray time coordinate")
        
        # Create boolean mask for daytime hours
        if isinstance(hours, (pd.Series, pd.Index)):
            hours = hours.values  # Convert to numpy array for boolean operations
        
        # Ensure the length matches the series
        if len(hours) != len(series):
            hours = np.tile(hours, len(series) // len(hours) + 1)[:len(series)]
            
        daytime = (hours >= day_start_hour) & (hours < day_end_hour)
        low_irradiance = low_irradiance & (daytime if isinstance(daytime, (np.ndarray, pd.Series)) else False)
    
    # Find events
    events = []
    start = None
    
    for i, val in enumerate(low_irradiance):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= min_duration:
                events.append((start, i-1))
            start = None
    
    if start is not None and len(low_irradiance) - start >= min_duration:
        events.append((start, len(low_irradiance)-1))
    
    return events

def find_low_irradiance_6hourly(series, time_index, threshold=30.0, min_consecutive=1):
    """
    Find low irradiance events in 6-hourly ERA5 data.
    
    Parameters:
    -----------
    series : pandas.Series or array-like
        Time series of solar radiation values (W/m²)
    time_index : pandas.DatetimeIndex or Series
        Corresponding timestamps
    threshold : float, default=30.0
        Irradiance threshold (W/m²)
    min_consecutive : int, default=1
        Minimum number of consecutive daytime points below threshold
        (1 = single 6h period, 2 = 12h period, etc.)
        
    Returns:
    --------
    list of tuples
        Each tuple contains (start_index, end_index) of low irradiance events
    
    Example:
    --------
    >>> events = find_low_irradiance_6hourly(
    ...     df['ssrd_wm2'],
    ...     time_index=df['time'],
    ...     threshold=30,
    ...     min_consecutive=1
    ... )
    """
    # Convert to pandas Series if not already
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
        
    # Get hour of day
    if hasattr(time_index, 'hour'):
        hours = time_index.hour
    elif hasattr(time_index, 'dt'):
        hours = time_index.dt.hour
    else:
        hours = pd.Series(time_index).dt.hour
    
    # Consider 6:00-18:00 as "daytime"
    daytime = (hours >= 6) & (hours < 18)
    
    # Find low values during daytime
    low = (series < threshold) & daytime
    
    # Find consecutive low values
    events = []
    start = None
    
    for i, val in enumerate(low):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if (i - start) >= min_consecutive:
                events.append((start, i-1))
            start = None
    
    if start is not None and (len(low) - start) >= min_consecutive:
        events.append((start, len(low)-1))
    
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

def find_wind_lulls(wind_speed_series, threshold=2.0, min_duration=6):
    """
    Identify periods of low wind speeds (lulls) that could impact wind energy production.
    
    Parameters:
    -----------
    wind_speed_series : pandas.Series or xarray.DataArray
        Time series of wind speed values (m/s)
    threshold : float, default=2.0
        Wind speed threshold below which conditions are considered a lull (m/s)
    min_duration : int, default=6
        Minimum number of consecutive hours to be considered a significant lull
        
    Returns:
    --------
    list of tuples
        Each tuple contains (start_index, end_index) of wind lull events
    """
    below = wind_speed_series < threshold
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

def find_wind_storms(wind_speed_series, threshold=25.0, min_duration=3):
    """
    Identify periods of high wind speeds (storms) that could impact wind turbine operations.
    
    Parameters:
    -----------
    wind_speed_series : pandas.Series or xarray.DataArray
        Time series of wind speed values (m/s)
    threshold : float, default=25.0
        Wind speed threshold above which conditions are considered a storm (m/s)
    min_duration : int, default=3
        Minimum number of consecutive hours to be considered a significant storm
        
    Returns:
    --------
    list of tuples
        Each tuple contains (start_index, end_index) of wind storm events
    """
    above = wind_speed_series > threshold
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

def plot_wind_events(wind_speed_series, time_series, lull_events=None, storm_events=None, 
                    lull_threshold=2.0, storm_threshold=25.0, title='Wind Speed Events'):
    """
    Create an interactive plot of wind speed with lull and storm events highlighted.
    
    Parameters:
    -----------
    wind_speed_series : array-like
        Wind speed values (m/s)
    time_series : array-like
        Corresponding timestamps
    lull_events : list of tuples, optional
        Output from find_wind_lulls()
    storm_events : list of tuples, optional
        Output from find_wind_storms()
    lull_threshold : float, default=2.0
        Wind speed threshold for lulls (m/s)
    storm_threshold : float, default=25.0
        Wind speed threshold for storms (m/s)
    title : str, optional
        Plot title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plot of wind speed with events highlighted
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(rows=1, cols=1)
    
    # Add wind speed trace
    fig.add_trace(go.Scatter(
        x=time_series,
        y=wind_speed_series,
        name='Wind Speed (m/s)',
        line=dict(color='blue', width=1),
        opacity=0.8
    ))
    
    # Add threshold lines
    fig.add_hline(y=lull_threshold, line_dash='dash', 
                 line_color='orange', opacity=0.7,
                 annotation_text=f'Lull Threshold ({lull_threshold} m/s)', 
                 annotation_position='bottom right')
    
    fig.add_hline(y=storm_threshold, line_dash='dash', 
                 line_color='red', opacity=0.7,
                 annotation_text=f'Storm Threshold ({storm_threshold} m/s)',
                 annotation_position='top right')
    
    # Add lull events
    if lull_events:
        for start, end in lull_events:
            if start < len(time_series) and end < len(time_series):
                fig.add_vrect(
                    x0=time_series[start], 
                    x1=time_series[end],
                    fillcolor='orange', 
                    opacity=0.2, 
                    line_width=0,
                    annotation_text='Lull',
                    annotation_position='top left'
                )
    
    # Add storm events
    if storm_events:
        for start, end in storm_events:
            if start < len(time_series) and end < len(time_series):
                fig.add_vrect(
                    x0=time_series[start], 
                    x1=time_series[end],
                    fillcolor='red', 
                    opacity=0.2, 
                    line_width=0,
                    annotation_text='Storm',
                    annotation_position='top right'
                )
    
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Wind Speed (m/s)',
        showlegend=True,
        hovermode='x',
        height=500,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=6, label='6m', step='month', stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(visible=True),
            type='date'
        )
    )
    
    return fig

# Example usage in a notebook:
# ds = xr.open_dataset('data/processed/era5_processed_Bonn_2024_months_1-6.nc')
# df = ds.to_dataframe().reset_index()
# 
# # Detect wind lulls and storms
# lull_events = find_wind_lulls(df['wind_speed'], threshold=2.0, min_duration=6)
# storm_events = find_wind_storms(df['wind_speed'], threshold=25.0, min_duration=3)
# 
# # Create interactive plot
# fig = plot_wind_events(
#     wind_speed_series=df['wind_speed'],
#     time_series=df['time'],
#     lull_events=lull_events,
#     storm_events=storm_events,
#     title='Wind Speed Extremes Analysis'
# )
# fig.show()
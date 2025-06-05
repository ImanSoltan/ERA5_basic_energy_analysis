"""
PV Performance Analysis Module

This module provides functionality to analyze and compare PV performance
using ERA5 data and PVGIS reference data.
"""

import pandas as pd
import numpy as np
import requests
from typing import Dict, Optional, Tuple, Union, List
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import xarray as xr
from datetime import datetime

class PVGISClient:
    """Client for interacting with the PVGIS API."""
    
    BASE_URL = "https://re.jrc.ec.europa.eu/api/seriescalc"
    
    @classmethod
    def test_connection(cls, lat: float, lon: float, year: int = None) -> bool:
        """
        Test connection to PVGIS API with a simple request.
        
        Args:
            lat: Latitude of the location
            lon: Longitude of the location
            year: Year to test (default: current year)
            
        Returns:
            bool: True if connection is successful, False otherwise
        """
        if year is None:
            year = datetime.now().year
            
        try:
            test_params = {
                'lat': lat,
                'lon': lon,
                'startyear': year,
                'endyear': year,
                'peakpower': 1.0,
                'loss': 14.0,
                'outputformat': 'json'
            }
            
            response = requests.get(
                cls.BASE_URL, 
                params=test_params, 
                timeout=30
            )
            return response.status_code == 200
            
        except Exception as e:
            print(f"PVGIS connection test failed: {e}")
            return False
    
    @classmethod
    def get_pvgis_data(
        cls,
        lat: float,
        lon: float,
        start_year: int,
        end_year: int,
        peak_power: float = 1.0,
        system_loss: float = 14.0,
        mounting: str = 'free',
        angle: float = 35.0,
        aspect: float = 0.0,
        pv_tech: str = 'crystSi',
        use_sample_data: bool = False
    ) -> pd.DataFrame:
        """
        Fetch PV generation data from PVGIS API for multiple years.
        
        Args:
            lat: Latitude of the location
            lon: Longitude of the location
            start_year: Start year (inclusive)
            end_year: End year (inclusive)
            peak_power: Peak power of the PV system in kW
            system_loss: Total system losses in percent
            mounting: Mounting type ('free' or 'building')
            angle: Tilt angle of the PV modules in degrees
            aspect: Azimuth/orientation (0 = South, 90 = West, -90 = East)
            pv_tech: PV technology type
            use_sample_data: If True, returns sample data instead of making API calls
            
        Returns:
            DataFrame containing PVGIS data with datetime index
        """
        if use_sample_data:
            return cls._get_sample_data(start_year, end_year)
            
        all_data = []
        
        for year in range(start_year, end_year + 1):
            try:
                params = {
                    'lat': lat,
                    'lon': lon,
                    'startyear': year,
                    'endyear': year,
                    'peakpower': peak_power,
                    'loss': system_loss,
                    'angle': angle,
                    'aspect': aspect,
                    'outputformat': 'json',
                    'pvtechchoice': pv_tech,
                    'mountingplace': mounting,
                    'components': 1
                }
                
                print(f"Fetching PVGIS data for {year}...")
                response = requests.get(
                    cls.BASE_URL, 
                    params=params, 
                    timeout=60
                )
                response.raise_for_status()
                data = response.json()
                
                # Convert to DataFrame
                df = pd.DataFrame(data['outputs']['hourly'])
                df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')
                df = df.set_index('time')
                all_data.append(df)
                
            except requests.exceptions.RequestException as e:
                print(f"Network error fetching PVGIS data for {year}: {e}")
                continue
            except Exception as e:
                print(f"Error processing PVGIS data for {year}: {e}")
                continue
        
        if not all_data:
            print("No PVGIS data was retrieved. Using sample data instead.")
            return cls._get_sample_data(start_year, end_year)
            
        return pd.concat(all_data).sort_index()
    
    @staticmethod
    def _get_sample_data(start_year: int, end_year: int) -> pd.DataFrame:
        """Generate sample PVGIS data for demonstration purposes."""
        print("Generating sample PVGIS data...")
        date_range = pd.date_range(
            start=f"{start_year}-01-01", 
            end=f"{end_year}-12-31 23:00", 
            freq='H'
        )
        # Create realistic diurnal pattern
        hours = np.array([t.hour for t in date_range])
        daily_pattern = np.sin((hours - 6) * np.pi / 12)  # Peak at noon
        daily_pattern[daily_pattern < 0] = 0  # Set night to 0
        
        # Add some seasonal variation
        days_since_start = (date_range - date_range[0]).days
        seasonal_variation = 1 + 0.3 * np.sin(days_since_start * 2 * np.pi / 365)
        
        # Generate power values
        power = 800 * daily_pattern * seasonal_variation * np.random.lognormal(0, 0.1, len(date_range))
        
        df = pd.DataFrame({
            'P': power,
            'G(i)': power * 1.2,  # Rough estimate of irradiance
            'H_sun': np.where(power > 0, 1, 0),  # Sun above horizon
            'T2m': 15 + 10 * np.sin(days_since_start * 2 * np.pi / 365) + np.random.normal(0, 3, len(date_range)),
            'WS10m': 3 + np.random.weibull(2, len(date_range)) * 2
        }, index=date_range)
        
        # Add capacity factor
        df['capacity_factor'] = df['P'] / 1000  # Assuming 1 kWp system
        
        return df

class ERA5Processor:
    """Class for processing ERA5 data for PV performance analysis."""
    
    @staticmethod
    def process_era5_data(
        era5_ds: xr.Dataset,
        lat: float,
        lon: float,
        ssrd_var: str = 'ssrd',
        system_efficiency: float = 0.15,  # Typical PV system efficiency
        tracking: str = 'fixed'  # 'fixed' or 'single_axis'
    ) -> pd.DataFrame:
        """
        Process ERA5 dataset for PV performance analysis with improved capacity factor calculation.
        
        Args:
            era5_ds: xarray Dataset containing ERA5 data
            lat: Target latitude (in degrees, positive for North)
            lon: Target longitude (in degrees, positive for East)
            ssrd_var: Name of the surface solar radiation variable
            system_efficiency: PV system efficiency (0-1)
            tracking: Tracking system type ('fixed' or 'single_axis')
            
        Returns:
            DataFrame with processed ERA5 data
        """
        # Select nearest point
        era5_point = era5_ds.sel(
            latitude=lat,
            longitude=lon,
            method='nearest'
        )
        
        # Convert to DataFrame
        df = era5_point.to_dataframe().reset_index()
        
        if ssrd_var not in df.columns:
            raise ValueError(f"Column {ssrd_var} not found in ERA5 data")
        
        # Convert J/m² to W/m² (6-hourly data)
        df['ssrd_wm2'] = df[ssrd_var] / (6 * 3600)  # 6 hours in seconds
        
        # Calculate solar geometry
        df['doy'] = df['time'].dt.dayofyear
        df['hour'] = df['time'].dt.hour + df['time'].dt.minute/60
        
        # Solar declination in radians
        df['declination'] = np.radians(23.45) * np.sin(2 * np.pi * (284 + df['doy']) / 365)
        
        # Hour angle in radians (converting from hours to degrees to radians)
        df['hour_angle'] = np.radians(15 * (df['hour'] - 12))
        
        # Solar zenith angle
        lat_rad = np.radians(lat)
        cos_zenith = (
            np.sin(lat_rad) * np.sin(df['declination']) +
            np.cos(lat_rad) * np.cos(df['declination']) * np.cos(df['hour_angle'])
        )
        df['zenith'] = np.arccos(cos_zenith.clip(-1, 1))
        
        # Extraterrestrial irradiance (W/m²)
        df['extraterrestrial'] = 1367 * (1 + 0.033 * np.cos(2 * np.pi * df['doy'] / 365))
        
        # Clear-sky radiation (simplified)
        df['clear_sky'] = df['extraterrestrial'] * 0.7 ** (1 / np.cos(df['zenith']) ** 0.678)
        
        # Clear-sky index (ratio of actual to clear-sky radiation)
        df['kt'] = (df['ssrd_wm2'] / df['clear_sky']).clip(0, 1.2)
        
        # Filter out nighttime (sun below horizon)
        df = df[df['zenith'] < np.radians(90)]
        
        # For fixed systems, use GHI as POA approximation
        df['poa'] = df['ssrd_wm2']
        
        # Calculate capacity factor
        df['capacity_factor'] = (df['poa'] * system_efficiency / 1000).clip(0, 1)
        
        # Clean up temporary columns
        keep_cols = ['time', 'latitude', 'longitude', 'ssrd_wm2', 'capacity_factor']
        df = df[keep_cols + [c for c in df.columns if c not in keep_cols]]
        
        return df

class PVPerformanceAnalyzer:
    """
    A class to analyze and compare PV performance between ERA5 data and PVGIS.
    """
    
    def __init__(self, lat: float, lon: float):
        """
        Initialize the PV Performance Analyzer.
        
        Args:
            lat: Latitude of the location
            lon: Longitude of the location
        """
        self.lat = lat
        self.lon = lon
        self.pvgis_client = PVGISClient()
        self.era5_processor = ERA5Processor()
    
    def get_pvgis_data(self, **kwargs) -> pd.DataFrame:
        """Get PVGIS data for the configured location."""
        return self.pvgis_client.get_pvgis_data(
            lat=self.lat,
            lon=self.lon,
            **kwargs
        )
    
    def test_pvgis_connection(self, year: int = None) -> bool:
        """Test connection to PVGIS API."""
        return PVGISClient.test_connection(self.lat, self.lon, year)
    
    def process_era5_data(self, era5_ds: xr.Dataset, **kwargs) -> pd.DataFrame:
        """Process ERA5 data for the configured location."""
        return self.era5_processor.process_era5_data(
            era5_ds=era5_ds,
            lat=self.lat,
            lon=self.lon,
            **kwargs
        )
    
    @staticmethod
    def compare_datasets(
        era5_data: pd.DataFrame,
        pvgis_data: pd.DataFrame,
        era5_cf_col: str = 'capacity_factor',
        pvgis_cf_col: str = 'P'
    ) -> pd.DataFrame:
        """
        Compare ERA5 and PVGIS data.
        
        Args:
            era5_data: DataFrame with ERA5 data
            pvgis_data: DataFrame with PVGIS data
            era5_cf_col: Column name for capacity factor in ERA5 data
            pvgis_cf_col: Column name for capacity factor in PVGIS data
            
        Returns:
            DataFrame with comparison results
        """
        # Ensure timezone-naive indices for comparison
        era5_data.index = era5_data.index.tz_localize(None) if era5_data.index.tz is not None else era5_data.index
        pvgis_data.index = pvgis_data.index.tz_localize(None) if pvgis_data.index.tz is not None else pvgis_data.index
        
        # Resample ERA5 to hourly to match PVGIS
        era5_hourly = era5_data[era5_cf_col].resample('H').interpolate()
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'era5_cf': era5_hourly,
            'pvgis_cf': pvgis_data[pvgis_cf_col]
        }).dropna()
        
        # Calculate performance ratio
        comparison['performance_ratio'] = comparison['era5_cf'] / comparison['pvgis_cf']
        
        return comparison
    
    @staticmethod
    def plot_comparison(comparison: pd.DataFrame, title: str = None) -> go.Figure:
        """
        Plot comparison between ERA5 and PVGIS data.
        
        Args:
            comparison: DataFrame from compare_datasets()
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add capacity factor traces
        fig.add_trace(
            go.Scatter(
                x=comparison.index,
                y=comparison['era5_cf'],
                name='ERA5 Capacity Factor',
                line=dict(color='blue')
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=comparison.index,
                y=comparison['pvgis_cf'],
                name='PVGIS Capacity Factor',
                line=dict(color='red', dash='dash')
            ),
            secondary_y=False
        )
        
        # Add performance ratio
        fig.add_trace(
            go.Scatter(
                x=comparison.index,
                y=comparison['performance_ratio'],
                name='Performance Ratio (ERA5/PVGIS)',
                line=dict(color='green')
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title=title or 'ERA5 vs PVGIS Comparison',
            xaxis_title='Date',
            yaxis_title='Capacity Factor',
            yaxis2=dict(
                title='Performance Ratio',
                overlaying='y',
                side='right',
                range=[0, 2]  # Typical range for performance ratio
            ),
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
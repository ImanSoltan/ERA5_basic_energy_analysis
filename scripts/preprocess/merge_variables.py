import os
import glob
import xarray as xr
import pandas as pd
import numpy as np

def get_time_coord_name(ds):
    """
    Determine which coordinate represents time in the dataset.
    Returns the name of the time coordinate.
    """
    if 'time' in ds.coords:
        return 'time'
    elif 'valid_time' in ds.coords:
        return 'valid_time'
    else:
        raise ValueError("No time coordinate found in dataset")

def ensure_unique_times(times, values):
    """
    Ensure time values are unique by averaging values at duplicate timestamps.
    
    Parameters:
    -----------
    times : np.ndarray
        Array of time values
    values : np.ndarray
        Array of data values corresponding to times
        
    Returns:
    --------
    unique_times : np.ndarray
        Array of unique time values
    unique_values : np.ndarray
        Array of averaged values for each unique time
    """
    # Convert times to pandas datetime if they aren't already
    times = pd.to_datetime(times)
    
    # Create a DataFrame with times as index
    # Handle multidimensional arrays by keeping spatial dimensions intact
    if len(values.shape) > 2:
        # For 3D arrays (time, lat, lon), reshape preserving spatial structure
        n_times = len(times)
        spatial_shape = values.shape[1:]  # (lat, lon)
        values_2d = values.reshape(n_times, -1)  # Flatten spatial dimensions
        
        # Create DataFrame with each spatial point as a column
        df = pd.DataFrame(values_2d, index=times)
        
        # Group by time and average
        df_unique = df.groupby(level=0).mean()
        
        # Get unique times and reshape values back to 3D
        unique_times = df_unique.index.values
        unique_values = df_unique.values.reshape(-1, *spatial_shape)
        
    else:
        # For 1D or 2D arrays, keep original behavior
        df = pd.DataFrame({'values': values.reshape(len(times), -1)}, index=times)
        df_unique = df.groupby(level=0).mean()
        unique_times = df_unique.index.values
        unique_values = df_unique.values.reshape(df_unique.shape)
    
    return unique_times, unique_values

def standardize_time_coords(ds, var_name):
    """
    Standardize time coordinates for a dataset.
    If the dataset has both time and step dimensions, combine them into a single time coordinate.
    Ensures unique time values by averaging duplicates.
    """
    if var_name not in ds.data_vars:
        print(f"Available variables: {list(ds.data_vars.keys())}")
        raise ValueError(f"Variable {var_name} not found in dataset")
    
    # First, identify the time coordinate
    time_coord = get_time_coord_name(ds)
    
    if 'step' in ds.dims and time_coord in ds.dims:
        # This is an accumulated/mean variable with time and step dimensions
        # Get the time values
        if hasattr(ds, 'valid_time'):
            time_values = ds.valid_time.values
        else:
            time_values = ds[time_coord].values
            
        # Convert to datetime if needed
        if not np.issubdtype(time_values.dtype, np.datetime64):
            time_values = pd.to_datetime(time_values)
        
        # Reshape the data to combine time and step dimensions
        data_vals = ds[var_name].values
        if len(time_values.shape) > 1:
            time_values = time_values.ravel()
            data_vals = data_vals.reshape(-1, *data_vals.shape[-2:])
        
        # Ensure unique times
        unique_times, unique_values = ensure_unique_times(time_values, data_vals)
        print(f"Reduced {len(time_values)} timestamps to {len(unique_times)} unique timestamps")
        
        # Create new dataset with single time dimension
        new_ds = xr.Dataset(
            {var_name: (('time', 'latitude', 'longitude'), unique_values)},
            coords={
                'time': unique_times,
                'latitude': ds.latitude,
                'longitude': ds.longitude
            }
        )
        return new_ds
    else:
        # For variables with just time dimension
        if time_coord != 'time':
            ds = ds.rename({time_coord: 'time'})
        
        # Ensure unique times for single time dimension
        data_vals = ds[var_name].values
        unique_times, unique_values = ensure_unique_times(
            ds.time.values,
            data_vals
        )
        
        if len(unique_times) < len(ds.time):
            print(f"Reduced {len(ds.time)} timestamps to {len(unique_times)} unique timestamps")
            return xr.Dataset(
                {var_name: (('time', 'latitude', 'longitude'), unique_values)},
                coords={
                    'time': unique_times,
                    'latitude': ds.latitude,
                    'longitude': ds.longitude
                }
            )
        
        return ds

def merge_variable_files(processed_dir, output_file=None, area_name='Bonn'):
    """
    Merge individual variable NetCDF files into a single dataset.
    Handles different time dimensions and step coordinates.
    Only keeps data at 00:00, 06:00, 12:00, and 18:00 UTC.
    
    Parameters:
    -----------
    processed_dir : str
        Directory containing the processed NetCDF files
    output_file : str, optional
        Path to save the merged dataset. If None, will create a default name
    area_name : str
        Name of the area (used in default output filename)
        
    Returns:
    --------
    xarray.Dataset
        Merged dataset containing all variables at 00, 06, 12, 18 UTC
    """
    if output_file is None:
        output_file = os.path.join(processed_dir, f'era5_merged_{area_name}.nc')
    
    print(f"Looking for NetCDF files in {processed_dir}")
    nc_files = glob.glob(os.path.join(processed_dir, "*.nc"))
    
    if not nc_files:
        raise ValueError(f"No NetCDF files found in {processed_dir}")
    
    print(f"Found {len(nc_files)} NetCDF files")
    
    # First, load and standardize all datasets
    standardized_datasets = {}
    for file in nc_files:
        try:
            var_name = os.path.basename(file).split('_')[0]  # Extract variable name from filename
            print(f"\nProcessing {var_name}...")
            
            # Open dataset with decode_timedelta=True to handle warnings
            ds = xr.open_dataset(file, decode_timedelta=True)
            
            # Print original dataset info
            print(f"Original coordinates: {list(ds.coords.keys())}")
            print(f"Original dimensions: {dict(ds.sizes)}")  # Using sizes instead of dims
            print(f"Available variables: {list(ds.data_vars.keys())}")
            
            # Standardize time coordinates
            standardized_ds = standardize_time_coords(ds, var_name)
            print(f"Standardized coordinates: {list(standardized_ds.coords.keys())}")
            print(f"Standardized dimensions: {dict(standardized_ds.sizes)}")  # Using sizes
            
            # Store the standardized dataset
            standardized_datasets[var_name] = standardized_ds
            print(f"Successfully processed {var_name}")
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    if not standardized_datasets:
        raise ValueError("No datasets could be processed successfully")
    
    print("\nAttempting to merge standardized datasets...")
    try:
        # Get the superset of all times from standardized datasets
        all_times = []
        for ds in standardized_datasets.values():
            try:
                # Handle both numpy arrays and xarray DataArrays
                times = ds.time.values if hasattr(ds, 'time') else ds.indexes['time']
                all_times.extend(pd.to_datetime(times).astype('datetime64[ns]'))
            except Exception as e:
                print(f"Warning: Could not process times for a dataset: {e}")
                continue
        
        if not all_times:
            raise ValueError("No valid time data found in any dataset")
            
        # Convert to pandas DatetimeIndex for easier manipulation
        all_times = pd.DatetimeIndex(sorted(set(all_times)))
        print(f"Total unique timestamps before filtering: {len(all_times)}")
        
        # Filter to only keep 00:00, 06:00, 12:00, 18:00
        filtered_times = all_times[all_times.hour.isin([0, 6, 12, 18])]
        print(f"After filtering to 00, 06, 12, 18 UTC: {len(filtered_times)} timestamps")
        
        if len(filtered_times) == 0:
            raise ValueError("No timestamps remaining after filtering for 00, 06, 12, 18 UTC")
            
        # Ensure all datasets cover the same time period
        aligned_datasets = []
        for var_name, ds in standardized_datasets.items():
            print(f"Aligning {var_name}...")
            try:
                # Convert times to datetime64[ns] for consistent comparison
                if 'time' not in ds.coords:
                    print(f"Warning: No time coordinate found in {var_name}, skipping")
                    continue
                    
                # Convert time to pandas DatetimeIndex for consistent handling
                ds_time = pd.DatetimeIndex(ds.time.values)
                
                # Filter to only keep the desired hours
                mask = ds_time.hour.isin([0, 6, 12, 18])
                if not mask.any():
                    print(f"Warning: No data at 00, 06, 12, 18 UTC for {var_name}")
                    continue
                    
                # Select only the filtered times
                ds_filtered = ds.sel(time=mask)
                
                # Check for missing times
                ds_times = pd.DatetimeIndex(ds_filtered.time.values)
                missing_times = set(filtered_times) - set(ds_times)
                
                if len(missing_times) > 0:
                    print(f"Warning: {var_name} is missing {len(missing_times)} timestamps "
                          f"(out of {len(filtered_times)} total)")
                    
                    # Reindex to include all filtered times, filling with NaN where data is missing
                    ds_filtered = ds_filtered.reindex(time=filtered_times)
                
                aligned_datasets.append(ds_filtered)
                print(f"Aligned {var_name} to filtered time axis")
                
            except Exception as e:
                print(f"Error aligning {var_name}: {e}")
                continue
        
        # Try to merge all variables into a single dataset
        print("\nAttempting to merge all variables into a single dataset...")
        # First, drop any existing valid_time variables to avoid conflicts
        datasets_to_merge = []
        for ds in aligned_datasets:
            # Create a copy and drop valid_time if it exists
            ds_copy = ds.copy()
            if 'valid_time' in ds_copy.coords:
                ds_copy = ds_copy.drop_vars('valid_time', errors='ignore')
            datasets_to_merge.append(ds_copy)
            
        # Now try to merge with compat='override' to handle any remaining conflicts
        merged_ds = xr.merge(datasets_to_merge, compat='override')
        
        # Add metadata
        merged_ds.attrs['creation_date'] = str(pd.Timestamp.now())
        merged_ds.attrs['description'] = f'Merged ERA5 variables for {area_name}'
        merged_ds.attrs['source_files'] = ', '.join([os.path.basename(f) for f in nc_files])
        merged_ds.attrs['time_standardization'] = 'All variables aligned to common timesteps 00, 06, 12, 18 UTC'
        
        # Save the merged dataset
        merged_ds.to_netcdf(output_file)
        print(f"\nSuccessfully merged {len(standardized_datasets)} variables into {output_file}")
        
        # Print summary of the merged dataset
        print("\nMerged dataset summary:")
        print(merged_ds)
        
        return merged_ds
        
    except Exception as e:
        print(f"\nError merging datasets: {e}")
        
        # Try to provide more detailed error information
        print("\nDataset details for debugging:")
        for var_name, ds in standardized_datasets.items():
            print(f"\n{var_name}:")
            print("Coordinates:")
            for coord_name, coord in ds.coords.items():
                print(f"  {coord_name}: {coord.values.shape}")
            print(f"  time dtype: {ds.time.dtype}")
            print(f"  number of unique times: {len(np.unique(ds.time.values))}")
        
        raise

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Merge individual ERA5 variable files into a single dataset')
    parser.add_argument('processed_dir', help='Directory containing processed NetCDF files')
    parser.add_argument('--output', '-o', help='Output file path (optional)')
    parser.add_argument('--area', default='Bonn', help='Area name (default: Bonn)')
    
    args = parser.parse_args()
    
    try:
        merge_variable_files(args.processed_dir, args.output, args.area)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(main()) 
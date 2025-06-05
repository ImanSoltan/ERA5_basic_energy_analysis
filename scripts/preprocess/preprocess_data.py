import os
import glob
import xarray as xr
import numpy as np
import zipfile
import tempfile
import shutil
import pygrib

# Variable name mapping from GRIB shortName to standard names
VAR_NAME_MAP = {
    # Wind components
    '10u': 'u10',  # 10m u-component of wind
    '10v': 'v10',  # 10m v-component of wind
    
    # Temperature
    '2t': 't2m',   # 2m temperature
    
    # Radiation components
    'ssrd': 'ssrd',  # surface solar radiation downwards
    'fdir': 'fdir',  # total sky direct solar radiation at surface
    'ssr': 'ssr',    # surface net solar radiation
    'str': 'str',    # surface net thermal radiation
    'strd': 'strd',  # surface thermal radiation downwards
    
    # Other meteorological variables
    'tp': 'tp',    # total precipitation
    'sp': 'sp',    # surface pressure
    'tcc': 'tcc',  # total cloud cover
    'cbh': 'cbh',  # cloud base height
    
    # Additional radiation-related variables
    'tisr': 'tisr',  # top incoming solar radiation
    'ssrc': 'ssrc',  # clear-sky surface solar radiation
    'fdir': 'fdir'   # direct solar radiation at surface
}

def kelvin_to_celsius(temp_k):
    return temp_k - 273.15

def calculate_wind_speed(u, v):
    return np.sqrt(u**2 + v**2)

def extract_nc_from_zip(file_path):
    """
    If file_path is a zip file, extract the first .nc file inside and return its path.
    If not a zip, return file_path.
    """
    if zipfile.is_zipfile(file_path):
        with zipfile.ZipFile(file_path, 'r') as z:
            nc_files = [f for f in z.namelist() if f.endswith('.nc')]
            if not nc_files:
                raise ValueError(f"No .nc file found in zip archive: {file_path}")
            extract_path = os.path.join(os.path.dirname(file_path), nc_files[0])
            z.extract(nc_files[0], os.path.dirname(file_path))
            return extract_path
    return file_path

def extract_grib_from_zip(file_path):
    """
    If file_path is a zip file, extract the first .grib file inside and return its path.
    If not a zip, return file_path.
    """
    if zipfile.is_zipfile(file_path):
        with zipfile.ZipFile(file_path, 'r') as z:
            grib_files = [f for f in z.namelist() if f.endswith('.grib')]
            if not grib_files:
                raise ValueError(f"No .grib file found in zip archive: {file_path}")
            extract_path = os.path.join(os.path.dirname(file_path), grib_files[0])
            z.extract(grib_files[0], os.path.dirname(file_path))
            return extract_path
    return file_path

def try_open_dataset(file_path):
    """
    Try to open the file as NetCDF (with all supported engines) or as GRIB (with cfgrib).
    Returns the engine used if successful, else None.
    """
    # Try NetCDF engines
    for engine in ['netcdf4', 'h5netcdf', 'scipy']:
        try:
            xr.open_dataset(file_path, engine=engine).close()
            return engine
        except Exception:
            continue
    # Try GRIB engine
    try:
        xr.open_dataset(file_path, engine='cfgrib').close()
        return 'cfgrib'
    except Exception:
        pass
    return None

def list_grib_variable_combinations(file_path):
    """
    List all unique combinations of relevant keys in a GRIB file.
    Returns a list of dicts suitable for filter_by_keys.
    Also prints all unique key combinations for diagnostics.
    """
    combos = set()
    try:
        with pygrib.open(file_path) as grbs:
            for grb in grbs:
                combo = (
                    grb.shortName,
                    getattr(grb, 'typeOfLevel', None),
                    getattr(grb, 'stepType', None),
                    getattr(grb, 'level', None),
                    getattr(grb, 'step', None),
                    getattr(grb, 'dataType', None),
                    getattr(grb, 'productDefinitionTemplateNumber', None),
                    getattr(grb, 'forecastTime', None),
                    getattr(grb, 'typeOfStatisticalProcessing', None)
                )
                combos.add(combo)
    except Exception as e:
        print(f"Could not read GRIB file {file_path} with pygrib: {e}")
    # Print all unique key combinations for diagnostics
    print(f"\nUnique GRIB variable key combinations in {file_path}:")
    for combo in combos:
        print(combo)
    # Convert to list of dicts for filter_by_keys
    combo_dicts = []
    for (shortName, typeOfLevel, stepType, level, step, dataType, productDefinitionTemplateNumber, forecastTime, typeOfStatisticalProcessing) in combos:
        d = {'shortName': shortName}
        if typeOfLevel is not None:
            d['typeOfLevel'] = typeOfLevel
        if stepType is not None:
            d['stepType'] = stepType
        if level is not None:
            d['level'] = level
        if step is not None:
            d['step'] = step
        if dataType is not None:
            d['dataType'] = dataType
        if productDefinitionTemplateNumber is not None:
            d['productDefinitionTemplateNumber'] = productDefinitionTemplateNumber
        if forecastTime is not None:
            d['forecastTime'] = forecastTime
        if typeOfStatisticalProcessing is not None:
            d['typeOfStatisticalProcessing'] = typeOfStatisticalProcessing
        combo_dicts.append(d)
    return combo_dicts

def preprocess_era5_data(input_dir='data/raw', output_dir='data/processed', year=2024, months=None, area_name='Bonn'):
    """
    Preprocess ERA5 NetCDF files: merge, convert units, calculate derived variables, and save processed data.
    Handles zipped NetCDF files automatically.
    """
    if months is None:
        months = list(range(1, 13))  # Default: Jan-Dec
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all files for the specified year and months
    data_files = []
    years = year if isinstance(year, (list, tuple)) else [year]
    for y in years:
        for month in months:
            month_dir = os.path.join(input_dir, str(y), f"{month:02d}")
            grib_pattern = os.path.join(month_dir, "*.grib")
            nc_pattern = os.path.join(month_dir, "*.nc")
            print(f"Looking for files with pattern: {grib_pattern}")
            found_grib = glob.glob(grib_pattern)
            print(f"Found {len(found_grib)} GRIB files.")
            print(f"Looking for files with pattern: {nc_pattern}")
            found_nc = glob.glob(nc_pattern)
            print(f"Found {len(found_nc)} NetCDF files.")
            data_files.extend(found_grib)
            data_files.extend(found_nc)
    
    if not data_files:
        print(f"No files found for {year} months {months} in {input_dir}")
        return

    print(f"Found {len(data_files)} files. Processing...")

    # Dictionary to store datasets by variable
    datasets_by_var = {}
    
    for f in data_files:
        print(f"\nProcessing file: {f}")
        
        try:
            # For GRIB files, process each variable separately
            if f.endswith('.grib'):
                # First, list all variables in the file
                try:
                    with pygrib.open(f) as grbs:
                        messages = [grb for grb in grbs]
                        print(f"Found {len(messages)} messages in {f}")
                        
                        # Group messages by shortName
                        messages_by_var = {}
                        for msg in messages:
                            shortName = msg.shortName
                            if shortName not in messages_by_var:
                                messages_by_var[shortName] = []
                            messages_by_var[shortName].append(msg)
                        
                        # Process each variable separately
                        for shortName, msgs in messages_by_var.items():
                            print(f"Processing {shortName} with {len(msgs)} messages")
                            
                            # Special handling for total precipitation
                            if shortName == 'tp':
                                filter_keys = {
                                    'shortName': 'tp',
                                    'stepType': 'instant',
                                    'typeOfLevel': 'surface'
                                }
                            else:
                                filter_keys = {'shortName': shortName}
                            
                            try:
                                ds = xr.open_dataset(f, engine='cfgrib', filter_by_keys=filter_keys)
                                
                                # Map variable name if needed
                                standard_name = VAR_NAME_MAP.get(shortName, shortName)
                                if shortName in ds.data_vars:
                                    ds = ds.rename({shortName: standard_name})
                                
                                # Store in our dictionary using the standard name
                                if standard_name not in datasets_by_var:
                                    datasets_by_var[standard_name] = []
                                datasets_by_var[standard_name].append(ds)
                                
                            except Exception as e:
                                print(f"Error processing {shortName}: {e}")
                                continue
                            
                except Exception as e:
                    print(f"Error reading GRIB file with pygrib: {e}")
                    continue
                    
            # For NetCDF files, process normally
            else:
                ds = xr.open_dataset(f)
                for var in ds.data_vars:
                    # Map variable name if needed for NetCDF files too
                    standard_name = VAR_NAME_MAP.get(var, var)
                    if var != standard_name:
                        ds = ds.rename({var: standard_name})
                    
                    if standard_name not in datasets_by_var:
                        datasets_by_var[standard_name] = []
                    datasets_by_var[standard_name].append(ds[standard_name])
                    
        except Exception as e:
            print(f"Error processing {f}: {e}")
            continue

    # Merge and save each variable separately
    print("\nMerging and saving variables...")
    merged_datasets = {}
    individual_var_files_to_keep = [] # Initialize list to store paths of individual var files
    
    for var, ds_list in datasets_by_var.items():
        try:
            print(f"Merging {var}...")
            if len(ds_list) > 1:
                merged = xr.merge(ds_list)
            else:
                merged = ds_list[0]
            
            # Save individual variable
            out_path = os.path.join(output_dir, f"{var}_{area_name}.nc")
            merged.to_netcdf(out_path)
            print(f"Saved {var} to {out_path}")
            individual_var_files_to_keep.append(os.path.abspath(out_path)) # Add to keep list
            
            merged_datasets[var] = merged
            
        except Exception as e:
            print(f"Error merging {var}: {e}")
            continue
    
    # Try to merge all variables into a single dataset
    try:
        print("\nAttempting to merge all variables into a single dataset...")
        final_ds = xr.merge(list(merged_datasets.values()))
        final_path = os.path.join(output_dir, f"era5_all_vars_{area_name}.nc")
        final_ds.to_netcdf(final_path)
        print(f"Successfully saved merged dataset to {final_path}")
    except Exception as e:
        print(f"Error creating final merged dataset: {e}")
        print("Individual variable files are still available in the output directory.")

    # --- Merge all variables into a single dataset ---
    print("\nMerging all variables into a single dataset...")
    try:
        from .merge_variables import merge_variable_files
        merged_file = os.path.join(output_dir, f'era5_merged_{area_name}.nc')
        merge_variable_files(
            processed_dir=output_dir,
            output_file=merged_file,
            area_name=area_name
        )
        print(f"Successfully created merged dataset: {merged_file}")
    except Exception as e:
        print(f"Warning: Could not merge variables: {e}")
        print("Individual variable files will be preserved.")
        merged_file = None

    # --- Extract site time series for all variables ---
    print("\nExtracting site time series for all variables...")
    site_timeseries_file = os.path.join(output_dir, f"site_timeseries_{area_name}.nc")
    extract_site_timeseries(output_dir, site_timeseries_file)

    # --- Cleanup: Remove intermediate NetCDF files except the final outputs ---
    print("\nCleaning up intermediate NetCDF files...")
    keep_files = [
        os.path.abspath(merged_file) if merged_file else None,
        os.path.abspath(site_timeseries_file)
    ]
    keep_files.extend(individual_var_files_to_keep) # Add individual var files to keep_files
    keep_files = [f for f in keep_files if f and os.path.exists(f)] # Filter out None or non-existent paths
    
    for f in glob.glob(os.path.join(output_dir, "*.nc")):
        f_abs = os.path.abspath(f)
        if f_abs not in keep_files:
            try:
                os.remove(f)
                print(f"Deleted intermediate file: {f}")
            except Exception as e:
                print(f"Could not delete {f}: {e}")
    
    print("\nPreprocessing complete. The following files were preserved:")
    for f in keep_files:
        print(f"  - {f}")

def extract_site_timeseries(processed_dir, output_file, site_lat=50.73, site_lon=7.10):
    """
    For each NetCDF or GRIB file in processed_dir, extract the nearest point to (site_lat, site_lon)
    and combine all variables into a single Dataset (if possible).
    """
    files = glob.glob(os.path.join(processed_dir, "*.nc")) + glob.glob(os.path.join(processed_dir, "*.grib"))
    site_data = {}
    processed_count = 0
    
    if not files:
        print(f"Warning: No NetCDF or GRIB files found in {processed_dir}")
        return
    
    print(f"Processing {len(files)} files for site timeseries extraction...")
    
    for f in files:
        try:
            # Skip files that are already site timeseries to avoid recursion
            if 'site_timeseries' in f or '_site.' in f:
                continue
                
            # Choose engine based on file extension
            if f.endswith('.grib'):
                engine = 'cfgrib'
            else:
                engine = None
                
            # Open the dataset
            with xr.open_dataset(f, engine=engine) as ds:
                if not ds.data_vars:
                    print(f"Warning: No variables found in {os.path.basename(f)}, skipping")
                    continue
                    
                # Process each variable in the file
                for var_name, var_data in ds.data_vars.items():
                    try:
                        # Skip if not a data variable with lat/lon dimensions
                        if not all(dim in var_data.dims for dim in ['latitude', 'longitude']):
                            continue
                            
                        # Find nearest point
                        point = var_data.sel(
                            latitude=site_lat,
                            longitude=site_lon,
                            method="nearest"
                        )
                        
                        # Drop lat/lon dims if present
                        point = point.drop_vars([c for c in point.coords if c in ['latitude', 'longitude']])
                        
                        # Store the point data
                        site_data[var_name] = point
                        processed_count += 1
                        
                    except Exception as var_e:
                        print(f"Warning: Could not process {var_name} in {os.path.basename(f)}: {str(var_e)}")
                        continue
                        
        except Exception as e:
            print(f"Error processing {os.path.basename(f)}: {str(e)}")
            continue
    
    # Try to combine into a single Dataset (if we have any data)
    if not site_data:
        print("Warning: No valid site data was extracted from any files.")
        return
        
    print(f"\nSuccessfully extracted {processed_count} variables from {len(files)} files")
    
    try:
        combined = xr.Dataset(site_data)
        combined.attrs['site_latitude'] = site_lat
        combined.attrs['site_longitude'] = site_lon
        combined.attrs['extraction_time'] = str(pd.Timestamp.now())
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Save the combined dataset
        combined.to_netcdf(output_file)
        print(f"\nSaved combined site dataset with {len(site_data)} variables to {output_file}")
        
    except Exception as e:
        print(f"Warning: Could not combine all variables into a single file: {str(e)}")
        print("Saving variables as individual files...")
        
        # Save each variable separately
        for var_name, da in site_data.items():
            try:
                var_file = os.path.join(processed_dir, f"{var_name}_site.nc")
                da.to_netcdf(var_file)
                print(f"  - Saved {var_name} to {os.path.basename(var_file)}")
            except Exception as save_e:
                print(f"  - Failed to save {var_name}: {str(save_e)}")

if __name__ == "__main__":
    preprocess_era5_data() 
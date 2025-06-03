import os
import glob
import xarray as xr
import numpy as np
import zipfile
import tempfile

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
    print(f"Found {len(data_files)} files. Loading and merging...")

    # Prepare a list to hold valid files
    valid_files = []
    temp_dirs = []
    for f in data_files:
        # If it's a zip, extract all contents to a temp dir
        if zipfile.is_zipfile(f):
            temp_dir = tempfile.mkdtemp()
            temp_dirs.append(temp_dir)
            with zipfile.ZipFile(f, 'r') as z:
                z.extractall(temp_dir)
            extracted_files = [os.path.join(temp_dir, name) for name in z.namelist()]
            for ef in extracted_files:
                engine = try_open_dataset(ef)
                if engine:
                    valid_files.append((ef, engine))
                else:
                    print(f"Skipping extracted {ef}: not a valid NetCDF or GRIB file.")
        else:
            engine = try_open_dataset(f)
            if engine:
                valid_files.append((f, engine))
            else:
                print(f"Skipping {f}: not a valid NetCDF or GRIB file.")
    if not valid_files:
        print("No valid NetCDF or GRIB files to process after extraction/validation.")
        return

    # Open and merge all files along the time dimension
    datasets = []
    for f, engine in valid_files:
        ds = xr.open_dataset(f, engine=engine)
        datasets.append(ds)
    ds = xr.merge(datasets)
    
    # Unit conversions
    if '2m_temperature' in ds:
        ds['2m_temperature_c'] = kelvin_to_celsius(ds['2m_temperature'])
    if 'surface_thermal_radiation_downwards' in ds:
        ds['surface_thermal_radiation_downwards'] = ds['surface_thermal_radiation_downwards']
    # Convert accumulated J/m2 to W/m2 for solar/thermal radiation (if needed)
    for var in [
        'surface_solar_radiation_downwards',
        'surface_net_solar_radiation',
        'surface_net_thermal_radiation',
        'surface_thermal_radiation_downwards',
        'total_sky_direct_solar_radiation_at_surface',
    ]:
        if var in ds:
            # These are accumulated over the hour, so divide by 3600 to get W/m2
            ds[var + '_w_m2'] = ds[var] / 3600.0
    # Derived variables
    if '10m_u_component_of_wind' in ds and '10m_v_component_of_wind' in ds:
        ds['10m_wind_speed'] = calculate_wind_speed(ds['10m_u_component_of_wind'], ds['10m_v_component_of_wind'])
    # Save processed data
    out_path = os.path.join(output_dir, f"era5_processed_{area_name}_{year}_months_{months[0]}-{months[-1]}.nc")
    ds.to_netcdf(out_path)
    print(f"Processed data saved to {out_path}")

if __name__ == "__main__":
    preprocess_era5_data() 
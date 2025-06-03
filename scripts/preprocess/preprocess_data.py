import os
import glob
import xarray as xr
import numpy as np
import zipfile

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

def preprocess_era5_data(input_dir='data/raw', output_dir='data/processed', year=2024, months=None, area_name='Bonn'):
    """
    Preprocess ERA5 NetCDF files: merge, convert units, calculate derived variables, and save processed data.
    Handles zipped NetCDF files automatically.
    """
    if months is None:
        months = list(range(1, 13))  # Default: Jan-Dec
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all NetCDF or GRIB files for the specified year and months
    data_files = []
    for month in months:
        nc_pattern = os.path.join(input_dir, str(year), f"{month:02d}", f"era5_{year}_{month:02d}.nc")
        grib_pattern = os.path.join(input_dir, str(year), f"{month:02d}", f"era5_{year}_{month:02d}.grib")
        found_nc = glob.glob(nc_pattern)
        found_grib = glob.glob(grib_pattern)
        for f in found_nc:
            try:
                data_files.append(extract_nc_from_zip(f))
            except Exception as e:
                print(f"Skipping {f}: {e}")
        for f in found_grib:
            try:
                data_files.append(extract_grib_from_zip(f))
            except Exception as e:
                print(f"Skipping {f}: {e}")
    if not data_files:
        print(f"No NetCDF or GRIB files found for {year} months {months} in {input_dir}")
        return
    print(f"Found {len(data_files)} files. Loading and merging...")
    
    # Validate all files are NetCDF or GRIB before merging
    valid_files = []
    for f in data_files:
        try:
            if f.endswith('.nc'):
                xr.open_dataset(f).close()
            elif f.endswith('.grib'):
                xr.open_dataset(f, engine='cfgrib').close()
            else:
                raise ValueError("Unknown file type")
            valid_files.append(f)
        except Exception as e:
            print(f"Skipping {f}: not a valid NetCDF or GRIB file. {e}")
    if not valid_files:
        print("No valid NetCDF or GRIB files to process after extraction/validation.")
        return
    
    # Open and merge all files along the time dimension
    datasets = []
    for f in valid_files:
        if f.endswith('.nc'):
            ds = xr.open_dataset(f)
        elif f.endswith('.grib'):
            ds = xr.open_dataset(f, engine='cfgrib')
        else:
            continue
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
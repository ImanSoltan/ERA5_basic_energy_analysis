import xarray as xr
import sys

def print_netcdf_info(filepath):
    print(f"\nInspecting NetCDF file: {filepath}")
    ds = xr.open_dataset(filepath)
    print("\nDataset info:")
    print(ds.info())
    print("\nTime dimension length:", len(ds.time))
    print("\nFirst few timestamps:")
    print(ds.time.values[:5])
    print("\nLast few timestamps:")
    print(ds.time.values[-5:])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_netcdf.py <netcdf_file>")
        sys.exit(1)
    print_netcdf_info(sys.argv[1]) 
import cdsapi
import os
import calendar

# Default parameters (can be customized by the user)
DEFAULT_YEARS = [2024] # Consider updating to 2025 or a relevant recent year for actual runs if needed
DEFAULT_MONTHS = list(range(1, 13))  # January to December
DEFAULT_AREA = [50.75, 7.0, 50.70, 7.15]  # North, West, South, East for Bonn, Germany

# Updated DEFAULT_VARIABLES with common short names for ERA5 Single Levels
# Refer to the CDS documentation for the definitive list and potential alternatives:
# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form
# Click on "Variables" and then "Show MARS parameters" for short names
DEFAULT_VARIABLES = [
    # Radiation components
    "surface_solar_radiation_downwards",                  # ssrd
    "surface_net_solar_radiation",                       # ssr
    "surface_thermal_radiation_downwards",               # strd
    "surface_net_thermal_radiation",                     # str
    "total_sky_direct_solar_radiation_at_surface",       # fdir
    "top_net_solar_radiation",                          # tisr
    "surface_solar_radiation_downwards_clear_sky",       # ssrc
    
    # Wind components
    "10m_u_component_of_wind",                          # u10
    "10m_v_component_of_wind",                          # v10
    
    # Temperature
    "2m_temperature",                                   # t2m
    
    # Other meteorological variables
    "total_precipitation",                              # tp
    "surface_pressure",                                 # sp
    "total_cloud_cover",                               # tcc
    "cloud_base_height",                               # cbh
]
# Note on Fluxes and Radiation:
# - Many flux/radiation parameters are accumulated over the time step
# - Values represent the average rate over the time step
# - Units are typically in W/m² for radiation
# - Some variables have both instantaneous and accumulated versions

DEFAULT_BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw"))

def download_era5_data(
    years=None,
    months=None,
    area=None,
    variables=None,
    base_path=None,
    product_type='reanalysis',
):
    """
    Downloads ERA5 data from the Copernicus Climate Data Store (CDS).
    """
    if years is None:
        years = DEFAULT_YEARS
    if months is None:
        months = DEFAULT_MONTHS
    if area is None:
        area = DEFAULT_AREA
    if variables is None:
        variables = DEFAULT_VARIABLES
    if base_path is None:
        base_path = DEFAULT_BASE_PATH

    hourly_times = ["00:00", "06:00", "12:00", "18:00"]

    c = cdsapi.Client() # Assumes .cdsapirc file is configured

    for year in years:
        year_path = os.path.join(base_path, str(year))
        os.makedirs(year_path, exist_ok=True)

        for month in months:
            month_path = os.path.join(year_path, f"{month:02d}")
            os.makedirs(month_path, exist_ok=True)

            num_days = calendar.monthrange(year, month)[1]
            days_in_month = [f"{day:02d}" for day in range(1, num_days + 1)]

            target_file = os.path.join(month_path, f"era5_{year}_{month:02d}.grib")

            if os.path.exists(target_file):
                print(f"File {target_file} already exists. Skipping download.")
                continue

            print(f"Requesting data for {year}-{month:02d}...")
            print(f"Variables (short names): {variables}")
            print(f"Area: {area}")

            request = {
                "product_type": product_type,
                "variable": variables,
                "year": str(year),
                "month": f"{month:02d}",
                "day": days_in_month,
                "time": hourly_times,
                "area": area,
                "format": "grib",
            }

            try:
                c.retrieve(
                    'reanalysis-era5-single-levels',
                    request,
                    target_file
                )
                print(f"Successfully downloaded {target_file}")
            except Exception as e:
                print(f"Error downloading data for {year}-{month:02d}: {e}")
                if os.path.exists(target_file):
                    os.remove(target_file)

if __name__ == "__main__":
    # Create a dummy __file__ for testing if not present
    if '__file__' not in globals():
        __file__ = 'dummy_downloader_script.py'
        # Create dummy directory structure for DEFAULT_BASE_PATH to resolve
        os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")), exist_ok=True)


    print("Starting ERA5 data download process...")
    download_era5_data()
    print("ERA5 data download process finished.")
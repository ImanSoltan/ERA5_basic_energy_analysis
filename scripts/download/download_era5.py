import cdsapi
import os
import calendar

# Default parameters (can be customized by the user)
DEFAULT_YEARS = [2024]
DEFAULT_MONTHS = list(range(1, 13))  # January to December
DEFAULT_AREA = [50.75, 7.0, 50.70, 7.15]  # North, West, South, East for Bonn, Germany
DEFAULT_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "surface_pressure",
    "total_precipitation",
    "surface_latent_heat_flux",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
    "surface_sensible_heat_flux",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
    "total_sky_direct_solar_radiation_at_surface",
    "cloud_base_height",
    "total_cloud_cover"
]
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

    hourly_times = [f"{h:02d}:00" for h in range(24)]

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
            print(f"Variables: {variables}")
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
    print("Starting ERA5 data download process...")
    download_era5_data()
    print("ERA5 data download process finished.") 
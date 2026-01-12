==========================================
Data Download & Management
==========================================
**Section under Construction**
The ``WAS_Download`` module provides a unified interface to retrieve climate data required for seasonal forecasting. It handles authentication (CDS API), protocol management (HTTP/FTP), file format conversion (TIFF/GRIB to NetCDF), and spatiotemporal standardization.

Three types of data can be downloaded:
1.  **GCM Data**: Seasonal forecasts (Hindcasts and Real-time).
2.  **Reanalysis**: Historical baselines (ERA5, ERSST).
3.  **Observations**: Satellite-based products (CHIRPS, TAMSAT, AgERA5).

**Prerequisites**:
For C3S data (ECMWF, UKMO, Météo-France, etc.), you must have a `.cdsapirc` file configured in your home directory. See `CDS API How-to <https://cds.climate.copernicus.eu/api-how-to>`_.

-------------------------------------------------------------------------------

1. Seasonal GCM Forecasts
=========================

This section handles data from the Copernicus Climate Change Service (C3S) and the North American Multi-Model Ensemble (NMME).

Monthly GCM Data
----------------

**Method**: ``WAS_Download_Models``

Downloads monthly mean hindcasts or forecasts. It automatically handles the differences between C3S (NetCDF via API) and NMME (NetCDF/CPT via FTP).

**Parameters**:

* ``dir_to_save`` (str): Target directory.
* ``center_variable`` (list): Format ``"CENTER_SYSTEM.VAR"``.
    * *Centers*: ECMWF_51, UKMO_604, METEOFRANCE_8, NCEP_2, CMCC_35, DWD_21, JMA_3, ECCC_4.
    * *NMME*: CFSV2_1, CMC1_1, CMC2_1, GFDL_1, NASA_1, NCAR_CCSM4_1.
    * *Variables*: PRCP, TEMP, TMAX, TMIN, SST, SLP, UGRD10, VGRD10.
* ``month_of_initialization`` (int): 1-12.
* ``lead_time`` (list): List of lead months (e.g., ``['01', '02', '03']``).
* ``year_forecast`` (int, optional): If provided, downloads real-time forecast. If None, downloads hindcasts.



.. code-block:: python

    from wass2s import WAS_Download

    downloader = WAS_Download()

    # Download Hindcasts (1993-2016) for ECMWF and NCEP Precipitation
    downloader.WAS_Download_Models(
        dir_to_save="./data/GCM",
        center_variable=["ECMWF_51.PRCP", "NCEP_2.PRCP"],
        month_of_initialization=5,  # May starts
        lead_time=["01", "02", "03"], # Jun, Jul, Aug
        year_start_hindcast=1993,
        year_end_hindcast=2016,
        area=[20, -20, 0, 10]  # [N, W, S, E]
    )

Daily GCM Data
--------------

**Method**: ``WAS_Download_Models_Daily``

Downloads daily or sub-daily data (e.g., for heatwave or dry spell analysis). Note that daily data is voluminous.

**Parameters**:
Adds ``day_of_initialization`` and uses ``leadtime_hour`` (e.g., "24", "48"...) instead of months.

.. code-block:: python

    # Download Daily Forecast for specific initialization
    downloader.WAS_Download_Models_Daily(
        dir_to_save="./data/GCM_Daily",
        center_variable=["ECMWF_51.TMAX"],
        month_of_initialization=5,
        day_of_initialization=1,
        leadtime_hour=["24", "48", "72", "96"], # First 4 days
        year_start_hindcast=2000,
        year_end_hindcast=2020,
        area=[20, -20, 0, 10]
    )

-------------------------------------------------------------------------------

2. Reanalysis Data
==================

Used for model calibration and verification. Supports ERA5 (Atmosphere), ERA5-Land (Surface), and NOAA ERSST (Ocean).

ERA5 & ERSST
------------

**Method**: ``WAS_Download_Reanalysis``

Downloads monthly means. It handles cross-year seasons (e.g., DJF) by downloading appropriate months from adjacent years and aggregating them.

* **ERA5**: Downloads from CDS.
* **NOAA ERSST**: Downloads V5/V6 from NCEI/IRIDL.

**Usage Example**:

.. code-block:: python

    # Download SST for Nino 3.4 calculation
    downloader.WAS_Download_Reanalysis(
        dir_to_save="./data/Reanalysis",
        center_variable=["NOAA.SST"],
        year_start=1981,
        year_end=2020,
        area=[5, -170, -5, -120], # Pacific box
        seas=["11", "12", "01"],  # NDJ Season
        force_download=False
    )

ERA5-Land
---------

**Method**: ``WAS_Download_ERA5Land`` & ``WAS_Download_ERA5Land_daily``

Higher resolution (9km) land data, ideal for hydrological applications (Runoff, Soil Moisture).

.. code-block:: python

    # Download monthly ERA5-Land Soil Moisture
    downloader.WAS_Download_ERA5Land(
        dir_to_save="./data/Reanalysis",
        center_variable=["ERA5Land.SOILWATER1"],
        year_start=1981,
        year_end=2020,
        area=[15, -18, 4, 10], # West Africa
        seas=["06", "07", "08"] # JJA
    )

-------------------------------------------------------------------------------

3. Observational Data (Satellite)
=================================

High-resolution gridded observations for calibration and verification.

Agro-Climatic Indicators (AgERA5)
---------------------------------

**Method**: ``WAS_Download_AgroIndicators``

Derived from ERA5, corrected against stations. Good for temperature and general indices.

* **Variables**: ``AGRO.PRCP``, ``AGRO.TMAX``, ``AGRO.TMIN``, ``AGRO.DSWR`` (Solar Radiation), ``AGRO.ETP`` (Evapotranspiration).

CHIRPS Precipitation
--------------------

**Method**: ``WAS_Download_CHIRPSv3_Seasonal`` & ``_Daily``

Downloads high-resolution (0.05°) precipitation data from the Climate Hazards Group.
* Fetches TIFF files from UCSB servers.
* Merges, reprojects, and saves as NetCDF.



.. code-block:: python

    # Download Seasonal (aggregated) CHIRPS
    downloader.WAS_Download_CHIRPSv3_Seasonal(
        dir_to_save="./data/Obs",
        variables=["PRCP"],
        year_start=1981,
        year_end=2020,
        region="africa",
        season_months=["06", "07", "08", "09"], # JJAS
        area=[20, -20, 0, 15]
    )

TAMSAT Precipitation
--------------------

**Method**: ``WAS_Download_TAMSAT_Seasonal`` & ``_Daily``

Downloads rainfall estimates (RFE) or Soil Moisture from the University of Reading (TAMSAT).
* **Product**: ``rfe`` (Rainfall) or ``soil_moisture``.

.. code-block:: python

    # Download Daily TAMSAT Rainfall
    downloader.WAS_Download_TAMSAT_Daily(
        dir_to_save="./data/Obs",
        product="rfe",
        year_start=2023,
        year_end=2023,
        area=[20, -20, 0, 15]
    )

----------------
Download module
----------------

Three types of data can be downloaded with wass2s:

- GCM data on seasonal time scales
- Reanalysis data
- Observational data (satellite data, products combining satellite and observational data)

For some data, for instance `C3S <https://cds.climate.copernicus.eu/>`_, it requires creating an account, accepting the terms of use, and configuring an API key (`CDS API key<https://hmandela.github.io/WAS_S2S_Training/s2s_data.html>`_). 
Please refer also to the `CDS documentation <https://cds.climate.copernicus.eu/api-how-to>`_ for more instructions on how to set up the API key. 
For more information on C3S seasonal data, browse the `MetaData <https://confluence.ecmwf.int/display/CKB/Description+of+the+C3S+seasonal+multi-system>`_.

==============================================
Download GCM data
==============================================

The ``WAS_Download_Models`` method allows downloading seasonal forecast model data from various centers for specified variables, initialization months, lead times, and years.

**Parameters:**

- ``dir_to_save`` (str): Directory to save the downloaded files.
- ``center_variable`` (list): List of center-variable identifiers, e.g., ["ECMWF_51.PRCP", "UKMO_604.TEMP"].
- ``month_of_initialization`` (int): Initialization month as an integer (1-12).
- ``lead_time`` (list): List of lead times in months.
- ``year_start_hindcast`` (int): Start year for hindcast data.
- ``year_end_hindcast`` (int): End year for hindcast data.
- ``area`` (list): Bounding box as [North, West, South, East] for clipping.
- ``year_forecast`` (int, optional): Forecast year if downloading forecast data. Defaults to None.
- ``ensemble_mean`` (str, optional): Can be "median", "mean", or None. Defaults to None.
- ``force_download`` (bool): If True, forces download even if file exists.

**Available centers and variables:**

- **Centers:** BOM_2, ECMWF_51, UKMO_604, UKMO_603, METEOFRANCE_8, METEOFRANCE_9, DWD_21, DWD_22, CMCC_35, NCEP_2, JMA_3, ECCC_4, ECCC_5, CFSV2_1, CMC1_1, CMC2_1, GFDL_1, NASA_1, NCAR_CCSM4_1, NMME_1
- **Variables:** PRCP, TEMP, TMAX, TMIN, UGRD10, VGRD10, SST, SLP, DSWR, DLWR, HUSS_1000, HUSS_925, HUSS_850, UGRD_1000, UGRD_925, UGRD_850, VGRD_1000, VGRD_925, VGRD_850

**Note:** Some models are part of the NMME (North American Multi-Model Ensemble) project. For more information, see the `NMME documentation <https://www.cpc.ncep.noaa.gov/products/NMME/>`_. 
If ``year_forecast`` is not specified, hindcast data is downloaded; otherwise, forecast data for the specified year is retrieved.

**Example:**

.. code-block:: python

    from wass2s import *

    downloader = WAS_Download()

    downloader.WAS_Download_Models(
        dir_to_save="/path/to/save",
        center_variable=["ECMWF_51.PRCP"],
        month_of_initialization="03",
        lead_time=["01", "02", "03"],
        year_start_hindcast=1993,
        year_end_hindcast=2016,
        area=[60, -180, -60, 180],
        force_download=False
    )

==============================================
Download daily GCM data
==============================================

The ``WAS_Download_Models_Daily`` method allows downloading daily or sub-daily seasonal forecast model data from various centers for specified variables, initialization dates, lead times, and years.

**Parameters:**

- ``dir_to_save`` (str): Directory to save the downloaded files.
- ``center_variable`` (list): List of center-variable identifiers, e.g., ["ECMWF_51.PRCP", "UKMO_604.TEMP"].
- ``month_of_initialization`` (int): Initialization month as an integer (1-12).
- ``day_of_initialization`` (int): Initialization day as an integer (1-31).
- ``leadtime_hour`` (list): List of lead times in hours, e.g., ["24", "48", ..., "5160"].
- ``year_start_hindcast`` (int): Start year for hindcast data.
- ``year_end_hindcast`` (int): End year for hindcast data.
- ``area`` (list): Bounding box as [North, West, South, East] for clipping.
- ``year_forecast`` (int, optional): Forecast year if downloading forecast data. Defaults to None.
- ``ensemble_mean`` (str, optional): Can be "mean", "median", or None. Defaults to None.
- ``force_download`` (bool): If True, forces download even if file exists.

**Available centers and variables:**

- **Centers:** ECMWF_51, UKMO_604, UKMO_603, METEOFRANCE_8, DWD_21, DWD_22, CMCC_35, NCEP_2, JMA_3, ECCC_4, ECCC_5
- **Variables:** PRCP, TEMP, TMAX, TMIN, UGRD10, VGRD10, SST, SLP, DSWR, DLWR, HUSS_1000, HUSS_925, HUSS_850, UGRD_1000, UGRD_925, UGRD_850, VGRD_1000, VGRD_925, VGRD_850

**Example:**

.. code-block:: python

    from wass2s import *

    downloader = WAS_Download()
    downloader.WAS_Download_Models_Daily(
        dir_to_save="/path/to/save",
        center_variable=["ECMWF_51.PRCP"],
        month_of_initialization="01",
        day_of_initialization="01",
        leadtime_hour=["24", "48", "72"],
        year_start_hindcast=1993,
        year_end_hindcast=2016,
        area=[60, -180, -60, 180],
        force_download=False
    )

==============================================
Download reanalysis data
==============================================

The ``WAS_Download_Reanalysis`` method downloads reanalysis data for specified center-variable combinations, years, and months, handling cross-year seasons.

**Parameters:**

- ``dir_to_save`` (str): Directory to save the downloaded files.
- ``center_variable`` (list): List of center-variable identifiers, e.g., ["ERA5.PRCP", "MERRA2.TEMP"].
- ``year_start`` (int): Start year for the data to download.
- ``year_end`` (int): End year for the data to download.
- ``area`` (list): Bounding box as [North, West, South, East] for clipping.
- ``seas`` (list): List of month strings representing the season, e.g., ["11", "12", "01"] for NDJ.
- ``force_download`` (bool): If True, forces download even if file exists.
- ``run_avg`` (int): Number of months for running average (default=3).

**Available centers and variables:**

- **Centers:** ERA5, MERRA2, NOAA (for SST)
- **Variables:** PRCP, TEMP, TMAX, TMIN, UGRD10, VGRD10, SST, SLP, DSWR, DLWR, HUSS_1000, HUSS_925, HUSS_850, UGRD_1000, UGRD_925, UGRD_850, VGRD_1000, VGRD_925, VGRD_850

**Example:**

.. code-block:: python

    from wass2s import *

    downloader = WAS_Download()
    downloader.WAS_Download_Reanalysis(
        dir_to_save="/path/to/save",
        center_variable=["ERA5.PRCP"],
        year_start=1993,
        year_end=2016,
        area=[60, -180, -60, 180],
        seas=["11", "12", "01"],
        force_download=False
    )


==============================================
Download observational data
==============================================

Observational data includes agro-meteorological indicators and satellite-based precipitation data like CHIRPS.

Agro-meteorological indicators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``WAS_Download_AgroIndicators`` method downloads agro-meteorological indicators for specified variables, years, and months, handling cross-year seasons.

**Parameters:**

- ``dir_to_save`` (str): Directory to save the downloaded files.
- ``variables`` (list): List of shorthand variables, e.g., ["AGRO.PRCP", "AGRO.TMAX"].
- ``year_start`` (int): Start year for the data to download.
- ``year_end`` (int): End year for the data to download.
- ``area`` (list): Bounding box as [North, West, South, East] for clipping.
- ``seas`` (list): List of month strings representing the season, e.g., ["11", "12", "01"] for NDJ.
- ``force_download`` (bool): If True, forces download even if file exists.

**Available variables:**

- AGRO.PRCP: precipitation_flux
- AGRO.TMAX: 2m_temperature (24_hour_maximum)
- AGRO.TEMP: 2m_temperature (24_hour_mean)
- AGRO.TMIN: 2m_temperature (24_hour_minimum)

**Example:**

.. code-block:: python

    from wass2s import *

    downloader = WAS_Download()
    downloader.WAS_Download_AgroIndicators(
        dir_to_save="/path/to/save",
        variables=["AGRO.PRCP"],
        year_start=1993,
        year_end=2016,
        area=[60, -180, -60, 180],
        seas=["11", "12", "01"],
        force_download=False
    )

Download daily agro-meteorological indicators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``WAS_Download_AgroIndicators_daily`` method downloads daily agro-meteorological indicators for specified variables and years.

**Parameters:**

- ``dir_to_save`` (str): Directory to save the downloaded files.
- ``variables`` (list): List of shorthand variables, e.g., ["AGRO.PRCP", "AGRO.TMAX"].
- ``year_start`` (int): Start year for the data to download.
- ``year_end`` (int): End year for the data to download.
- ``area`` (list): Bounding box as [North, West, South, East] for clipping.
- ``force_download`` (bool): If True, forces download even if file exists.

**Available variables:**

- AGRO.PRCP: precipitation_flux
- AGRO.TMAX: 2m_temperature (24_hour_maximum)
- AGRO.TEMP: 2m_temperature (24_hour_mean)
- AGRO.TMIN: 2m_temperature (24_hour_minimum)

**Example:**

.. code-block:: python

    from wass2s import *

    downloader = WAS_Download()
    downloader.WAS_Download_AgroIndicators_daily(
        dir_to_save="/path/to/save",
        variables=["AGRO.PRCP"],
        year_start=1993,
        year_end=2016,
        area=[60, -180, -60, 180],
        force_download=False
    )

CHIRPS precipitation data
^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``WAS_Download_CHIRPSv3`` method downloads CHIRPS v3.0 monthly precipitation data for a specified cross-year season.

**Parameters:**

- ``dir_to_save`` (str): Directory to save the downloaded files.
- ``variables`` (list): List of variables, typically ["PRCP"].
- ``year_start`` (int): Start year for the data to download.
- ``year_end`` (int): End year for the data to download.
- ``area`` (list, optional): Bounding box as [North, West, South, East] for clipping.
- ``season_months`` (list): List of month strings representing the season, e.g., ["03", "04", "05"] for MAM.
- ``force_download`` (bool): If True, forces download even if file exists.

**Note:** CHIRPS data is available for land areas between 50°S and 50°N.

**Example:**

.. code-block:: python

    from wass2s import *

    downloader = WAS_Download()
    downloader.WAS_Download_CHIRPSv3(
        dir_to_save="/path/to/save",
        variables=["PRCP"],
        year_start=1993,
        year_end=2016,
        area=[15, -20, -5, 20],  # Example for Africa
        season_months=["03", "04", "05"],
        force_download=False
    )


















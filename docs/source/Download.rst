Data Download
=============

The ``WAS_Download`` class provides a unified interface for retrieving all
climate data needed for seasonal forecasting: GCM hindcasts and real-time
forecasts, ERA5/ERSST reanalysis, and satellite-based observational products
(CHIRPS, TAMSAT, AgERA5).

All methods share the same behaviour:

* They check for an existing cached NetCDF file before triggering any network
  request. Pass ``force_download=True`` to override the cache.
* They aggregate monthly fields into seasonal totals internally, including
  cross-year seasons such as DJF.
* They return files with standardised ``T``, ``Y`` (latitude), and ``X``
  (longitude) dimensions.

.. code-block:: python

   from wass2s import WAS_Download

   downloader = WAS_Download()

.. note::
   C3S data (ECMWF, UKMO, Météo-France, DWD, CMCC, JMA, ECCC) requires a
   ``~/.cdsapirc`` credentials file. See :doc:`Installation` for setup
   instructions. NMME data is downloaded over HTTP without authentication.

-------------------------------------------------------------------------------

1. Seasonal GCM Forecasts
--------------------------

Monthly GCM Data
~~~~~~~~~~~~~~~~

**Method**: ``WAS_Download_Models``

Downloads monthly mean hindcasts (1993–2016 by default) or real-time forecasts
from C3S and NMME. The method automatically adapts the download protocol
(CDS API vs. IRI FTP) based on the requested centre.

Supported centres and their identifiers:
.. list-table:: Supported GCM centres
   :header-rows: 1
   :widths: 30 70

   * - Source
     - Centre identifiers
   * - C3S / ECMWF
     - ``ECMWF_51``
   * - C3S / UK Met Office
     - ``UKMO_603``, ``UKMO_604``, ``UKMO_605``, ``UKMO_610``
   * - C3S / Météo-France
     - ``METEOFRANCE_8``, ``METEOFRANCE_9``
   * - C3S / DWD
     - ``DWD_21``, ``DWD_22``
   * - C3S / CMCC
     - ``CMCC_35``, ``CMCC_4``
   * - C3S / JMA
     - ``JMA_3``, ``JMA_4``
   * - C3S / ECCC
     - ``ECCC_4``, ``ECCC_5``
   * - NMME / NCEP CFS
     - ``NCEP_2``, ``CFSV2_1``
   * - NMME
     - ``CMC1_1``, ``CMC2_1``, ``GFDL_1``, ``NASA_1``, ``NCAR_CCSM4_1``, ``NCAR_CESM1_1``
   * - BOM
     - ``BOM_2``
     
Available variables: ``PRCP``, ``TEMP``, ``TMAX``, ``TMIN``, ``SST``,
``SLP``, ``UGRD10``, ``VGRD10``, ``DSWR``, ``DLWR``, ``NOLR``.

Pressure-level wind and humidity are also available:
``UGRD_850``, ``VGRD_850``, ``HUSS_850``, and equivalents at 925 hPa
and 1000 hPa.

.. code-block:: python

   # Download ECMWF and NCEP precipitation hindcasts (May initialisation, JAS target)
   downloader.WAS_Download_Models(
       dir_to_save="./data/GCM/",
       center_variable=["ECMWF_51.PRCP", "NCEP_2.PRCP"],
       month_of_initialization=5,
       lead_time=["01", "02", "03"],       # June, July, August
       year_start_hindcast=1993,
       year_end_hindcast=2016,
       area=[20, -20, 0, 10],              # [N, W, S, E]
       ensemble_mean="mean"
   )

   # Download the 2025 real-time forecast
   downloader.WAS_Download_Models(
       dir_to_save="./data/GCM/",
       center_variable=["ECMWF_51.PRCP"],
       month_of_initialization=5,
       lead_time=["01", "02", "03"],
       year_start_hindcast=1993,
       year_end_hindcast=2016,
       area=[20, -20, 0, 10],
       year_forecast=2025,
       ensemble_mean="mean"
   )

Daily GCM Data
~~~~~~~~~~~~~~

**Method**: ``WAS_Download_Models_Daily``

Downloads daily or sub-daily data for applications such as heat-wave or
dry-spell analysis. Daily data volumes are large; restrict the area and
lead-time range accordingly.

.. code-block:: python

   downloader.WAS_Download_Models_Daily(
       dir_to_save="./data/GCM_Daily/",
       center_variable=["ECMWF_51.TMAX"],
       month_of_initialization=5,
       day_of_initialization=1,
       leadtime_hour=["24", "48", "72", "96"],
       year_start_hindcast=2000,
       year_end_hindcast=2020,
       area=[20, -20, 0, 10]
   )

-------------------------------------------------------------------------------

2. Reanalysis Data
------------------

ERA5 and ERSST
~~~~~~~~~~~~~~

**Method**: ``WAS_Download_Reanalysis``

Downloads monthly means from ERA5 (atmosphere and ocean surface) or NOAA
ERSSTv5. Cross-year seasons are handled automatically: requesting NDJ
(November–December–January) downloads the correct months from adjacent
calendar years and aggregates them.

.. code-block:: python

   # Global SST for predictor indices (NDJ season)
   downloader.WAS_Download_Reanalysis(
       dir_to_save="./data/Reanalysis/",
       center_variable=["ERA5.SST"],
       year_start=1981,
       year_end=2024,
       area=[45, -180, -45, 180],
       seas=["11", "12", "01"],
       force_download=False
   )

   # NOAA ERSST for Niño 3.4 calculation
   downloader.WAS_Download_Reanalysis(
       dir_to_save="./data/Reanalysis/",
       center_variable=["NOAA.SST"],
       year_start=1981,
       year_end=2024,
       area=[5, -170, -5, -120],
       seas=["11", "12", "01"]
   )

ERA5-Land
~~~~~~~~~

**Methods**: ``WAS_Download_ERA5Land`` (monthly) and
``WAS_Download_ERA5Land_daily``

ERA5-Land provides higher-resolution (~9 km) land surface variables, making
it particularly suited to hydrological applications.

.. code-block:: python

   # Monthly ERA5-Land soil moisture (JJA)
   downloader.WAS_Download_ERA5Land(
       dir_to_save="./data/Reanalysis/",
       center_variable=["ERA5Land.SOILWATER1"],
       year_start=1981,
       year_end=2024,
       area=[15, -18, 4, 10],
       seas=["06", "07", "08"]
   )

-------------------------------------------------------------------------------

3. Observational Data
---------------------

AgERA5 Agro-Climatic Indicators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Method**: ``WAS_Download_AgroIndicators``

AgERA5 is a bias-adjusted product derived from ERA5 and corrected against
station observations. It provides consistent daily time series of variables
widely used for agroclimatic index computation.

Available variables: ``AGRO.PRCP``, ``AGRO.TMAX``, ``AGRO.TMIN``,
``AGRO.DSWR`` (solar radiation), ``AGRO.ETP`` (reference evapotranspiration).

.. code-block:: python

   downloader.WAS_Download_AgroIndicators(
       dir_to_save="./data/Obs/",
       variables=["AGRO.PRCP"],
       year_start=1991,
       year_end=2024,
       area=[30, -25, 0, 30],
       seas=["06", "07", "08", "09"],
       force_download=False
   )

CHIRPS Precipitation
~~~~~~~~~~~~~~~~~~~~~

**Methods**: ``WAS_Download_CHIRPSv3_Seasonal`` and
``WAS_Download_CHIRPSv3_Daily``

CHIRPS v3 provides high-resolution (0.05°) precipitation estimates. The
seasonal method aggregates monthly totals; the daily method downloads
individual files.

.. code-block:: python

   # CHIRPS seasonal totals (JJAS)
   downloader.WAS_Download_CHIRPSv3_Seasonal(
       dir_to_save="./data/Obs/",
       variables=["PRCP"],
       year_start=1981,
       year_end=2024,
       region="africa",
       season_months=["06", "07", "08", "09"],
       area=[20, -20, 0, 15]
   )

   # CHIRPS daily files
   downloader.WAS_Download_CHIRPSv3_Daily(
       dir_to_save="./data/Obs_Daily/",
       variables=["PRCP"],
       year_start=2020,
       year_end=2024,
       area=[20, -20, 0, 15]
   )

TAMSAT Precipitation
~~~~~~~~~~~~~~~~~~~~~

**Methods**: ``WAS_Download_TAMSAT_Seasonal`` and
``WAS_Download_TAMSAT_Daily``

TAMSAT provides African rainfall estimates (RFE) and soil moisture from the
University of Reading.

.. code-block:: python

   # TAMSAT seasonal rainfall
   downloader.WAS_Download_TAMSAT_Seasonal(
       dir_to_save="./data/Obs/",
       product="rfe",
       year_start=1983,
       year_end=2024,
       area=[20, -20, 0, 15],
       season_months=["06", "07", "08", "09"]
   )

   # TAMSAT daily rainfall
   downloader.WAS_Download_TAMSAT_Daily(
       dir_to_save="./data/Obs_Daily/",
       product="rfe",
       year_start=2023,
       year_end=2023,
       area=[20, -20, 0, 15]
   )

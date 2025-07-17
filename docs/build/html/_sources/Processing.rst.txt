------------------
Processing Modules
------------------
The Processing modules provide tools for computing various climate indices or predictands from daily data, such as onset and cessation of the rainy season, dry and wet spells, number of rainy days, extreme precipitation indices, and heat wave indices. Additionally, it offers methods for merging or adjusting gridded data with station observations to correct biases.

These modules are divided into two main parts:

1. **Computing Predictands**: Classes for calculating different climate indices from daily data.
2. **Merging and Adjusting Data**: Classes for combining gridded data with station observations to improve accuracy.

**Prerequisites**

* **Dask**: Required for parallel processing in gridded data computations.
* **Data Formats**: Gridded data should be in xarray DataArray format with coordinates (T, Y, X). Station data should be in CDT format for daily data or CPT format for seasonal aggregation before merging.

**Climate Data Tools (CDT)**: Format for daily data.

============ ======== ========== 
ID           ALLADA   APLAHOUE   
============ ======== ========== 
LON          2.133333 1.666667    
LAT          6.65     6.916667      
DAILY/ELEV   92.0     153.0             
19810101     0.0      0.0               
19810102     0.0      0.0                
19810103     0.0      0.0                
19810104     0.0      0.0                
19810105     0.0      0.0               
19810106     0.0      0.0               
19810107     0.0      0.0              
19810108     0.0      0.0               
19810109     0.0      0.0               
19810110     0.0      0.0        
...      
============ ======== ==========


**Climate Prediction Tools (CPT)**: Format for seasonal aggregation (used in climate prediction tools) before merging.

======= ===== ====== =====
STATION ABEO  ABUJ   ADEK 
======= ===== ====== =====
LAT     7.2   7.6    9.0  
LON     3.3   5.2    7.2  
1991    514.9 715.1  934.3
1992    503.6 736.4  714.6
1993    414.6 891.0  709.6
1994    345.6 1034.7 491.7
1995    492.2 837.6  938.8
...
======= ===== ====== =====

==============================================
Computing Predictands
==============================================

This section includes classes for computing various climate indices:

* ``WAS_compute_onset``: Computes the onset of the rainy season.
* ``WAS_compute_cessation``: Computes the cessation of the rainy season.
* ``WAS_compute_onset_dry_spell``: Computes the longest dry spell after the onset.
* ``WAS_compute_cessation_dry_spell``: Computes the longest dry spell in flourishing period.
* ``WAS_count_wet_spells``: Computes the number of wet spells between onset and cessation.
* ``WAS_count_dry_spells``: Computes the number of dry spells between onset and cessation.
* ``WAS_count_rainy_days``: Computes the number of rainy days between onset and cessation.
* ``WAS_r95_99p``: Computes extreme precipitation indices R95p and R99p.
* ``WAS_compute_HWSDI``: Computes the Heat Wave Severity Duration Index.

Each class has methods for computing the index from gridded data (``compute``) and, where applicable, from station data in CDT format (``compute_insitu``).

**Onset Computation**

The ``WAS_compute_onset`` class computes the onset of the rainy season based on user-defined or default criteria for different zones.

**Initialization**

* ``__init__(self, user_criteria=None)``: Initializes the class with user-defined criteria. If not provided, default criteria are used.
* Dictionaries ``onset_criteria``,  ``cessation_criteria``, ``onset_dryspell_criteria``, ``cessation_dryspell_criteria`` show how to define the criteria for onset, cessation, onset dry spell and cessation dry spell computations.

**Methods**

* ``compute(self, daily_data, nb_cores)``: Computes onset dates for gridded daily rainfall data.
  * ``daily_data``: xarray DataArray with daily rainfall data (coords: T, Y, X).
  * ``nb_cores``: Number of CPU cores for parallel processing.
  * Returns: xarray DataArray with onset dates.

* ``compute_insitu(self, daily_df)``: Computes onset dates for station data in CDT format.
  * ``daily_df``: pandas DataFrame in CDT format.
  * Returns: pandas DataFrame in CPT format with onset dates.

**Criteria Dictionary**

The criteria dictionary defines parameters for onset computation:

.. code-block:: python

    {
        0: {"zone_name": "Sahel100_0mm", "start_search": "06-01", "cumulative": 10, "number_dry_days": 25, "thrd_rain_day": 0.85, "end_search": "08-30"},
        1: {"zone_name": "Sahel200_100mm", "start_search": "05-15", "cumulative": 15, "number_dry_days": 25, "thrd_rain_day": 0.85, "end_search": "08-15"},
        ...
    }

* ``zone_name``: Name of the zone.
* ``start_search``: Start date for searching the onset (e.g., "06-01").
* ``cumulative``: Cumulative rainfall threshold (mm).
* ``number_dry_days``: Maximum number of dry days allowed after onset.
* ``thrd_rain_day``: Rainfall threshold to consider a day as rainy (mm).
* ``end_search``: End date for searching the onset.

**Example**

.. code-block:: python

    from wass2s import *
    # Download daily rainfall data  
    downloader = WAS_Download()
    downloader.WAS_Download_AgroIndicators_daily(
        dir_to_save="/path/to/save",
        variables=["AGRO.PRCP"],
        year_start=1993,
        year_end=2016,
        area=[60, -180, -60, 180],
        force_download=False
    )

    # Load daily rainfall data
    rainfall = prepare_predictand(dir_to_save, variables, year_start, year_end, daily=True, ds=False)
    ## NB: prepare_predictand is a utility function that loads the data and prepares it for the computation of the predictand. 
    ## ds is set to False because the data will be loaded as dataarray.  

    # Print predefined  onset criteria
    onset_criteria
    # Define user criteria
    user_criteria = onset_criteria
    # adjust user criteria
    user_criteria[0]["start_search"] = "06-15"
    user_criteria[1]["end_search"] = "09-01"
    # Compute onset
    was_onset = WAS_compute_onset(user_criteria)
    onset = was_onset.compute(daily_data=rainfall, nb_cores=4)
    # Plot the mean onset date to check the results
    plot_date(onset.mean(dim='T'))

**Cessation Computation**

The ``WAS_compute_cessation`` class computes the cessation of the rainy season based on soil moisture balance criteria.

* Similar initialization and methods as ``WAS_compute_onset`` with criteria including:
  * ``date_dry_soil``: Date when soil is assumed dry (e.g., "01-01").
  * ``ETP``: Evapotranspiration rate (mm/day).
  * ``Cap_ret_maxi``: Maximum soil water retention capacity (mm).

**Dry Spell Computation**

The ``WAS_compute_onset_dry_spell`` class computes the longest dry spell after the onset.

* Includes an additional ``nbjour`` parameter in the criteria for the number of days to check after onset.

The ``WAS_compute_cessation_dry_spell`` class computes the longest dry spell in flourishing period.

* Includes an additional ``nbjour`` parameter in the criteria for the number of days to check after cessation.  

The ``WAS_count_dry_spells`` class computes the number of dry spells between onset and cessation. Requires onset and cessation dates as inputs.

**Wet Spell Computation**

The ``WAS_count_wet_spells`` class computes the number of wet spells between onset and cessation. Requires onset and cessation dates as inputs.

**Rainy Days Computation**

The ``WAS_count_rainy_days`` class computes the number of rainy days between onset and cessation. Requires onset and cessation dates as inputs.

**Extreme Precipitation Indices**

The ``WAS_r95_99p`` class computes R95p and R99p indices. Initialization with a base period (e.g., ``slice("1991-01-01", "2020-12-31")``) and optional season (list of months).

* Methods:
  * ``compute_r95p`` and ``compute_r99p`` for gridded data.
  * ``compute_insitu_r95p`` and ``compute_insitu_r99p`` for station data.

**Heat Wave Indices**

The ``WAS_compute_HWSDI`` class computes the Heat Wave Severity Duration Index. Computes TXin90 (90th percentile of daily max temperature) and counts heatwave days with at least 6 consecutive hot days.

==============================================
Merging and Adjusting Data
==============================================

The ``WAS_Merging`` class provides methods for merging gridded data with station observations to adjust for biases.

**Initialization**

* ``__init__(self, df, da, date_month_day="08-01")``: Initializes with station data DataFrame (CPT format), gridded data DataArray, and a date string.

**Methods**

* ``simple_bias_adjustment(self, missing_value=-999.0, do_cross_validation=False)``: Adjusts gridded data using kriging of residuals.
* ``regression_kriging(self, missing_value=-999.0, do_cross_validation=False)``: Uses linear regression followed by kriging of residuals.
* ``neural_network_kriging(self, missing_value=-999.0, do_cross_validation=False)``: Uses a neural network followed by kriging of residuals.
* ``multiplicative_bias(self, missing_value=-999.0, do_cross_validation=False)``: Applies a multiplicative bias correction.

Each method returns the adjusted gridded data as an xarray DataArray and optionally cross-validation results as a DataFrame.

* ``plot_merging_comparaison(self, df_Obs, da_estimated, da_corrected, missing_value=-999.0)``: Visualizes the comparison between observations, original estimates, and corrected data.


**Example: Merging Onset with Station Observations**

.. code-block:: python

    # Load station onset data in CPT format
    cpt_input_file_path = "./path/to/cpt_file.csv"
    df = pd.read_csv(cpt_input_file_path, na_values=-999.0, encoding="latin1")

    # Filter for relevant years and stations
    year_start, year_end = 1981, 2020  # Example years
    onset_df = df[(df['STATION'] == 'LAT') | (df['STATION'] == 'LON') | 
                  (pd.to_numeric(df['STATION'], errors='coerce').between(year_start, year_end))]

    # Verify station network 
    verify_station_network(onset_df, area)
    ## NB: verify_station_network is a utility function that verifies the station network. area is the extent of the gridded onset domain.

    # Instantiate WAS_Merging
    data_merger = WAS_Merging(onset_df, onset, date_month_day='02-01')
    ## NB: date_month_day is set to '02-01' because the onset start_search criteria is set to the month of February. 
    ## Important to verify the T dimension in the gridded onset computed. the month and day must match the date_month_day.      

    # Perform simple bias adjustment
    onset_adjusted, _ = data_merger.simple_bias_adjustment(do_cross_validation=False)

    # Plot comparison
    data_merger.plot_merging_comparaison(onset_df, onset, onset_adjusted)
    ## NB: plot_merging_comparaison is a utility function that plots the comparison between the station onset, the gridded onset and the adjusted onset.

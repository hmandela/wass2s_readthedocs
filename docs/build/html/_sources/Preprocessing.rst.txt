------------------
Preprocessing Modules
------------------
The Processing modules provide tools for computing various climate indices or predictands from daily data, such as onset and cessation of the rainy season, dry and wet spells, number of rainy days, extreme precipitation indices, and heat wave indices. Additionally, it offers methods for merging or adjusting gridded data with station observations to correct biases.

These modules are divided into two main parts:

1. **Computing Predictands**: Classes for calculating different climate indices from daily data.
2. **Merging and Adjusting Data**: Classes for combining gridded data with station observations to improve accuracy.
3. **Bias correction**: 
4. **Data Transformation**:

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

=======================
Computing Predictands
=======================

The ``WAS_compute_predictand`` module provides a suite of tools for calculating climate indices from daily meteorological data. It supports both **Gridded Data** (xarray) and **In-Situ Station Data** (CDT/CPT formats).

The module is divided into three main categories:

1.  **Agro-Climatology**: Onset, Cessation, and Season Length.
2.  **Spell Analysis**: Dry and Wet spells relative to the growing season.
3.  **ETCCDI Extremes**: Temperature percentiles, Heat Waves, and Extreme Precipitation (R95p).



-------------------------------------------------------------------------------

1. Agro-Climatic Seasonality
============================

These classes determine the start and end of the rainy season using zone-specific rainfall thresholds and soil moisture balance.

Onset Computation
-----------------

**Class**: ``WAS_compute_onset``

Calculates the start of the rains based on:
1.  **Start Search Date**: The earliest allowed date.
2.  **Cumulative Rainfall**: E.g., 20mm over 3 days.
3.  **Dry Spell Constraint**: No dry spell > 7 days in the following 30 days.

.. code-block:: python

   from wass2s import WAS_compute_onset

   # 1. Define Zone Criteria (or use defaults)
   # Zones map specific lat/lon regions to rainfall rules
   criteria = {
       0: {
           "zone_name": "Sahel", 
           "start_search": "06-01", 
           "cumulative": 20, 
           "number_dry_days": 10, 
           "thrd_rain_day": 0.85, 
           "end_search": "08-30"
       }
   }

   # 2. Initialize
   onset_calc = WAS_compute_onset(user_criteria=criteria)

   # 3. Compute (Gridded)
   # daily_rain must be xarray (T, Y, X)
   onset_da = onset_calc.compute(daily_data=daily_rain, nb_cores=4)

Cessation Computation
---------------------

**Class**: ``WAS_compute_cessation``

Calculates the end of the season based on a **Soil Water Balance** model. The season ends when the soil water content drops to zero after the rainy period.



.. code-block:: python

   from wass2s import WAS_compute_cessation

   # Cessation requires Evapotranspiration (ETP) and Soil Capacity
   criteria_cess = {
       0: {
           "zone_name": "Sahel", 
           "date_dry_soil": "01-01", # Date soil is assumed dry
           "start_search": "09-01", 
           "ETP": 5.0,               # mm/day
           "Cap_ret_maxi": 70,       # Max soil holding capacity (mm)
           "end_search": "10-30"
       }
   }

   cess_calc = WAS_compute_cessation(user_criteria=criteria_cess)
   cessation_da = cess_calc.compute(daily_data=daily_rain, nb_cores=4)

-------------------------------------------------------------------------------

2. Spell Analysis (Dry/Wet)
===========================

These classes analyze the distribution of rain *within* the computed season (between Onset and Cessation).

Post-Onset Dry Spell
--------------------

**Class**: ``WAS_compute_onset_dry_spell``

Finds the maximum length of a dry spell occurring immediately after the onset date (critical for seedling survival).

.. code-block:: python

   from wass2s import WAS_compute_onset_dry_spell

   # "nbjour": 30 means check for dry spells in the 30 days post-onset
   dry_calc = WAS_compute_onset_dry_spell()
   
   max_dry_spell_da = dry_calc.compute(daily_data=daily_rain, nb_cores=4)

Counts of Spells & Rainy Days
-----------------------------

**Classes**: 
* ``WAS_count_dry_spells``
* ``WAS_count_wet_spells``
* ``WAS_count_rainy_days``

These classes require the **Onset** and **Cessation** dates as inputs.

.. code-block:: python

   from wass2s import WAS_count_rainy_days

   # Inputs: Rainfall data, Onset Map, Cessation Map
   counter = WAS_count_rainy_days()

   # Count days where rain > 1.0mm between Onset and Cessation
   nb_rainy_da = counter.compute(
       daily_data=daily_rain,
       onset_date=onset_da,
       cessation_date=cessation_da,
       rain_threshold=1.0,
       nb_cores=4
   )

-------------------------------------------------------------------------------

3. ETCCDI Temperature Extremes
==============================

This module implements standard indices defined by the **Expert Team on Climate Change Detection and Indices (ETCCDI)**. It uses a 5-day centered window bootstrapping method for robust percentile calculation.



Factory Classes
---------------

Helper classes are provided to easily instantiate complex calculations.

**Class**: ``ETCCDITempIndices``

* ``hot_days``: Percentage of days where Tmax > 90th percentile (TX90p).
* ``hot_nights``: Percentage of days where Tmin > 90th percentile (TN90p).
* ``cold_days``: Percentage of days where Tmax < 10th percentile (TX10p).
* ``cold_nights``: Percentage of days where Tmin < 10th percentile (TN10p).

**Example: Calculating Hot Days (TX90p)**

.. code-block:: python

   from wass2s import ETCCDITempIndices

   # 1. Initialize Calculator
   # Define the climatological base period
   tx90p_calc = ETCCDITempIndices.hot_days(
       base_period=slice("1981", "2010"),
       season=[6, 7, 8] # Optional: JJA season
   )

   # 2. Compute
   # tmax_da is xarray (T, Y, X)
   tx90p_da = tx90p_calc.compute_xarray(tmax_da, parallel=True)

Heat Wave Indices
-----------------

**Class**: ``ETCCDIHeatWaveIndices``

* ``wsdi``: Warm Spell Duration Index (Count of days in heatwaves > 6 days).
* ``heat_wave_frequency``: Number of heatwave events.
* ``compound_heat_wave``: Heatwaves where **both** Tmax and Tmin exceed thresholds.

.. code-block:: python

   from wass2s import ETCCDIHeatWaveIndices

   # Compute Warm Spell Duration Index (WSDI)
   # Definition: Days in spells of at least 6 days where Tmax > 90th percentile
   wsdi_calc = ETCCDIHeatWaveIndices.wsdi(base_period=slice("1981", "2010"))
   
   wsdi_da = wsdi_calc.compute_xarray(tmax_da)

-------------------------------------------------------------------------------

4. ETCCDI Precipitation Extremes
================================

**Class**: ``WAS_PrecipIndices``

Computes indices based on extreme rainfall percentiles (e.g., total rainfall from days > 95th percentile).

Supported Indices
-----------------
* **R95p**: Very wet days (Rainfall > 95th percentile).
* **R99p**: Extremely wet days (Rainfall > 99th percentile).

**Example: Computing R95p**

.. code-block:: python

   from wass2s import WAS_PrecipIndices

   # 1. Initialize
   # Percentiles are calculated only using "Wet Days" (>= 1mm) in the base period
   r95p_calc = WAS_PrecipIndices(
       base_period=slice("1981", "2010"),
       percentile=95,
       wet_day_threshold=1.0
   )

   # 2. Compute
   r95p_da = r95p_calc.compute_xarray(daily_rain)

-------------------------------------------------------------------------------

Input Data Formats
==================

Gridded Data (xarray)
---------------------
* **Dimensions**: Must include ``time`` (or ``T``), ``lat`` (or ``Y``), and ``lon`` (or ``X``).
* **Time**: Must be standard datetime objects.

In-Situ Data (CDT Format)
-------------------------
Used for ``compute_insitu`` methods.

.. code-block:: text

    ID           STATION_A  STATION_B
    LON          2.5        3.0
    LAT          10.0       10.5
    ELEV         200        250
    19810101     0.0        0.0
    19810102     10.5       0.0
    ...

Output Data (CPT Format)
------------------------
Most ``compute_insitu`` methods return data in CPT format, ready for Climate Prediction Tools.

.. code-block:: text

    STATION  STATION_A  STATION_B
    LAT      10.0       10.5
    LON      2.5        3.0
    1981     55.2       60.1
    1982     40.0       45.5
    ...

=========================================
Merging Gridded Data with Observations
=========================================

The ``WAS_Merging`` module provides advanced methods for blending coarse gridded data (such as satellite estimates or reanalysis) with high-quality local station observations. 

This process, often called "Data Merging" or "Bias Adjustment," uses spatial interpolation techniques (Kriging) and Machine Learning (Linear Regression, Neural Networks) to correct the gridded field towards the station values while preserving spatial patterns.

.. code-block:: python

   import pandas as pd
   import xarray as xr
   from wass2s import WAS_Merging

Prerequisites & Data Formats
============================

The merging class requires two specific inputs:

1.  **Station Data (Pandas DataFrame)**: Must be in **CPT Format** (Climate Prediction Tool).
    
    * **Rows 1-2**: Metadata (Latitude, Longitude).
    * **Row 3+**: Data (Years in first column, Station values in subsequent columns).
    
    *Example CPT DataFrame structure:*

    ======= ===== ===== =====
    STATION ST_A  ST_B  ST_C
    LAT     12.5  12.8  13.0
    LON     -1.5  -1.2  -1.0
    1981    100.5 90.2  110.0
    1982    105.2 95.1  112.5
    ======= ===== ===== =====

2.  **Gridded Data (xarray DataArray)**:
    
    * Must have dimensions ``(T, Y, X)``.
    * Coordinates must match the station data's spatial extent.

Class Initialization
====================

.. class:: WAS_Merging(df, da, date_month_day="08-01")

   Initializes the merging object.

   :param df: ``pd.DataFrame`` containing station data in CPT format.
   :param da: ``xr.DataArray`` or ``xr.Dataset`` containing the gridded estimate.
   :param date_month_day: ``str`` (Format "MM-DD"). Used to assign a specific day/month to the yearly CPT data for time alignment.

   .. code-block:: python
   
      # Initialize
      merger = WAS_Merging(
          df=station_df, 
          da=satellite_da, 
          date_month_day="08-01" # Aligns '1981' in CPT to '1981-08-01'
      )

Merging Methods
===============

1. Simple Bias Adjustment (Residual Kriging)
--------------------------------------------

Calculates the residuals (:math:`Station - Grid`) at station locations, interpolates these residuals onto the full grid using Ordinary Kriging, and adds them back to the original grid.

.. py:method:: simple_bias_adjustment(missing_value=-999.0, do_cross_validation=False)

   :param missing_value: Value representing NaNs in the input DataFrame.
   :param do_cross_validation: If ``True``, performs Leave-One-Out (LOO) cross-validation to estimate RMSE.
   :return: (``xr.DataArray`` Corrected Grid, ``pd.DataFrame`` CV Results)

   .. code-block:: python

      # Apply Simple Bias Adjustment
      corrected_da, cv_results = merger.simple_bias_adjustment(
          do_cross_validation=True
      )

2. Regression Kriging
---------------------

Combines a global trend estimation with local residual correction.
1. Fits a **Linear Regression** between the Grid (predictor) and Stations (predictand).
2. Kriges the residuals from this regression.
3. Final = Linear Prediction + Kriged Residuals.

.. py:method:: regression_kriging(missing_value=-999.0, do_cross_validation=False)

   :return: (``xr.DataArray`` Corrected Grid, ``pd.DataFrame`` CV Results)

   .. code-block:: python

      # Apply Regression Kriging (Better for strong linear relationships)
      corrected_da, _ = merger.regression_kriging()

3. Neural Network Kriging
-------------------------

Uses a Multi-Layer Perceptron (MLP) to capture non-linear relationships between the grid and stations, followed by Kriging of the residuals. 

* **Features**: Automatically tunes hyperparameters (hidden layers, activation) using ``GridSearchCV``.
* **Use Case**: Complex topography or non-linear biases.

.. py:method:: neural_network_kriging(missing_value=-999.0, do_cross_validation=False)

   :return: (``xr.DataArray`` Corrected Grid, ``pd.DataFrame`` CV Results)

   .. code-block:: python

      # Apply Neural Network Kriging (Computationally intensive)
      corrected_da, _ = merger.neural_network_kriging()

4. Multiplicative Bias
----------------------

Calculates the ratio (:math:`Station / Grid`) instead of the difference. Interpolates the ratio field and multiplies the original grid.
* **Use Case**: Strictly positive variables like precipitation where subtraction might yield negative values.

.. py:method:: multiplicative_bias(missing_value=-999.0, do_cross_validation=False)

   :return: (``xr.DataArray`` Corrected Grid, ``pd.DataFrame`` CV Results)

Visualization
=============

.. py:method:: plot_merging_comparaison(df_Obs, da_estimated, da_corrected, missing_value=-999.0)

   Generates a 2-panel scatter plot comparing:
   1. Observation vs. Original Estimate
   2. Observation vs. Corrected Result

   Includes a 1:1 line to visually assess bias reduction.

   .. code-block:: python

      merger.plot_merging_comparaison(
          df_Obs=station_df, 
          da_estimated=satellite_da, 
          da_corrected=corrected_da
      )

Usage Example
===============

This example demonstrates creating dummy data and running the ``regression_kriging`` method.

.. code-block:: python
  import pandas as pd
  import numpy as np
  import xarray as xr
  from wass2s import WAS_Merging

  # --- 1. Create Dummy Station Data (CPT Format) ---
  # Add St_D and St_E to satisfy n_splits=5
  data = {
       'STATION': ['LAT', 'LON', '1981', '1982', '1983'],
       'St_A': [10.0, 2.0, 150, 160, 140],
       'St_B': [10.5, 2.5, 180, 175, 190],
       'St_C': [11.0, 3.0, 200, 210, 205],
       'St_D': [10.2, 2.8, 160, 170, 150], # NEW
       'St_E': [10.8, 2.2, 190, 180, 195]  # NEW
   }
  df_cpt = pd.DataFrame(data)

  # --- 2. Create Dummy Gridded Data (xarray) ---
  # Create a grid covering the station area
  lons = np.linspace(1.5, 3.5, 10)
  lats = np.linspace(9.5, 11.5, 10)
  times = pd.to_datetime(['1981-08-01', '1982-08-01', '1983-08-01'])
  
  # Random data with coords
  grid_values = np.random.randint(100, 250, size=(3, 10, 10)).astype(float)
  da_grid = xr.DataArray(
      grid_values, 
      coords={'T': times, 'Y': lats, 'X': lons}, 
      dims=('T', 'Y', 'X')
  )

  # --- 3. Initialize Merger ---
  # Note: date_month_day must match the month/day in da_grid for alignment
  merger = WAS_Merging(df=df_cpt, da=da_grid, date_month_day="08-01")

  # --- 4. Run Simple Bias Adjustement ---
  corrected_da, cv_stats = merger.simple_bias_adjustment(
      missing_value=-999.0,
      do_cross_validation=True
  )

  print("Correction Complete.")
  print(corrected_da)
  
  if cv_stats is not None:
      print("\nCross Validation RMSE:")
      print(cv_stats)

  # --- 5. Visualize ---
  merger.plot_merging_comparaison(df_cpt, da_grid, corrected_da)

=======================
Bias Correction Modules
=======================

This module provides advanced statistical bias correction methods for climate data. It is divided into two specialized classes depending on the nature of the climate variable being processed:

1.  **WAS_Qmap**: Designed for **Precipitation**. It handles "wet-day" frequencies, intermittency (zeros), and extreme value corrections.
2.  **WAS_bias_correction**: Designed for **Continuous Variables** (Temperature, Wind, Humidity, Pressure). It handles mean adjustments, variance scaling, and continuous distribution mapping.

Both classes support **NumPy arrays** and **xarray.DataArrays** (preserving coordinates).

Prerequisites
=============

.. code-block:: python

   import numpy as np
   import xarray as xr
   from wass2s import WAS_Qmap, WAS_bias_correction

-------------------------------------------------------------------------------

1. Precipitation Bias Correction (WAS_Qmap)
===========================================

The ``WAS_Qmap`` class implements Quantile Mapping (QM) techniques adapted from the R ``qmap`` package. It is specifically robust for rainfall data where the model might drizzle too often (drizzle effect) or miss extremes.

Supported Methods
-----------------

* ``QUANT``: Empirical Quantile Mapping (Non-parametric). Maps the CDF of the model to the CDF of observations.
* ``RQUANT``: Robust Quantile Mapping (Recommended). Uses local linear least squares to estimate quantile corrections, making it robust for tails/extremes.
* ``SSPLIN``: Smoothing Splines. Fits a smooth spline between quantiles.
* ``PTF``: Parametric Transformation Functions (Power, Exponential, Linear).
* ``DIST``: Distribution-based mapping (e.g., Bernoulli-Gamma for rain).

Example: Correcting Daily Rainfall
----------------------------------

This example demonstrates how to create dummy input data with explicit coordinates and apply **Robust Quantile Mapping (RQUANT)**.

**1. Prepare Input Data (xarray)**

Inputs must be 3D DataArrays with dimensions ``(T, Y, X)``.

.. code-block:: python

    import pandas as pd
    import numpy as np
    import xarray as xr
    from wass2s import WAS_Qmap

    # --- Define Explicit Coordinates ---
    # Time: 2 years of daily data
    times = pd.date_range(start="1981-01-01", end="1982-12-31", freq="D")
    # Space: 2x2 grid
    lats = [12.5, 13.0] 
    lons = [-2.0, -1.5]

    # --- Generate Synthetic Rainfall Data ---
    # Observations: Gamma distribution (more variance)
    # Model: Gamma distribution (less variance, different mean)
    np.random.seed(42)
    obs_data = np.random.gamma(shape=2, scale=2, size=(len(times), len(lats), len(lons)))
    mod_data = np.random.gamma(shape=2, scale=1.5, size=(len(times), len(lats), len(lons)))
    
    # Introduce "dry days" (zeros)
    obs_data[obs_data < 1.0] = 0
    mod_data[mod_data < 0.5] = 0

    # Create DataArrays
    obs_da = xr.DataArray(
        obs_data, 
        coords={"time": times, "Y": lats, "X": lons}, 
        dims=("time", "Y", "X"),
        name="pr_obs"
    )
    
    mod_da = xr.DataArray(
        mod_data, 
        coords={"time": times, "Y": lats, "X": lons}, 
        dims=("time", "Y", "X"),
        name="pr_model"
    )

    # Future/Forecast data (to be corrected)
    fut_da = mod_da.copy() * 1.1 # Simulating a wetter future

**2. Fit and Apply Correction**

.. code-block:: python

    # --- Step 1: Fit the Correction ---
    # Uses RQUANT method. 
    # wet_day=True ensures dry-day frequency is corrected first.
    fit_obj = WAS_Qmap.fitQmap(
        obs=obs_da, 
        mod=mod_da, 
        method='RQUANT', 
        wet_day=True,
        qstep=0.01        # Calculate correction at every 1st percentile
    )

    # --- Step 2: Apply to Data ---
    # Apply the fitted relationship to the forecast data
    corrected_da = WAS_Qmap.doQmap(fut_da, fit_obj)

    print("Original Max:", fut_da.max().values)
    print("Corrected Max:", corrected_da.max().values)

**3. Evaluate Performance**

The module includes built-in tools to validate the correction (Obs vs Corrected Hist).

.. code-block:: python

    # Compare Historical Model vs Observation vs Corrected
    # Returns datasets containing dry fraction diffs and extreme quantiles
    ds_dry_wet, ds_extreme = WAS_Qmap.evaluate_bias_correction(
        obs=obs_da,
        mod=mod_da,
        corrected=WAS_Qmap.doQmap(mod_da, fit_obj), # Correcting the historical period for validation
        wet_threshold=1.0, 
        extreme_quantiles=[0.95, 0.99]
    )
    
    # Plot spatial maps of dry/wet fractions
    WAS_Qmap.plot_fraction_group(ds_dry_wet, group_prefix='dry_fraction_')
    
    # Plot spatial maps of extreme quantiles (95th, 99th)
    WAS_Qmap.plot_extreme_quantiles_group(ds_extreme)

-------------------------------------------------------------------------------

2. Continuous Bias Correction (WAS_bias_correction)
===================================================

The ``WAS_bias_correction`` class is optimized for variables like **Temperature (TAS, TASMAX, TASMIN)** or **Wind Speed**. It assumes the data is continuous and does not require complex "dry day" handling.

Supported Methods
-----------------

* ``MEAN``: Simple additive bias correction (adjusts the mean only).
* ``VARSCALE``: Variance Scaling. Adjusts both the mean and the standard deviation (useful for temperature).
* ``QUANT``: Empirical Quantile Mapping (Non-parametric).
* ``NORM``: Fits a Normal distribution to both Obs and Model and maps the CDFs.
* ``DIST``: Fits a specific distribution (e.g., Weibull for wind, Gamma for skewed temp).

Example: Correcting Temperature
-------------------------------

**1. Prepare Input Data (Temperature)**

.. code-block:: python

    from wass2s import WAS_bias_correction

    # Create dummy Temperature data (Normal distribution)
    # Obs mean = 28C, Model mean = 26C (Model is cold biased)
    obs_temp = np.random.normal(loc=28, scale=2, size=(730, 2, 2))
    mod_temp = np.random.normal(loc=26, scale=1.5, size=(730, 2, 2))

    # Xarray wrapper
    obs_t_da = xr.DataArray(obs_temp, dims=("time", "Y", "X"), coords={"time": times, "Y": lats, "X": lons})
    mod_t_da = xr.DataArray(mod_temp, dims=("time", "Y", "X"), coords={"time": times, "Y": lats, "X": lons})

**2. Fit and Apply Variance Scaling**

Variance scaling is often preferred for temperature as it preserves the shape of the distribution while correcting the first two moments (mean and variance).

.. code-block:: python

    # --- Step 1: Fit ---
    # Fits Mean and Standard Deviation for Obs and Model
    fit_temp = WAS_bias_correction.fitBC(
        obs=obs_t_da, 
        mod=mod_t_da, 
        method='VARSCALE'
    )

    # --- Step 2: Apply ---
    corrected_temp = WAS_bias_correction.doBC(mod_t_da, fit_temp)

    # Verify Mean Correction
    print(f"Obs Mean: {obs_t_da.mean().values:.2f}")
    print(f"Model Mean: {mod_t_da.mean().values:.2f}")
    print(f"Corrected Mean: {corrected_temp.mean().values:.2f}")

Example: Correcting Wind Speed (Weibull)
----------------------------------------

For variables like wind speed which are positively skewed, use the ``DIST`` method with a Weibull distribution.

.. code-block:: python

    # --- Step 1: Fit Weibull ---
    fit_wind = WAS_bias_correction.fitBC(
        obs=obs_wind_da, 
        mod=mod_wind_da, 
        method='DIST',
        distr='weibull' # Options: 'gamma', 'lognormal', 'normal'
    )

    # --- Step 2: Apply ---
    corrected_wind = WAS_bias_correction.doBC(mod_wind_da, fit_wind)


==========================================
Data Transformation & Skewness Analysis
==========================================

The ``WAS_TransformData`` module handles the statistical preprocessing of geospatial time-series data. Climate data (especially precipitation) is often non-normal and highly skewed. This module provides tools to:

1.  **Analyze Skewness**: Detect and map positive/negative skewness.
2.  **Transform Data**: Apply and invert transformations (Box-Cox, Yeo-Johnson, Log) to normalize data.
3.  **Fit Distributions**: spatially fit statistical distributions (e.g., Gamma, Weibull) using clustering techniques.



Prerequisites & Input Data
==========================

The module requires an ``xarray.DataArray`` with dimensions ``(T, Y, X)``.

**Example: Creating Synthetic Input Data**

.. code-block:: python

   import numpy as np
   import pandas as pd
   import xarray as xr
   from wass2s import WAS_TransformData

   # 1. Define Coordinates
   # Time: Daily data for 2 years
   times = pd.date_range(start="2000-01-01", end="2001-12-31", freq="D")
   # Space: A 10x10 grid (approx 1 degree resolution)
   lats = np.linspace(5.0, 15.0, 10)
   lons = np.linspace(-5.0, 5.0, 10)

   # 2. Generate Skewed Synthetic Data (Gamma Distribution)
   # Shape=2.0, Scale=2.0 generates positively skewed precipitation-like data
   np.random.seed(42)
   data_values = np.random.gamma(shape=2.0, scale=2.0, size=(len(times), len(lats), len(lons)))

   # 3. Create Xarray DataArray (Must have T, Y, X dims)
   rainfall_da = xr.DataArray(
       data_values,
       coords={"T": times, "Y": lats, "X": lons},
       dims=("T", "Y", "X"),
       name="precip",
       attrs={"units": "mm"}
   )

   # 4. Initialize the Transformer
   transformer = WAS_TransformData(data=rainfall_da, n_clusters=5)

-------------------------------------------------------------------------------

1. Skewness Detection and Handling
==================================

Before applying statistical models, it is often necessary to normalize the data.

**Step 1: Detect Skewness**

Calculates Fisher-Pearson skewness for every grid cell along the time dimension.

.. code-block:: python

   # Returns a dataset with skewness values and categorical classes
   # Classes: 'symmetric', 'moderate_positive', 'high_positive', etc.
   skew_ds, skew_summary = transformer.detect_skewness()
   
   print("Skewness Counts:", skew_summary['class_counts'])

**Step 2: Handle Skewness**

Based on the detected class and data properties (e.g., presence of zeros), this method recommends specific transformations.

.. code-block:: python

   # Returns dataset with 'recommended_methods' per pixel
   handle_ds, recommendations = transformer.handle_skewness()

**Step 3: Apply Transformation**

Applies the recommended method (or a specific forced method) to the data.



.. code-block:: python

   # Option A: Auto-apply recommendations
   transformed_da = transformer.apply_transformation()

   # Option B: Force a specific method (e.g., 'box_cox' or 'yeo_johnson')
   # Box-Cox requires strictly positive data; Yeo-Johnson handles zeros.
   transformed_da_forced = transformer.apply_transformation(method='yeo_johnson')

**Step 4: Inverse Transformation**

After analysis/modeling, you can revert the data to its original scale.

.. code-block:: python

   original_scale_da = transformer.inverse_transform()

-------------------------------------------------------------------------------

2. Distribution Fitting (Clustering Approach)
=============================================

Fitting distributions pixel-by-pixel can be noisy and computationally expensive. This module uses **K-Means Clustering** to define homogeneous zones based on mean and standard deviation, pools the data within those zones, and fits the best distribution using **AIC (Akaike Information Criterion)**.



Supported Distributions
-----------------------
* **Continuous**: ``norm``, ``lognorm``, ``gamma``, ``weibull_min``, ``expon``, ``t``
* **Discrete**: ``poisson``, ``nbinom``

**Usage Example**

.. code-block:: python

   # Fit distributions using the transformed data
   # n_clusters was defined in __init__ (default=5 or 1000 depending on resolution)
   best_code, best_shape, best_loc, best_scale, cluster_map = \
       transformer.fit_best_distribution_grid_onlycluster(use_transformed=True)

   # The outputs are xarray DataArrays of the same shape as (Y, X)
   # best_code: Integer mapping to the distribution name (see transformer.distribution_map)
   # best_shape/loc/scale: The fitted parameters for that pixel's assigned distribution

**Fitting Modes**

The method ``fit_best_distribution_grid_two_options`` allows choosing the strategy:

.. code-block:: python

   # Mode 'cluster': Fits distributions to homogeneous zones (Faster, robust)
   res_cluster = transformer.fit_best_distribution_grid_two_options(mode='cluster')

   # Mode 'grid': Fits distributions independently to every pixel (Slower, captures local detail)
   res_grid = transformer.fit_best_distribution_grid_two_options(mode='grid')

-------------------------------------------------------------------------------

3. Visualization
================

The module includes a specific plotter for categorical maps (like distribution types or skewness classes) using Cartopy.

.. code-block:: python

   # 1. Define a color map for your distributions
   # Codes: norm=1, lognorm=2, expon=3, gamma=4, weibull=5, etc.
   dist_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

   # 2. Plot the best distribution map
   transformer.plot_best_fit_map(
       data_array=best_code,
       map_dict=transformer.distribution_map,
       output_file='distribution_map.png',
       title='Best Fitting Distribution per Zone',
       colors=dist_colors,
       show_plot=True
   )

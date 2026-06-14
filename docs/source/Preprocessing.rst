Preprocessing
=============

Before building forecasting models, raw data must go through several
preprocessing stages:

1. :ref:`compute-predictand` — compute agroclimatic indices (onset,
   cessation, dry/wet spells, ETCCDI extremes) from daily data.
2. :ref:`merge-obs` — blend coarse gridded estimates with station observations.
3. :ref:`bias-correction` — correct systematic biases in GCM or reanalysis output.
4. :ref:`data-transform` — handle skewed distributions and fit parametric
   distributions for probabilistic post-processing.

Input data format conventions
------------------------------

.. _cdt-format:

**CDT — daily station data**

The Climate Data Tools (CDT) format is used as input to all ``compute_insitu``
methods:

.. code-block:: text

   ID           ALLADA     APLAHOUE
   LON          2.133333   1.666667
   LAT          6.65       6.916667
   DAILY/ELEV   92.0       153.0
   19810101     0.0        0.0
   19810102     0.0        0.0
   ...

.. _cpt-format:

**CPT — seasonal station data**

The Climate Prediction Tools (CPT) format is used when passing station data
to the merging classes or when reading seasonal aggregates:

.. code-block:: text

   STATION   ABEO    ABUJ    ADEK
   LAT        7.2     7.6     9.0
   LON        3.3     5.2     7.2
   1991      514.9   715.1   934.3
   1992      503.6   736.4   714.6
   ...

Gridded data must be ``xarray.DataArray`` objects with dimensions
``(T, Y, X)`` and standard datetime coordinates on the ``T`` axis.

-------------------------------------------------------------------------------

.. _compute-predictand:

Computing Agroclimatic Predictands
------------------------------------

The ``was_compute_predictand`` module computes climate indices directly from
daily gridded or station data. It supports parallel execution on large grids
via Dask.

Onset of the rainy season
~~~~~~~~~~~~~~~~~~~~~~~~~

**Class**: ``WAS_compute_onset``

Detects the first date of reliable season onset using the AGRHYMET / Sivakumar
(1988) methodology:

#. Search starts from a zone-specific calendar date.
#. A cumulative rainfall threshold must be met within a short window (e.g. 20 mm
   in 3 consecutive days).
#. No prolonged dry spell is allowed in the following weeks.

Default agro-ecological zones and criteria:

+------------------------+-------------+------------+-------------+
| Zone                   | Search start| Cumulative | Max dry days|
+========================+=============+============+=============+
| Sahel (0–100 mm/yr)    | 01 Jun      | 10 mm      | 25          |
+------------------------+-------------+------------+-------------+
| Sahel (100–200 mm/yr)  | 15 May      | 15 mm      | 25          |
+------------------------+-------------+------------+-------------+
| Sahel (200–400 mm/yr)  | 01 May      | 15 mm      | 20          |
+------------------------+-------------+------------+-------------+
| Sahel (400–600 mm/yr)  | 15 Mar      | 20 mm      | 20          |
+------------------------+-------------+------------+-------------+
| Soudanian              | 15 Mar      | 20 mm      | 10          |
+------------------------+-------------+------------+-------------+
| Gulf of Guinea         | 01 Feb      | 20 mm      | 10          |
+------------------------+-------------+------------+-------------+

Zone boundaries are automatically assigned based on mean annual rainfall and
latitude; custom criteria can be supplied via ``user_criteria``.

.. code-block:: python

   from wass2s import WAS_compute_onset

   # Use default zone criteria
   onset_calc = WAS_compute_onset()

   # Gridded computation (daily_rain is xarray DataArray T, Y, X)
   onset_da = onset_calc.compute(daily_data=daily_rain, nb_cores=4)

   # Station computation (CDT format)
   onset_cpt = onset_calc.compute_insitu(daily_df=daily_df_cdt)

To override the defaults, pass a ``user_criteria`` dictionary:

.. code-block:: python

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
   onset_calc = WAS_compute_onset(user_criteria=criteria)

Cessation of the rainy season
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Class**: ``WAS_compute_cessation``

Detects the end of the season using a soil water-balance model. The season
ends when accumulated evapotranspiration demand exceeds accumulated
post-onset rainfall plus the maximum soil water retention capacity.

.. code-block:: python

   from wass2s import WAS_compute_cessation

   criteria_cess = {
       0: {
           "zone_name": "Sahel",
           "date_dry_soil": "01-01",
           "start_search": "09-01",
           "ETP": 5.0,           # mm/day
           "Cap_ret_maxi": 70,   # mm
           "end_search": "10-30"
       }
   }

   cess_calc = WAS_compute_cessation(user_criteria=criteria_cess)
   cessation_da = cess_calc.compute(daily_data=daily_rain, nb_cores=4)

Dry and wet spell analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Classes**: ``WAS_compute_onset_dry_spell``, ``WAS_count_dry_spells``,
``WAS_count_wet_spells``, ``WAS_count_rainy_days``

These classes characterise the intra-seasonal distribution of rainfall
between onset and cessation.

.. code-block:: python

   from wass2s import (
       WAS_compute_onset_dry_spell,
       WAS_count_dry_spells,
       WAS_count_rainy_days
   )

   # Longest dry spell in the 30 days after onset (critical for seedling survival)
   dry_spell_calc = WAS_compute_onset_dry_spell()
   max_dry_spell_da = dry_spell_calc.compute(daily_data=daily_rain, nb_cores=4)

   # Total number of rainy days (> 1 mm) between onset and cessation
   rain_days_calc = WAS_count_rainy_days()
   nb_rainy_da = rain_days_calc.compute(
       daily_data=daily_rain,
       onset_date=onset_da,
       cessation_date=cessation_da,
       rain_threshold=1.0,
       nb_cores=4
   )

   # Number of dry spells ≥ 7 days between onset and cessation
   ds_count_calc = WAS_count_dry_spells()
   n_dry_spells_da = ds_count_calc.compute(
       daily_data=daily_rain,
       onset_date=onset_da,
       cessation_date=cessation_da,
       d_len=7,
       nb_cores=4
   )

ETCCDI temperature extremes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Classes**: ``ETCCDITempIndices``, ``ETCCDIHeatWaveIndices``

These factory classes implement standard ETCCDI indices using a 5-day
centred bootstrapping window for percentile estimation.

.. code-block:: python

   from wass2s import ETCCDITempIndices, ETCCDIHeatWaveIndices

   # TX90p — percentage of days where Tmax > 90th percentile
   tx90p_calc = ETCCDITempIndices.hot_days(
       base_period=slice("1981", "2010"),
       season=[6, 7, 8]
   )
   tx90p_da = tx90p_calc.compute_xarray(tmax_da, parallel=True)

   # WSDI — Warm Spell Duration Index
   wsdi_calc = ETCCDIHeatWaveIndices.wsdi(base_period=slice("1981", "2010"))
   wsdi_da = wsdi_calc.compute_xarray(tmax_da)

ETCCDI precipitation extremes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Class**: ``WAS_PrecipIndices``

Computes extreme precipitation indices (R95p, R99p) based on wet-day
percentiles from a reference period.

.. code-block:: python

   from wass2s import WAS_PrecipIndices

   r95p_calc = WAS_PrecipIndices(
       base_period=slice("1981", "2010"),
       percentile=95,
       wet_day_threshold=1.0
   )
   r95p_da = r95p_calc.compute_xarray(daily_rain)

-------------------------------------------------------------------------------

.. _merge-obs:

Merging Gridded Data with Station Observations
------------------------------------------------

The ``WAS_Merging`` class blends a coarse gridded estimate (satellite product,
reanalysis, or model output) with sparse station observations to produce a
spatially complete, station-corrected field.

The class is initialised with a station DataFrame in :ref:`CPT format <cpt-format>`
and a gridded ``xarray.DataArray``:

.. code-block:: python

   from wass2s import WAS_Merging

   merger = WAS_Merging(
       df=station_df,          # CPT-format DataFrame
       da=satellite_da,        # (T, Y, X) DataArray
       date_month_day="08-01"  # Aligns CPT year labels to a specific day
   )

Available merging methods
~~~~~~~~~~~~~~~~~~~~~~~~~

**Simple Bias Adjustment** (Residual Kriging)

Computes ``Station − Grid`` residuals at station locations, interpolates them
over the full grid with Ordinary Kriging, and adds them back to the original
field.

.. code-block:: python

   corrected_da, cv_stats = merger.simple_bias_adjustment(
       missing_value=-999.0,
       do_cross_validation=True
   )

**Regression Kriging**

Fits a linear regression between the grid and the stations, then Kriges the
regression residuals. Preferred when there is a strong linear relationship
between the gridded product and station observations.

.. code-block:: python

   corrected_da, _ = merger.regression_kriging()

**Neural Network Kriging**

Replaces the linear regression with a Multi-Layer Perceptron to capture
non-linear biases, followed by Kriging of the residuals. More powerful for
complex terrain but computationally heavier.

.. code-block:: python

   corrected_da, _ = merger.neural_network_kriging()

**Multiplicative Bias**

Computes the ratio ``Station / Grid`` instead of the difference, interpolates
it spatially, and multiplies the original grid. Appropriate for strictly
positive variables like precipitation where subtraction can yield unphysical
negative values.

.. code-block:: python

   corrected_da, _ = merger.multiplicative_bias()

Visualising the correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   merger.plot_merging_comparaison(
       df_Obs=station_df,
       da_estimated=satellite_da,
       da_corrected=corrected_da
   )

This generates a two-panel scatter plot (observation vs. original estimate and
observation vs. corrected result) with a 1:1 reference line.

-------------------------------------------------------------------------------

.. _bias-correction:

Bias Correction
---------------

Two classes handle bias correction depending on the nature of the variable.

Precipitation — ``WAS_Qmap``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``WAS_Qmap`` implements quantile-mapping methods adapted from the R ``qmap``
package. They explicitly handle wet-day frequency (the drizzle effect common
in GCMs) before correcting the quantile structure.

Supported methods:

* ``QUANT`` — empirical quantile mapping.
* ``RQUANT`` — robust quantile mapping using local linear fits (recommended
  for precipitation tails).
* ``SSPLIN`` — PCHIP smoothing-spline quantile mapping.
* ``PTF`` — parametric transformation function (power, exponential, linear).
* ``DIST`` — full distribution mapping (Bernoulli-Gamma for rain, etc.).

.. code-block:: python

   from wass2s import WAS_Qmap

   # Fit correction on the historical period
   fit_obj = WAS_Qmap.fitQmap(
       obs=obs_da,
       mod=mod_da,
       method="RQUANT",
       wet_day=True,    # Correct dry-day frequency first
       qstep=0.01
   )

   # Apply to forecast data
   corrected_da = WAS_Qmap.doQmap(forecast_da, fit_obj)

   # Evaluate the correction (compare dry fractions and extreme quantiles)
   ds_dry_wet, ds_extreme = WAS_Qmap.evaluate_bias_correction(
       obs=obs_da,
       mod=mod_da,
       corrected=WAS_Qmap.doQmap(mod_da, fit_obj),
       wet_threshold=1.0,
       extreme_quantiles=[0.95, 0.99]
   )
   WAS_Qmap.plot_extreme_quantiles_group(ds_extreme)

Continuous variables — ``WAS_bias_correction``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``WAS_bias_correction`` targets temperature, wind speed, and similar
continuously distributed variables.

Supported methods: ``MEAN`` (additive), ``VARSCALE`` (mean + variance),
``QUANT``, ``NORM``, ``DIST``.

.. code-block:: python

   from wass2s import WAS_bias_correction

   # Temperature: variance scaling (corrects mean and spread)
   fit_temp = WAS_bias_correction.fitBC(
       obs=obs_temp_da,
       mod=mod_temp_da,
       method="VARSCALE"
   )
   corrected_temp = WAS_bias_correction.doBC(forecast_temp_da, fit_temp)

   # Wind speed: distribution mapping with Weibull
   fit_wind = WAS_bias_correction.fitBC(
       obs=obs_wind_da,
       mod=mod_wind_da,
       method="DIST",
       distr="weibull"
   )
   corrected_wind = WAS_bias_correction.doBC(forecast_wind_da, fit_wind)

-------------------------------------------------------------------------------

.. _data-transform:

Data Transformation and Distribution Fitting
---------------------------------------------

The ``WAS_TransformData`` class handles the statistical preprocessing of
geospatial time series, which is often necessary before applying parametric
probabilistic post-processing to skewed rainfall data.

Initialise with an ``(T, Y, X)`` DataArray:

.. code-block:: python

   from wass2s import WAS_TransformData

   transformer = WAS_TransformData(data=rainfall_da, n_clusters=5)

Skewness analysis
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Detect skewness class per grid cell
   skew_ds, skew_summary = transformer.detect_skewness()
   print("Skewness counts:", skew_summary["class_counts"])

   # Get transformation recommendations
   handle_ds, recommendations = transformer.handle_skewness()

Applying and inverting transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Auto-apply recommended transformation per grid cell
   transformed_da = transformer.apply_transformation()

   # Or force a specific method
   transformed_da = transformer.apply_transformation(method="yeo_johnson")

   # Invert after modelling to recover original units
   original_scale_da = transformer.inverse_transform()

Distribution fitting
~~~~~~~~~~~~~~~~~~~~~

Fitting distributions pixel-by-pixel can be noisy. ``WAS_TransformData``
uses K-Means clustering to pool data within homogeneous zones before fitting.
The best distribution per zone is selected by AIC.

Supported distributions: ``norm``, ``lognorm``, ``gamma``, ``weibull_min``,
``expon``, ``t``, ``poisson``, ``nbinom``.

.. code-block:: python

   # Cluster-based fitting (faster, recommended for large domains)
   best_code, best_shape, best_loc, best_scale, cluster_map = \
       transformer.fit_best_distribution(use_transformed=True, mode="cluster")

   # Pixel-by-pixel fitting (slower, captures local detail)
   best_code, best_shape, best_loc, best_scale, cluster_map = \
       transformer.fit_best_distribution(use_transformed=True, mode="grid")

   # Visualise the winning distribution per grid cell
   transformer.plot_best_fit_map(
       data_array=best_code,
       map_dict=transformer.distribution_map,
       output_file="distribution_map.png",
       title="Best Fitting Distribution",
       show_plot=True
   )

The outputs ``best_code``, ``best_shape``, ``best_loc``, and ``best_scale``
are ``(Y, X)`` DataArrays that can be passed directly to any model's
``compute_prob`` and ``forecast`` methods when using ``dist_method="bestfit"``.

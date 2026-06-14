Quick Start
===========

This page walks through a minimal end-to-end forecast in five steps: download
data, prepare predictors and predictand, cross-validate a model, verify the
hindcast, and produce an operational forecast map.

For more detailed explanations of each step see the :doc:`User guide <Download>`.

Prerequisites
-------------

Install wass2s and activate your environment (see :doc:`Installation`) then
open a Jupyter notebook or Python script and import the package:

.. code-block:: python

   from wass2s import *
   import numpy as np

Step 1 — Download data
----------------------

.. code-block:: python

   downloader = WAS_Download()

   # ERA5 global SST — predictor window (JFM)
   downloader.WAS_Download_Reanalysis(
       dir_to_save="./data/era5/",
       center_variable=["ERA5.SST"],
       year_start=1991, year_end=2025,
       area=[45, -180, -45, 180],   # [N, W, S, E]
       seas=["01", "02", "03"],
       force_download=False
   )

   # AgERA5 precipitation — predictand (JJAS)
   downloader.WAS_Download_AgroIndicators(
       dir_to_save="./data/obs/",
       variables=["AGRO.PRCP"],
       year_start=1991, year_end=2024,
       area=[30, -25, 0, 30],
       seas=["06", "07", "08", "09"],
       force_download=False
   )

Step 2 — Prepare predictors and predictand
------------------------------------------

.. code-block:: python

   clim_year_start, clim_year_end = 1991, 2020
   year_start, year_end = 1991, 2024

   # Observed predictand (T, Y, X)
   predictand = prepare_predictand(
       "./data/obs/", ["AGRO.PRCP"],
       year_start, year_end,
       seas=["06", "07", "08", "09"],
       ds=False
   )

   # SST indices as predictors
   sst_index_names = ["NINO34", "TNA", "TSA", "DMI"]
   predictors = compute_sst_indices(
       "./data/era5/", sst_index_names,
       "ERA5.SST", year_start, year_end,
       seas=["01", "02", "03"],
       clim_year_start=clim_year_start,
       clim_year_end=clim_year_end
   )
   predictors = (
       predictors
       .to_array()
       .rename({"variable": "features"})
       .transpose("T", "features")
   )

Step 3 — Cross-validate a Ridge regression model
-------------------------------------------------

.. code-block:: python

   # Optimise the regularisation parameter
   ridge = WAS_Ridge_Model(
       n_clusters=6,
       alpha_range=np.logspace(-4, 0.1, 20),
       nb_cores=4
   )
   alpha, clusters = ridge.compute_hyperparameters(
       predictand, predictors.isel(T=slice(None, -1)),
       clim_year_start, clim_year_end
   )

   # Leave-one-out cross-validation
   cv = WAS_Cross_Validator(
       n_splits=len(predictand.get_index("T")),
       nb_omit=2
   )
   hindcast_det, hindcast_prob = cv.cross_validate(
       ridge, predictand,
       predictors.isel(T=slice(None, -1)),
       clim_year_start, clim_year_end,
       alpha=alpha
   )

Step 4 — Verify the hindcast
-----------------------------

.. code-block:: python

   verifier = WAS_Verification()

   pearson = verifier.compute_deterministic_score(
       verifier.pearson_corr, predictand, hindcast_det
   )
   groc = verifier.compute_probabilistic_score(
       verifier.calculate_groc, predictand, hindcast_prob,
       clim_year_start=clim_year_start,
       clim_year_end=clim_year_end
   )

   verifier.plot_model_score(pearson, "Pearson",
       dir_save_score="./scores/", figure_name="pearson_ridge")
   verifier.plot_model_score(groc, "GROC",
       dir_save_score="./scores/", figure_name="groc_ridge")

Step 5 — Operational forecast for the target year
--------------------------------------------------

.. code-block:: python

   # The last time step of predictors is the real-time predictor
   forecast_det, forecast_prob = ridge.forecast(
       predictand, clim_year_start, clim_year_end,
       predictors.isel(T=slice(None, -1)),   # hindcast predictors
       hindcast_det,
       predictors.isel(T=[-1]),              # real-time predictor
       alpha=alpha
   )

   # Save and plot
   plot_prob_forecasts(
       dir_to_save="./forecasts/",
       forecast_prob=forecast_prob,
       model_name="Ridge",
       title="JJAS 2025 Sahel Rainfall Forecast — Ridge Regression"
   )

.. tip::
   Replace ``WAS_Ridge_Model`` with any other model class — ``WAS_PCR``,
   ``WAS_CCA``, ``WAS_SVR``, ``WAS_MLP``, ``WAS_Analog``, or an MME class —
   without changing the cross-validation or forecasting calls.

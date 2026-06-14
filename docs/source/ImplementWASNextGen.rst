Implementing WAS-NextGen Approaches
=====================================

The WAS-NextGen (West Africa Seasonal Forecasting Next Generation) framework
defines the methodological standards for operational seasonal forecasting
across the region, aligned with WMO guidelines for objective, reproducible,
and scientifically rigorous seasonal outlooks.

wass2s was designed from the ground up to implement this framework end-to-end.
This page describes the recommended workflow for producing a WAS-NextGen
compliant seasonal forecast.

Overview of the WAS-NextGen pipeline
--------------------------------------

A WAS-NextGen seasonal forecast consists of five main stages:

1. **Data preparation** — download and harmonise GCM hindcasts, reanalysis,
   and observational products.
2. **Predictor construction** — compute SST indices or spatial EOF predictors.
3. **Model calibration and cross-validation** — fit and validate one or more
   statistical or ML models using leave-one-out cross-validation.
4. **Multi-model combination** — combine calibrated models into an ensemble.
5. **Verification and dissemination** — score the hindcast and produce
   tercile-probability maps for end users.

Recommended model hierarchy
-----------------------------

The WAS-NextGen guidelines recommend applying models in order of complexity,
using verification scores to select the best performers:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Category
     - Recommended classes
     - Typical use case
   * - Baseline
     - ``WAS_LinearRegression_Model``
     - SST indices as predictors
   * - Regularised linear
     - ``WAS_Ridge_Model``, ``WAS_ElasticNet_Model``
     - Correlated SST indices or PC predictors
   * - Spatial dimensionality reduction
     - ``WAS_PCR`` + any regression back-end
     - Full SST or SLP fields
   * - Non-linear
     - ``WAS_SVR``, ``WAS_MLP``, stacking ensembles
     - When residuals show non-linear structure
   * - GCM recalibration
     - ``WAS_CCA``, ``WAS_mme_NGR_Gaussian``, ``WAS_mme_ELR``
     - GCM output as predictor
   * - MME combination
     - ``WAS_mme_Weighted``, ``WAS_mme_FastBMA``, ``WAS_mme_Stacking``
     - Multi-model ensembles

Running multiple models and selecting the best
------------------------------------------------

A typical WAS-NextGen workflow runs several candidate models, verifies each
with GROC and RPSS, selects the top performers, and combines them.

.. code-block:: python

   from wass2s import *
   import numpy as np

   clim_year_start, clim_year_end = 1993, 2020

   # --- Predictors and predictand (already downloaded) ---
   predictand  = prepare_predictand(...)
   predictors  = compute_sst_indices(...)  # SST indices
   predictor_field = load_gridded_predictor(...)  # Full SST field

   # --- Candidate models ---
   models = {
       "OLS":    WAS_LinearRegression_Model(nb_cores=4),
       "Ridge":  WAS_Ridge_Model(n_clusters=6, nb_cores=4),
       "ElNet":  WAS_ElasticNet_Model(nb_cores=4),
       "PCR":    WAS_PCR(WAS_Ridge_Model(nb_cores=4), n_modes=8, detrend=True),
   }

   verifier = WAS_Verification()
   cv       = WAS_Cross_Validator(
       n_splits=len(predictand.get_index("T")), nb_omit=2
   )

   hindcasts_det  = {}
   hindcasts_prob = {}
   groc_scores    = {}

   for name, model in models.items():
       # Hyperparameter tuning if available
       if hasattr(model, "compute_hyperparameters"):
           alpha, _ = model.compute_hyperparameters(
               predictand, predictors, clim_year_start, clim_year_end
           )
           params = {"alpha": alpha}
       else:
           params = {}

       hdet, hprob = cv.cross_validate(
           model, predictand,
           predictors.isel(T=slice(None, -1)),
           clim_year_start, clim_year_end, **params
       )
       hindcasts_det[name]  = hdet
       hindcasts_prob[name] = hprob

       groc_scores[name] = verifier.compute_probabilistic_score(
           verifier.calculate_groc, predictand, hprob,
           clim_year_start=clim_year_start, clim_year_end=clim_year_end
       )

   # Select best models (top GROC over the domain average)
   best_models = get_best_models(
       center_variable="PRCP",
       scores=groc_scores,
       metric="GROC",
       top_n=3
   )
   print("Best models:", best_models)

Multi-model ensemble of the best performers
--------------------------------------------

Once the best individual models are identified, combine them with a
skill-weighted ensemble:

.. code-block:: python

   from wass2s import WAS_mme_Weighted

   # Stack selected hindcasts along the model dimension
   hdcst_stack = xr.concat(
       [hindcasts_det[m] for m in best_models],
       dim="M"
   ).assign_coords(M=best_models)

   # Skill scores for weighting
   scores_stack = xr.concat(
       [groc_scores[m] for m in best_models],
       dim="M"
   ).assign_coords(M=best_models)

   mme = WAS_mme_Weighted(equal_weighted=False, metric="GROC", threshold=0.5)
   mme_det, mme_prob_fcst = mme.compute(
       predictand, hdcst_stack, fcst_stack, scores_stack
   )

Operational forecast dissemination
------------------------------------

Once the ensemble forecast is ready, generate the tercile-probability map:

.. code-block:: python

   plot_prob_forecasts(
       dir_to_save="./forecasts/",
       forecast_prob=mme_prob_fcst,
       model_name="WAS-NextGen MME",
       title=(
           "JJAS 2025 Rainfall Forecast — West Africa\n"
           "Issued April 2025"
       )
   )

The output map shows three panels (PB, PN, PA) with the dominant category
highlighted and probability values encoded in the colour scale, following
the standard WAS-NextGen cartographic conventions.

Agroclimatic forecast products
--------------------------------

In addition to seasonal rainfall totals, wass2s can produce forecasts for
agroclimatic indices that are directly useful to national agricultural
extension services:

.. code-block:: python

   from wass2s import WAS_compute_onset, WAS_compute_cessation

   # Compute onset and cessation dates from downscaled daily data
   onset_da     = WAS_compute_onset().compute(daily_data=daily_rain, nb_cores=4)
   cessation_da = WAS_compute_cessation().compute(daily_data=daily_rain, nb_cores=4)

   # Season length (days)
   season_length = cessation_da - onset_da

These outputs can then be used as predictands in the same modelling framework,
replacing seasonal totals with any agroclimatic index computed by
``was_compute_predictand``.

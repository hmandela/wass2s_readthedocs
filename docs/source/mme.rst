Multi-Model Ensemble Methods
============================

The ``was_mme`` module combines hindcasts and forecasts from multiple GCMs
to improve seasonal predictive skill. It provides methods ranging from simple
weighted averaging to advanced machine-learning stacking and statistical
calibration, all producing tercile-probability outputs.

.. note::
   Some classes in this module are still under active development. Check the
   :doc:`API reference <api>` for the latest status.

-------------------------------------------------------------------------------

1. Data Preparation
-------------------

**Function**: ``process_datasets_for_mme``

Loads, regrids, and harmonises hindcasts and real-time forecasts from
multiple models onto the common observational grid. This is always the
first step before any MME method.

.. code-block:: python

   from wass2s import process_datasets_for_mme

   hdcst, fcst, obs, scores = process_datasets_for_mme(
       rainfall=obs_data,
       dir_to_save_model="./data/GCM/",
       best_models=["ECMWF_51", "NCEP_2", "UKMO_604"],
       year_start=1993,
       year_end=2016,
       score_metric="GROC"
   )

The returned ``hdcst`` DataArray has dimensions ``(T, M, Y, X)`` where
``M`` is the model dimension.

-------------------------------------------------------------------------------

2. Weighted Ensemble Averaging
-------------------------------

**Class**: ``WAS_mme_Weighted``

Combines model hindcasts using weights derived from a historical skill score.
Models whose score falls below a threshold receive zero weight.

.. code-block:: python

   from wass2s import WAS_mme_Weighted

   mme = WAS_mme_Weighted(
       equal_weighted=False,
       metric="GROC",
       threshold=0.5
   )
   hcst_det, fcst_det = mme.compute(obs, hdcst, fcst, scores)

   # Tercile probabilities from the ensemble mean
   hcst_prob = mme.compute_prob(
       Predictant=obs,
       clim_year_start=1993, clim_year_end=2016,
       hindcast_det=hcst_det
   )

**Class**: ``WAS_ProbWeighted``

Computes weights at the probability level (per tercile category) rather than
on the deterministic mean.

-------------------------------------------------------------------------------

3. Min et al. (2009) Probabilistic MME
----------------------------------------

**Class**: ``WAS_Min2009_ProbWeighted``

Implements the PMME method from Min et al. (2009):

1. Computes per-model tercile probabilities assuming a Gaussian or log-normal
   distribution.
2. Weights models proportionally to the square root of their ensemble size.
3. Combines the weighted probabilities.
4. Produces a chi-square combined map to flag areas of significant ensemble
   signal.

.. code-block:: python

   from wass2s import WAS_Min2009_ProbWeighted

   pmme = WAS_Min2009_ProbWeighted(ensemble_sizes={"ECMWF_51": 25, "NCEP_2": 24})
   fcst_prob = pmme.compute_combined_map(
       hindcasts=hdcst, obs=obs,
       clim_year_start=1993, clim_year_end=2016,
       new_forecasts=fcst
   )

-------------------------------------------------------------------------------

4. Machine-Learning Ensemble Methods
--------------------------------------

These methods treat the MME problem as a regression task: given GCM outputs
as predictors, learn to reproduce observed rainfall. Hyperparameters are
optimised per spatial cluster.

Stacking super-ensemble — ``WAS_mme_Stacking``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A two-layer architecture:

* **Layer 0 (base)**: Random Forest, XGBoost, HP-ELM, MLP — each generates
  out-of-fold predictions.
* **Layer 1 (meta)**: Ridge, Lasso, or ElasticNet trained on the OOF outputs.

.. code-block:: python

   from wass2s import WAS_mme_Stacking

   stacking = WAS_mme_Stacking(
       meta_learner_type="ridge",
       meta_search_method="bayesian",
       n_clusters=4
   )
   fcst_det, fcst_prob = stacking.forecast(
       Predictant=obs,
       clim_year_start=1993, clim_year_end=2016,
       hindcast_det=hdcst,
       hindcast_det_cross=hdcst_cv,
       Predictor_for_year=fcst
   )

Individual machine-learning regressors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each of these classes shares the same ``forecast`` signature as
``WAS_mme_Stacking``:

* **Random Forest** — ``WAS_mme_RF``
* **XGBoost** — ``WAS_mme_XGBoosting``
* **Multi-Layer Perceptron** — ``WAS_mme_MLP``
* **Extreme Learning Machine** — ``WAS_mme_hpELM``

.. code-block:: python

   from wass2s import WAS_mme_RF

   rf_mme = WAS_mme_RF(n_clusters=5, nb_cores=4)
   fcst_det, fcst_prob = rf_mme.forecast(
       Predictant=obs,
       clim_year_start=1993, clim_year_end=2016,
       hindcast_det=hdcst,
       hindcast_det_cross=hdcst_cv,
       Predictor_for_year=fcst
   )

-------------------------------------------------------------------------------

5. Calibration and Post-Processing
------------------------------------

Non-Homogeneous Gaussian Regression — ``WAS_mme_NGR_Gaussian``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calibrates the ensemble mean and spread by minimising the Continuous Ranked
Probability Score (CRPS). The calibrated forecast distribution is:

.. math::

   \mu = a + b \, \bar{x}_{\mathrm{ens}}, \qquad
   \sigma = \sqrt{c^2 + d^2 \, s^2_{\mathrm{ens}}}

where :math:`a, b, c, d` are fitted coefficients, :math:`\bar{x}_{\mathrm{ens}}`
is the ensemble mean, and :math:`s^2_{\mathrm{ens}}` is the ensemble variance.

.. code-block:: python

   from wass2s import WAS_mme_NGR_Gaussian

   ngr = WAS_mme_NGR_Gaussian()
   fcst_det, fcst_prob = ngr.forecast(
       Predictant=obs,
       clim_year_start=1993, clim_year_end=2016,
       hindcast_det=hdcst,
       Predictor_for_year=fcst
   )

Bayesian Model Averaging — ``WAS_mme_FastBMA``, ``WAS_mme_FullBMA``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BMA assigns posterior weights to each model's predictive distribution:

* ``WAS_mme_FastBMA`` — EM algorithm; fast convergence for small ensembles.
* ``WAS_mme_FullBMA`` — grid search over kernel bandwidths; more accurate.

.. code-block:: python

   from wass2s import WAS_mme_FastBMA

   bma = WAS_mme_FastBMA()
   fcst_det, fcst_prob = bma.forecast(
       Predictant=obs,
       clim_year_start=1993, clim_year_end=2016,
       hindcast_det=hdcst,
       forecast_det=fcst
   )

Extended Logistic Regression — ``WAS_mme_xcELR``, ``WAS_mme_ELR``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Estimates tercile probabilities directly by fitting a logistic regression
on the ensemble mean as a continuous predictor of each category boundary.

* ``WAS_mme_xcELR`` — wraps the xcast ELR implementation (splits on ``S``
  dimension).
* ``WAS_mme_ELR`` — native wass2s implementation.

.. code-block:: python

   from wass2s import WAS_mme_ELR, WAS_Cross_Validator

   elr = WAS_mme_ELR()
   cv = WAS_Cross_Validator(n_splits=len(obs.get_index("T")), nb_omit=2)
   hindcast_det, hindcast_prob = cv.cross_validate(
       elr, obs, hdcst, clim_year_start=1993, clim_year_end=2016
   )

Logistic MME — ``WAS_mme_logistic``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multinomial logistic regression trained on all GCM outputs simultaneously.
Produces direct probability estimates for each tercile category.

.. code-block:: python

   from wass2s import WAS_mme_logistic, WAS_Cross_Validator

   logistic_mme = WAS_mme_logistic()
   cv = WAS_Cross_Validator(n_splits=len(obs.get_index("T")), nb_omit=2)
   hindcast_det, hindcast_prob = cv.cross_validate(
       logistic_mme, obs, hdcst, clim_year_start=1993, clim_year_end=2016
   )

Mean and Variance Adjustment — ``WAS_mme_MVA``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A simple bias-correction technique that rescales the forecast mean and
variance to match the observational climatology.

.. code-block:: python

   from wass2s import WAS_mme_MVA

   mva = WAS_mme_MVA()
   mva.fit(hindcast_da=hdcst, obs_da=obs)
   calibrated = mva.transform(forecast_da=fcst)

Multi-Model Ensemble (MME) Techniques
======================================

The ``WAS_mme`` module provides a suite of tools to combine hindcasts and forecasts from different climate models (e.g., GCMs) in order to improve predictive skill. It includes methods ranging from simple weighted averages to advanced Machine Learning stacking and statistical calibration.

.. note::
   This module is actively maintained. Some advanced features are still under development.

**Key Features**:
* **Spatial Clustering**: Uses ML models (RF, XGBoost, MLP) with K‑Means to cluster grid points into homogeneous zones for robust hyperparameter optimization.
* **Hyperparameter Optimization (HPO)**: Integrated **Grid**, **Random**, and **Bayesian (Optuna)** search methods.
* **Calibration**: Implements methods like NGR and BMA to correct bias and spread.
* **Probabilistic Output**: All methods generate tercile probabilities (Below, Normal, Above).

-------------------------------------------------------------------------------

1. Data Preparation
===================

**Function**: ``process_datasets_for_mme``

A utility function to load, interpolate, and harmonize hindcasts and forecasts from different sources (GCMs, agro‑parameters, etc.) onto a common grid (usually the observational rainfall grid).

**Parameters**:
   - ``rainfall``: Observational data.
   - ``dir_to_save_model``: Directory to save processed data.
   - ``best_models``: List of model identifiers to include.
   - ``year_start``, ``year_end``: Training period.
   - ``score_metric``: Metric used for evaluation (e.g., 'GROC').

**Example**:

.. code-block:: python

   from wass2s import process_datasets_for_mme

   hdcst, fcst, obs, scores = process_datasets_for_mme(
       rainfall=obs_data,
       dir_to_save_model="./data/",
       best_models=['NCEP', 'ECMWF', 'UKMO'],
       year_start=1981,
       year_end=2010,
       score_metric='GROC'
   )

-------------------------------------------------------------------------------

2. Weighted Ensembles (Linear)
==============================

These methods combine models using linear weights based on historical performance.

**Class**: ``WAS_mme_Weighted``

Combines models using weights derived from a skill score (e.g., GROC, Pearson). Supports stepwise weighting to exclude poorly performing models.

**Logic**: If a model's score is below a given threshold, its weight is set to zero. Otherwise, the weight is proportional to the score (or assigned stepwise values like 0.6, 0.8, 1.0).

**Example**:

.. code-block:: python

   from wass2s import WAS_mme_Weighted

   mme_weighted = WAS_mme_Weighted(
       equal_weighted=False,
       metric='GROC',
       threshold=0.5
   )

   hcst_det, fcst_det = mme_weighted.compute(obs, hdcst, fcst, scores)

-------------------------------------------------------------------------------

3. Min et al. (2009) Probabilistic MME
=======================================

**Class**: ``WAS_Min2009_ProbWeighted``

Implements the PMME method from Min et al. (2009):

1. Calculates individual model probabilities (assuming Gaussian or Lognormal distribution).
2. Weights models based on the square root of their ensemble size.
3. Combines probabilities directly.
4. Computes a **Chi‑Square Combined Map** to identify areas with significant signal.

-------------------------------------------------------------------------------

4. Machine Learning Ensembles
=============================

These methods treat the MME problem as a regression/classification task: given GCM outputs (predictors), predict observed rainfall. They typically employ **spatial clustering** to group grid cells and optimize hyperparameters per cluster.

**Stacking (Super‑Ensemble)**

**Class**: ``WAS_mme_Stacking``

A state‑of‑the‑art method that stacks multiple base learners (RF, XGBoost, MLP, ELM) and combines them using a meta‑learner (Ridge, Lasso, or ElasticNet).

* **Layer 0 (Base)**: Random Forest, XGBoost, HP‑ELM, MLP.
* **Layer 1 (Meta)**: Linear model trained on Out‑of‑Fold (OOF) predictions from Layer 0.

**Example**:

.. code-block:: python

   from wass2s import WAS_mme_Stacking

   stacking_model = WAS_mme_Stacking(
       meta_learner_type='ridge',
       meta_search_method='bayesian',
       n_clusters=4
   )

   fcst_det, fcst_prob = stacking_model.forecast(
       Predictant=obs,
       clim_year_start=1981,
       clim_year_end=2010,
       hindcast_det=hdcst,            # shape: (T, M, Y, X)
       hindcast_det_cross=hdcst_cv,   # for error variance
       Predictor_for_year=fcst_year
   )

**Individual ML Regressors**

* **Random Forest**: ``WAS_mme_RF``
* **XGBoost**: ``WAS_mme_XGBoosting``
* **MLP (Neural Network)**: ``WAS_mme_MLP``
* **Extreme Learning Machine**: ``WAS_mme_hpELM``

-------------------------------------------------------------------------------

5. Calibration & Post‑Processing
================================

Methods to correct bias and reliability errors in raw ensembles.

**Non‑homogeneous Gaussian Regression (NGR)**

**Class**: ``WAS_mme_NGR``

Calibrates the ensemble mean and spread by minimizing the **Continuous Ranked Probability Score (CRPS)**.

* **Model**:
   .. math::
      \mu = a + b \cdot \bar{x}_{\text{ens}}, \quad
      \sigma = \sqrt{c^2 + d^2 \cdot s^2_{\text{ens}}}

**Bayesian Model Averaging (BMA)**

* **Class**: ``WAS_GaussianBMA_EM`` & ``WAS_EnsembleBMA``
   * **Gaussian BMA**: Standard BMA for temperature or normally‑distributed variables using the Expectation‑Maximization (EM) algorithm.

**Extended Logistic Regression (ELR)**

* **Class**: ``WAS_mme_xcELR`` & ``WAS_mme_logistic``
   * **WAS_mme_xcELR**: Wraps `xcast` Extended Logistic Regression.
   * **WAS_mme_logistic**: Native implementation of Multinomial Logistic Regression to predict tercile probabilities directly.
   * **WAS_mme_Gaussian_process**: (Under development)

**Mean and Variance Adjustment (MVA)**

**Class**: ``WAS_mme_MVA``

A simple bias‑correction technique that rescales the forecast to match the climatological mean and variance of the observations.

**Example**:

.. code-block:: python

   from wass2s import WAS_mme_MVA

   mva = WAS_mme_MVA()
   mva.fit(hindcast_da, obs_da)
   calibrated_fcst = mva.transform(forecast_da)


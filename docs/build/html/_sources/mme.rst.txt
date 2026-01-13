Multi-Model Ensemble (MME) Techniques
-------------------------------------
**This section is under construction.**

The ``WAS_mme`` module provides a suite of tools to combine hindcasts/forecasts from different models (e.g., GCMs) to improve predictive skill. It ranges from simple weighted averages to complex Machine Learning stacking and statistical calibration.

**Key Features**:
* **Spatial Clustering**: ML models (RF, XGB, MLP) use K-Means to cluster grid points into homogeneous zones for robust hyperparameter optimization.
* **Hyperparameter Optimization (HPO)**: Integrated **Grid**, **Random**, and **Bayesian (Optuna)** search.
* **Calibration**: Methods like NGR and BMA to correct bias and spread.
* **Probabilistic Output**: All methods generate tercile probabilities (Below, Normal, Above).

-------------------------------------------------------------------------------

1. Data Preparation
===================

**Function**: ``process_datasets_for_mme``

A utility function to load, interpolate, and harmonize hindcasts and forecasts from different sources (GCMs, Agro-parameters, etc.) onto a common grid (usually the observational rainfall grid).

.. code-block:: python

   from wass2s import process_datasets_for_mme

   # Load and prep data for ensemble
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

Combines models using weights derived from a skill score (e.g., GROC, Pearson). It supports "stepwise" weighting to zero-out poor models.

* **Logic**: If Score < Threshold, Weight = 0. Otherwise, weight is proportional to the score (or steps like 0.6, 0.8, 1.0).

.. code-block:: python

   from wass2s import WAS_mme_Weighted

   mme_weighted = WAS_mme_Weighted(
       equal_weighted=False, 
       metric='GROC', 
       threshold=0.5
   )

   # Compute weighted ensemble
   hcst_det, fcst_det = mme_weighted.compute(obs, hdcst, fcst, scores)

3. Min et al. (2009) Probabilistic MME
============================================

**Class**: ``WAS_Min2009_ProbWeighted``

Implements the PMME method (Min et al., 2009).
1.  Calculates individual model probabilities (assuming Gaussian/Lognormal).
2.  Weights models based on the square root of their ensemble size.
3.  Combines probabilities directly.
4.  Computes a **Chi-Square Combined Map** to identify areas with significant signal.

-------------------------------------------------------------------------------

4. Machine Learning Ensembles
=============================

These classes treat the MME problem as a regression/classification task: *Given GCM outputs (predictors), predict Observed Rainfall*. They typically employ **Spatial Clustering** to group grid cells and optimize hyperparameters per cluster.

* Stacking (Super-Ensemble)


**Class**: ``WAS_mme_Stacking``

A state-of-the-art method that stacks multiple base learners (RF, XGBoost, MLP, ELM) and combines them using a meta-learner (Ridge, Lasso, or ElasticNet).


* **Layer 0 (Base)**: Random Forest, XGBoost, HP-ELM, MLP.
* **Layer 1 (Meta)**: Linear model trained on Out-of-Fold (OOF) predictions from Layer 0.

.. code-block:: python

   from wass2s import WAS_mme_Stacking

   # Initialize Stacking with Bayesian Optimization
   stacking_model = WAS_mme_Stacking(
       meta_learner_type='ridge',
       meta_search_method='bayesian',
       n_clusters=4
   )

   # Forecast
   # Automatically handles clustering, HPO, and stacking
   fcst_det, fcst_prob = stacking_model.forecast(
       Predictant=obs,
       clim_year_start=1981, clim_year_end=2010,
       hindcast_det=hdcst,           # (T, M, Y, X)
       hindcast_det_cross=hdcst_cv,  # For error variance
       Predictor_for_year=fcst_year
   )

* Individual ML Regressors

* **Random Forest**: ``WAS_mme_RF``
* **XGBoost**: ``WAS_mme_XGBoosting``
* **MLP (Neural Net)**: ``WAS_mme_MLP``
* **Extreme Learning Machine**: ``WAS_mme_hpELM``

-------------------------------------------------------------------------------

5. Calibration & Post-Processing
================================

Methods to correct bias and reliability errors in raw ensembles.

* **Non-homogeneous Gaussian Regression (NGR)**


**Class**: ``WAS_mme_NGR``

Calibrates the ensemble mean and spread by minimizing the **CRPS** (Continuous Ranked Probability Score).
* **Model**: :math:`\mu = a + b \cdot \bar{x}_{ens}`, :math:`\sigma = \sqrt{c^2 + d^2 \cdot s^2_{ens}}`

* **Bayesian Model Averaging (BMA)**


**Class**: ``WAS_GaussianBMA_EM`` & ``WAS_EnsembleBMA``

* **Gaussian BMA**: Standard BMA for temperature or normal-distributed variables using EM algorithm.



* **Extended Logistic Regression (ELR)**


**Class**: ``WAS_mme_xcELR`` & ``WAS_mme_logistic``

* **WAS_mme_xcELR**: Wraps `xcast` Extended Logistic Regression.
* **WAS_mme_logistic**: Native implementation of Multinomial Logistic Regression to predict tercile probabilities directly.
* **WAS_mme_Guassian_process**: 

* **Mean and Variance Adjustment (MVA)**


**Class**: ``WAS_mme_MVA``

A simple bias correction technique that rescales the forecast to match the climatological mean and variance of the observations.

.. code-block:: python

   from wass2s import WAS_mme_MVA
   
   mva = WAS_mme_MVA()
   mva.fit(hindcast_da, obs_da)
   calibrated_fcst = mva.transform(forecast_da)

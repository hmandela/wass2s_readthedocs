Verification
============

The ``WAS_Verification`` class provides deterministic, probabilistic, and
ensemble verification metrics for gridded seasonal forecasts. All scores are
computed pixel-wise on ``xarray.DataArrays`` and can be visualised directly
on Cartopy maps.

.. code-block:: python

   from wass2s import WAS_Verification

   verifier = WAS_Verification()

Metadata for all available scores (name, range, colormap, type) can be
retrieved as a dictionary:

.. code-block:: python

   metadata = verifier.get_scores_metadata()

-------------------------------------------------------------------------------

Deterministic Metrics
----------------------

These metrics compare point predictions to observations and are computed via
``compute_deterministic_score``, which applies the chosen function spatially:

.. code-block:: python

   score_map = verifier.compute_deterministic_score(
       score_func, obs_data, model_data
   )

Available deterministic metrics:

+--------------------+----------------------------+---------------+
| Method             | Score                      | Optimal value |
+====================+============================+===============+
| ``pearson_corr``   | Pearson correlation        | 1.0           |
+--------------------+----------------------------+---------------+
| ``spearman_corr``  | Spearman rank correlation  | 1.0           |
+--------------------+----------------------------+---------------+
| ``index_of_agreement`` | Willmott Index of Agreement | 1.0      |
+--------------------+----------------------------+---------------+
| ``nash_sutcliffe_efficiency`` | NSE             | 1.0           |
+--------------------+----------------------------+---------------+
| ``kling_gupta_efficiency`` | KGE                | 1.0           |
+--------------------+----------------------------+---------------+
| ``mean_absolute_error`` | MAE                  | 0.0           |
+--------------------+----------------------------+---------------+
| ``root_mean_square_error`` | RMSE              | 0.0           |
+--------------------+----------------------------+---------------+

**Example**

.. code-block:: python

   pearson = verifier.compute_deterministic_score(
       verifier.pearson_corr, obs_data, hindcast_det
   )
   kge = verifier.compute_deterministic_score(
       verifier.kling_gupta_efficiency, obs_data, hindcast_det
   )

   verifier.plot_model_score(
       pearson, "Pearson",
       dir_save_score="./scores/", figure_name="Pearson_score"
   )

To compare multiple models in a grid layout:

.. code-block:: python

   verifier.plot_models_score(
       {"Ridge": pearson_ridge, "MARS": pearson_mars},
       "Pearson",
       dir_save_score="./scores/"
   )

-------------------------------------------------------------------------------

Probabilistic Metrics
----------------------

Probabilistic metrics evaluate tercile-probability forecasts (PB, PN, PA)
and are computed via ``compute_probabilistic_score``.

Available probabilistic metrics:

+-------------------------------+----------------------------------------------+
| Method                        | Description                                  |
+===============================+==============================================+
| ``calculate_groc``            | Generalized ROC score (0–1, 0.5 = no skill) |
+-------------------------------+----------------------------------------------+
| ``calculate_groc_weighted``   | Area-weighted GROC                           |
+-------------------------------+----------------------------------------------+
| ``calculate_rpss``            | Ranked Probability Skill Score (−∞ to 1)    |
+-------------------------------+----------------------------------------------+
| ``ignorance_score``           | Ignorance score (lower = better)             |
+-------------------------------+----------------------------------------------+
| ``brier_score``               | Brier score for a given event category       |
+-------------------------------+----------------------------------------------+
| ``brier_skill_score``         | BSS relative to climatological reference     |
+-------------------------------+----------------------------------------------+
| ``resolution_score_grid``     | Resolution component of Brier decomposition  |
+-------------------------------+----------------------------------------------+

**Example**

.. code-block:: python

   groc = verifier.compute_probabilistic_score(
       verifier.calculate_groc, obs_data, hindcast_prob,
       clim_year_start=1991, clim_year_end=2020
   )
   rpss = verifier.compute_probabilistic_score(
       verifier.calculate_rpss, obs_data, hindcast_prob,
       clim_year_start=1991, clim_year_end=2020
   )

   verifier.plot_model_score(groc, "GROC", dir_save_score="./scores/")

Reliability diagrams and ROC curves
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Reliability diagram for each tercile category
   verifier.reliability_diagram(
       modelname="Ridge",
       dir_to_save_score="./scores/",
       y_true=obs_data,
       y_probs=hindcast_prob,
       clim_year_start=1991, clim_year_end=2020
   )

   # ROC curves with 95 % bootstrap confidence intervals
   verifier.plot_roc_curves(
       modelname="Ridge",
       dir_to_save_score="./scores/",
       y_true=obs_data,
       y_probs=hindcast_prob,
       clim_year_start=1991, clim_year_end=2020,
       n_bootstraps=1000, ci=0.95
   )

-------------------------------------------------------------------------------

Ensemble Metrics
-----------------

The Continuous Ranked Probability Score (CRPS) evaluates forecasts that
provide a full ensemble distribution (e.g. raw GCM ensembles).

.. code-block:: python

   crps_map = verifier.compute_crps(
       obs_data, ensemble_hindcast,
       member_dim="number", dim="T"
   )

-------------------------------------------------------------------------------

Year-by-Year Validation
------------------------

Two utilities support annual verification:

.. code-block:: python

   # Classify a specific year relative to climatology
   # (Well Above Average, Above Average, Near Average, …)
   verifier.ratio_to_average(
       predictant=obs_data,
       clim_year_start=1991, clim_year_end=2020,
       year=2023
   )

   # Compute RPSS for one year and plot the result on a map
   verifier.compute_one_year_rpss(
       obs=obs_data,
       prob_pred=hindcast_prob,
       clim_year_start=1991, clim_year_end=2020,
       year=2023
   )

-------------------------------------------------------------------------------

Tercile Classification Utilities
----------------------------------

.. code-block:: python

   # Assign each observed year to a tercile class (0=BN, 1=NN, 2=AN)
   classes = verifier.compute_class(
       Predictant=obs_data,
       clim_year_start=1991, clim_year_end=2020
   )

   # Classify with pre-computed thresholds
   classified = verifier.classify_data_into_terciles(
       y=obs_data, T1=t1_map, T2=t2_map
   )

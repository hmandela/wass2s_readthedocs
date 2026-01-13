--------------------
Verification Module
--------------------
**This section is under construction.**

The Verification module provides tools for evaluating the performance of climate forecasts using a variety of deterministic, probabilistic, and ensemble-based metrics. It is implemented in the `was_verification.py` module and leverages the `WAS_Verification` class to compute metrics such as Kling-Gupta Efficiency (KGE), Pearson Correlation, Ranked Probability Skill Score (RPSS), and Continuous Ranked Probability Score (CRPS). The module also includes visualization utilities for plotting scores, reliability diagrams, and ROC curves.

This module is designed to work with gridded climate data, typically stored in `xarray` DataArrays, and supports parallel computation using `dask` for efficiency with large datasets.


The `WAS_Verification` class is the core of the Verification module, providing methods to compute and visualize various performance metrics for climate forecasts.

**Initialization**

.. code-block:: python

    from wass2s.was_verification import WAS_Verification

    # Initialize with a distribution method for probabilistic forecasts
    verifier = WAS_Verification()



**Available Metrics**

The class defines a dictionary of scoring metrics with metadata, including:

- **Deterministic Metrics**:
  - `KGE`: Kling-Gupta Efficiency (-1 to 1).
  - `Pearson`: Pearson Correlation Coefficient (-1 to 1).
  - `IOA`: Index of Agreement (0 to 1).
  - `MAE`: Mean Absolute Error (0 to 100).
  - `RMSE`: Root Mean Square Error (0 to 100).
  - `NSE`: Nash-Sutcliffe Efficiency (None to 1).
  - `TAYLOR_DIAGRAM`: Taylor Diagram (visualization).

- **Probabilistic Metrics**:
  - `GROC`: Generalized Receiver Operating Characteristic (0 to 1).
  - `RPSS`: Ranked Probability Skill Score (-1 to 1).
  - `IGS`: Ignorance Score (0 to None).
  - `RES`: Resolution Score (0 to None).
  - `REL`: Reliability Score (None to None).
  - `RELIABILITY_DIAGRAM`: Reliability Diagram (visualization).
  - `ROC_CURVE`: Receiver Operating Characteristic Curve (visualization).

- **Ensemble Metrics**:
  - `CRPS`: Continuous Ranked Probability Score (0 to 100).

**Metadata Access**

.. code-block:: python

    metadata = verifier.get_scores_metadata()

This returns a dictionary containing the name, range, type, colormap, and computation function for each metric.

==============================================
Deterministic Metrics
==============================================

Deterministic metrics evaluate the performance of point forecasts against observations. They are computed using the `compute_deterministic_score` method, which applies a scoring function over `xarray` DataArrays.

**Example Usage**

.. code-block:: python

    # Compute Pearson Correlation
    pearson_score = verifier.compute_deterministic_score(
        verifier.pearson_corr, obs_data, model_data
    )

    # Plot the score
    verifier.plot_model_score(pearson_score, "Pearson", dir_save_score="./scores", figure_name="Pearson_Score")

**Key Methods**

- `kling_gupta_efficiency`: Computes KGE, balancing correlation, bias, and variability.
- `pearson_corr`: Computes Pearson Correlation Coefficient.
- `index_of_agreement`: Computes IOA, measuring agreement between predictions and observations.
- `mean_absolute_error`: Computes MAE, the average absolute difference.
- `root_mean_square_error`: Computes RMSE, the square root of mean squared differences.
- `nash_sutcliffe_efficiency`: Computes NSE, comparing prediction errors to the mean of observations.
- `taylor_diagram`: Placeholder for Taylor Diagram visualization (to be implemented).

**Plotting**

The `plot_model_score` method visualizes deterministic scores on a map using `cartopy`.

.. code-block:: python

    verifier.plot_model_score(score_result, "KGE", dir_save_score="./scores", figure_name="KGE_Model")

The `plot_models_score` method plots multiple model scores in a grid.

.. code-block:: python

    model_metrics = {
        "model1": score_result1,
        "model2": score_result2
    }
    verifier.plot_models_score(model_metrics, "Pearson", dir_save_score="./scores")

==============================================
Probabilistic Metrics
==============================================

Probabilistic metrics evaluate the performance of forecasts that provide probabilities for tercile categories (below-normal, near-normal, above-normal). These are computed using the `compute_probabilistic_score` method.

**Example Usage**

.. code-block:: python

    # Compute tercile probabilities
    proba_forecast = verifier.gcm_compute_prob(obs_data, clim_year_start=1981, clim_year_end=2010, hindcast_det=model_data)

    # Compute RPSS
    rpss_score = verifier.compute_probabilistic_score(
        verifier.calculate_rpss, obs_data, proba_forecast, clim_year_start=1981, clim_year_end=2010
    )

**Key Methods**

- `classify`: Classifies data into terciles based on climatology.
- `compute_class`: Computes tercile class labels for observations.
- `calculate_groc`: Computes GROC, averaging AUC across tercile categories.
- `calculate_rpss`: Computes RPSS, comparing forecast probabilities to climatology.
- `ignorance_score`: Computes Ignorance Score per Weijs (2010).
- `resolution_score_grid`: Computes Resolution Score, measuring how forecasts differ from climatology.
- `reliability_score_grid`: Computes Reliability Score, assessing forecast calibration.
- `reliability_diagram`: Plots Reliability Diagrams for each tercile category.
- `plot_roc_curves`: Plots ROC Curves with confidence intervals for each tercile.

**Visualization**

Reliability Diagrams and ROC Curves are generated for probabilistic forecasts.

.. code-block:: python

    # Plot Reliability Diagram
    verifier.reliability_diagram(
        modelname="Model1", dir_to_save_score="./scores", y_true=obs_data, y_probs=proba_forecast,
        clim_year_start=1981, clim_year_end=2010
    )

    # Plot ROC Curves with 95% confidence intervals
    verifier.plot_roc_curves(
        modelname="Model1", dir_to_save_score="./scores", y_true=obs_data, y_probs=proba_forecast,
        clim_year_start=1981, clim_year_end=2010, n_bootstraps=1000, ci=0.95
    )

==============================================
Ensemble Metrics
==============================================

Ensemble metrics evaluate forecasts with multiple members, such as those from GCMs. The primary metric is CRPS, computed using `xskillscore`.

**Example Usage**

.. code-block:: python

    # Compute CRPS for ensemble forecast
    crps_score = verifier.compute_crps(obs_data, model_data, member_dim='number', dim="T")

**Key Methods**

- `compute_crps`: Computes CRPS for ensemble forecasts, measuring the difference between predicted and observed distributions.


==============================================
Annual Year Validation
==============================================

The module provides utilities to validate forecasts for a specific year, including ratio-to-average classification and RPSS computation.

**Key Methods**

- `ratio_to_average`: Classifies forecast data relative to the climatological mean into categories (e.g., Well Above Average, Near Average).
- `compute_one_year_rpss`: Computes RPSS for a specific year and visualizes it on a map.

**Example Usage**

.. code-block:: python

    # Classify ratio to average for a specific year
    verifier.ratio_to_average(predictant=obs_data, clim_year_start=1981, clim_year_end=2010, year=2020)

    # Compute RPSS for a specific year
    verifier.compute_one_year_rpss(
        obs=obs_data, prob_pred=proba_forecast, clim_year_start=1981, clim_year_end=2010, year=2020
    )

This documentation provides an overview of the Verification module's capabilities, along with example usage for key methods. 
For detailed information on each method, refer to the source code and docstrings in API.



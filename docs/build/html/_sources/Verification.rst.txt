**Verification Module**

The Verification module provides tools for evaluating the performance of climate forecasts using a variety of deterministic, probabilistic, and ensemble-based metrics. It is implemented in the `was_verification.py` module and leverages the `WAS_Verification` class to compute metrics such as Kling-Gupta Efficiency (KGE), Pearson Correlation, Ranked Probability Skill Score (RPSS), and Continuous Ranked Probability Score (CRPS). The module also includes visualization utilities for plotting scores, reliability diagrams, and ROC curves.

This module is designed to work with gridded climate data, typically stored in `xarray` DataArrays, and supports parallel computation using `dask` for efficiency with large datasets.

**Prerequisites**

- **xarray**: For handling multi-dimensional data arrays.
- **numpy**: For numerical computations.
- **scipy**: For statistical functions.
- **scikit-learn**: For metrics and utilities like ROC curves and one-hot encoding.
- **xskillscore**: For ensemble-based metrics like CRPS.
- **matplotlib** and **cartopy**: For plotting maps and diagrams.
- **properscoring**: For scoring probabilistic forecasts.
- **dask**: For parallel computing.

==============================================
WAS_Verification Class
==============================================

The `WAS_Verification` class is the core of the Verification module, providing methods to compute and visualize various performance metrics for climate forecasts.

**Initialization**

.. code-block:: python

    from wass2s.was_verification import WAS_Verification

    # Initialize with a distribution method for probabilistic forecasts
    verifier = WAS_Verification(dist_method="gamma")

**Parameters**

- `dist_method`: Specifies the distribution method for computing tercile probabilities. Options include:
  - `"t"`: Student's t-based method.
  - `"gamma"`: Gamma distribution-based method (default).
  - `"normal"`: Normal distribution-based method.
  - `"lognormal"`: Lognormal distribution-based method.
  - `"weibull_min"`: Weibull minimum distribution-based method.
  - `"nonparam"`: Non-parametric method using historical errors.

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
Tercile Probability Computation
==============================================

The module provides multiple methods to compute tercile probabilities for probabilistic forecasts, based on different distributional assumptions.

**Key Methods**

- `calculate_tercile_probabilities`: Uses Student's t-distribution.
- `calculate_tercile_probabilities_gamma`: Uses Gamma distribution.
- `calculate_tercile_probabilities_normal`: Uses Normal distribution.
- `calculate_tercile_probabilities_lognormal`: Uses Lognormal distribution.
- `calculate_tercile_probabilities_weibull_min`: Uses Weibull minimum distribution.
- `calculate_tercile_probabilities_nonparametric`: Uses historical errors for a non-parametric approach.

**Example Usage**

.. code-block:: python

    # Compute probabilities using Gamma distribution
    hindcast_prob = verifier.gcm_compute_prob(
        Predictant=obs_data, clim_year_start=1981, clim_year_end=2010, hindcast_det=model_data
    )

The `gcm_compute_prob` method selects the appropriate distribution based on the `dist_method` parameter.

==============================================
GCM Validation
==============================================

The module includes methods to validate General Circulation Model (GCM) forecasts against observations, supporting both deterministic and probabilistic metrics.

**Key Methods**

- `gcm_validation_compute`: Validates GCM forecasts for multiple models, computing specified metrics.
- `weighted_gcm_forecasts`: Combines forecasts from multiple models using weights based on a performance metric (e.g., GROC).

**Example Usage**

.. code-block:: python

    # Validate GCM forecasts
    models_files_path = {
        "model1": "path/to/model1.nc",
        "model2": "path/to/model2.nc"
    }
    x_metric = verifier.gcm_validation_compute(
        models_files_path=models_files_path, Obs=obs_data, score="Pearson",
        month_of_initialization=3, clim_year_start=1981, clim_year_end=2010,
        dir_to_save_roc_reliability="./scores", lead_time=[1]
    )

    # Compute weighted GCM forecasts
    hindcast_det, hindcast_prob, forecast_prob = verifier.weighted_gcm_forecasts(
        Obs=obs_data, best_models={"model1_MarIc_JFM_1": score1}, scores={"GROC": x_metric},
        lead_time=[1], model_dir="./models", clim_year_start=1981, clim_year_end=2010, variable="PRCP"
    )

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

==============================================
Notes

- **Placeholder Functions**: Some methods (e.g., `taylor_diagram`) are placeholders and require implementation based on specific needs.
- **Gridded Data**: The module currently supports only gridded data validation. Non-gridded validation is not implemented.
- **Performance**: The use of `dask` ensures efficient computation for large datasets, but users should ensure proper chunking of `xarray` DataArrays.
- **Visualization**: Plots are saved to the specified directory and displayed using `matplotlib`. Ensure the output directory exists.

This documentation provides an overview of the Verification module's capabilities, along with example usage for key methods. For detailed information on each method, refer to the source code and docstrings in `was_verification.py`.



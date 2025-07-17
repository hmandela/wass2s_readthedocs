Multi-Model Ensemble (MME) Techniques
-------------------------------------
**This section is under construction.**

The `was_mme.py` module provides classes for combining predictions from multiple models, including:

- `WAS_mme_ELM`: Extreme Learning Machine for MME.
- `WAS_mme_EPOELM`: Enhanced Parallel Online Extreme Learning Machine.
- `WAS_mme_MLP`: Multi-Layer Perceptron for MME.
- `WAS_mme_GradientBoosting`: Gradient Boosting for MME.
- `WAS_mme_XGBoosting`: XGBoost for MME.
- `WAS_mme_AdaBoost`: AdaBoost for MME.
- `WAS_mme_LGBM_Boosting`: LightGBM Boosting for MME.
- `WAS_mme_Stack_MLP_RF`: Stacking model with MLP and Random Forest.
- `WAS_mme_Stack_Lasso_RF_MLP`: Stacking model with Lasso, Random Forest, and MLP.
- `WAS_mme_Stack_MLP_Ada_Ridge`: Stacking model with MLP, AdaBoost, and Ridge.
- `WAS_mme_Stack_RF_GB_Ridge`: Stacking model with Random Forest, Gradient Boosting, and Ridge.
- `WAS_mme_Stack_KNN_Tree_SVR`: Stacking model with KNN, Decision Tree, and SVR.
- `WAS_mme_GA`: Genetic Algorithm for MME.


Each MME class includes methods for computing the ensemble model and, where applicable, computing probabilities.

**Example Usage with WAS_mme_ELM**

.. code-block:: python

    from wass2s.was_mme import WAS_mme_ELM

    # Define ELM parameters
    elm_kwargs = {
        'regularization': 10,
        'hidden_layer_size': 4,
        'activation': 'lin',  # Options: 'sigm', 'tanh', 'lin', 'relu'
        'preprocessing': 'none',  # Options: 'minmax', 'std', 'none'
        'n_estimators': 10,
    }

    # Initialize the MME ELM model
    model = WAS_mme_ELM(elm_kwargs=elm_kwargs, dist_method="euclidean")

    # Process datasets for MME (user-defined function)
    all_model_hdcst, all_model_fcst, obs, best_score = process_datasets_for_mme(
        rainfall.sel(T=slice(str(year_start), str(year_end))),
        gcm=True, ELM_ELR=True, dir_to_save_model="./models",
        best_models=[], scores=[], year_start=1990, year_end=2020,
        model=True, month_of_initialization=3, lead_time=1, year_forecast=2021
    )

    # Initialize cross-validator
    was_mme_gcm = WAS_Cross_Validator(
        n_splits=len(rainfall.sel(T=slice(str(year_start), str(year_end))).get_index("T")),
        nb_omit=2
    )

    # Perform cross-validation
    hindcast_det_gcm, hindcast_prob_gcm = was_mme_gcm.cross_validate(
        model, obs, all_model_hdcst, clim_year_start, clim_year_end
    )


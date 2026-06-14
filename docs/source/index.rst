wass2s — West Africa Seasonal Forecasting System
=================================================

**wass2s** is a Python library that streamlines the full pipeline of seasonal
climate forecasting over West Africa and the Sahel. Starting from raw satellite,
reanalysis, and GCM data, it lets you build, validate, and operationalise
statistical and machine-learning-based forecasts in a reproducible way.

The library implements the *new-generation* seasonal forecasting framework
promoted by the World Meteorological Organization (WMO), covering everything
from predictor preparation to probabilistic tercile-probability maps
(Below-Normal / Near-Normal / Above-Normal).

.. note::
   A companion set of `Jupyter notebooks <https://github.com/hmandela/WASS2S_notebooks>`_
   is available for hands-on walkthroughs of each major workflow.

Key capabilities
----------------

* **Automated data acquisition** — ERA5, CHIRPS, TAMSAT, NMME, C3S GCMs, and
  AgERA5 agro-indicators in a single unified interface.
* **Agroclimatic predictand computation** — onset, cessation, dry/wet spells,
  ETCCDI extreme indices, and heat-wave metrics from daily gridded or station data.
* **Bias correction and data merging** — quantile mapping (QUANT, RQUANT,
  SSPLIN, PTF, DIST) and station–gridded merging (Kriging, Regression Kriging,
  Neural-Network Kriging, Multiplicative Bias).
* **Statistical and ML models** — OLS, Ridge, Lasso, ElasticNet, MARS, SVR,
  MLP, stacking ensembles, EOF/PCR, CCA, and analog methods, all sharing a
  common ``compute_model / compute_prob / forecast`` interface.
* **Multi-model ensemble post-processing** — weighted averaging, BMA, NGR,
  Extended Logistic Regression, Random Forest / XGBoost / ELM / MLP super-ensembles.
* **Leakage-free cross-validation** — a custom leave-one-out splitter with a
  symmetric exclusion window, wired to every model family automatically.
* **Forecast verification** — deterministic (KGE, Pearson, NSE, RMSE, MAE)
  and probabilistic (GROC, RPSS, Brier, Ignorance, ROC, Reliability) metrics.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   Installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guide

   Download
   Preprocessing
   Models
   Verification
   mme
   ImplementWASNextGen

.. toctree::
   :maxdepth: 1
   :caption: API reference

   api

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

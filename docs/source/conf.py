import os
import sys

# Point Sphinx at the wass2s source tree so autodoc can import the package.
# Adjust this path if you move the docs folder relative to the package.
sys.path.insert(0, os.path.abspath('/home/user/Documents/AICCRA_AGRHYMET_2024/My_ML_training/WAS_S2S_/WASS2S/wass2s'))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project   = 'wass2s'
copyright = '2025, Mandela C. M. HOUNGNIBO'
author    = 'Mandela C. M. HOUNGNIBO'
release   = '0.4.7.4'

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',       # Auto-generate docs from docstrings
    'sphinx.ext.autosummary',   # Summary tables across modules
    'sphinx.ext.viewcode',      # Links to highlighted source code
    'sphinx.ext.napoleon',      # Google / NumPy docstring support
    'sphinx.ext.intersphinx',   # Cross-links to xarray, numpy, etc.
]

# ---------------------------------------------------------------------------
# autodoc settings
# ---------------------------------------------------------------------------
autodoc_default_options = {
    'members':           True,
    'undoc-members':     True,
    'show-inheritance':  True,
    # Exclude private members and inherited sklearn internals to reduce noise.
    'private-members':   False,
    'inherited-members': False,
    'special-members':   '__init__',
}

# Order members as they appear in the source file
autodoc_member_order = 'bysource'

# ---------------------------------------------------------------------------
# Napoleon (docstring style)
# ---------------------------------------------------------------------------
napoleon_google_docstring         = True
napoleon_numpy_docstring          = True
napoleon_include_init_with_doc    = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

# ---------------------------------------------------------------------------
# intersphinx -- external cross-references
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy':  ('https://numpy.org/doc/stable', None),
    'xarray': ('https://docs.xarray.dev/en/stable', None),
    'scipy':  ('https://docs.scipy.org/doc/scipy', None),
}

# ---------------------------------------------------------------------------
# Warning suppression
# ---------------------------------------------------------------------------
# wass2s shares many method names (compute_model, compute_prob, forecast)
# across model classes.  autodoc raises a duplicate-object warning for each.
# Suppress those categories here rather than adding :no-index: everywhere.
suppress_warnings = [
    'autodoc',       # duplicate-object description warnings
    'ref.python',    # unresolved cross-refs from third-party docstrings
]

nitpick_ignore = [
    ('py:class', 'sklearn.base.BaseEstimator'),
    ('py:class', 'xgboost.sklearn.XGBRegressor'),
    ('py:class', 'scipy.stats._distn_infrastructure.rv_continuous_frozen'),
]

# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------
templates_path   = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
html_theme       = 'sphinx_rtd_theme'
html_static_path = ['_static']

# ---------------------------------------------------------------------------
# LaTeX / PDF output
# ---------------------------------------------------------------------------
latex_use_xindy = False   # Use makeindex instead (avoids xindy symlink bug)
latex_engine    = 'xelatex'

latex_elements = {
    'preamble': r"""
\setlength{\headheight}{24pt}
\usepackage{enumitem}
\setlistdepth{8}
\usepackage{fontspec}
\usepackage{imakeidx}
\makeindex
\usepackage{microtype}
""",
    'fontpkg': r"""
\setmainfont{FreeSerif}
\setsansfont{FreeSans}
\setmonofont{FreeMono}
""",
    'inputenc': '',
    'fontenc':  '',
    'babel':    '',
}

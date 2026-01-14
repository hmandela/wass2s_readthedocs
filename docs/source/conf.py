import os, sys
sys.path.insert(0, os.path.abspath('/home/user/Documents/AICCRA_AGRHYMET_2024/My_ML_training/WAS_S2S_/WASS2S/wass2s'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'wass2s: A python-based tool for seasonal climate forecast in West Africa and the Sahel.'
copyright = '2025, Mandela C. M. HOUNGNIBO'
author = 'Mandela C. M. HOUNGNIBO'
release = '0.3.4'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',        # For auto-generating docs from docstrings
    'sphinx.ext.autosummary',    # For generating summary tables
    'sphinx.ext.viewcode',       # Optional: adds links to source code
    'sphinx.ext.napoleon',       # For Google/NumPy style docstrings
]

# Automatically document all members (functions, classes, methods)
autodoc_default_options = {
    'members': True,
    'undoc-members': True,      # Include members without docstrings
    'private-members': True,    # Include private members (starting with _)
    'special-members': '__init__, __call__',  # Include special methods
    'inherited-members': True,  # Include inherited members
    'show-inheritance': True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True

#extensions = [
#    'sphinx.ext.autodoc',      # extract docs from docstrings
#    'sphinx.ext.napoleon',     # Google / NumPy docstring support
#    'sphinx.ext.viewcode',     # add links to source code
#		]
# Suppress some warnings
#suppress_warnings = [
#    'autodoc.duplicate_object',
#    'docutils.parsers.rst.duplicate_target'
#]

# Or use nitpick_ignore
nitpick_ignore = [
    ('py:class', 'wass2s.was_transformdata.WAS_TransformData.apply_transformation'),
    # Add other duplicates here
]        	

templates_path = ['_templates']
exclude_patterns = []

 
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


latex_engine = 'xelatex'
latex_elements = {
    'preamble': r'''
    \usepackage{enumitem}
    \setlistdepth{8}
    ''',
    'inputenc': '',
    'fontenc': '',
    'babel': '',
}


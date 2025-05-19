import os, sys
sys.path.insert(0, os.path.abspath('/home/user/Documents/AICCRA_AGRHYMET_2024/My_ML_training/WAS_S2S_/WASS2S/wass2s'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'wass2s'
copyright = '2025, Mandela C. M. HOUNGNIBO'
author = 'Mandela C. M. HOUNGNIBO'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # extract docs from docstrings
    'sphinx.ext.napoleon',     # Google / NumPy docstring support
    #'sphinx.ext.viewcode',     # add links to source code
		]	

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

latex_elements = {
    'preamble': r'''
    \usepackage{enumitem}
    \setlistdepth{8}
    ''',
}
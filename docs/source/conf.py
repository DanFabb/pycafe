# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Path setup --------------------------------------------------------------

# The repo ships two distributions; both source trees are on the path,
# so that autodoc reads pycafe and pycafe_vibro from the checkout.
_root = os.path.abspath("../..")
sys.path.insert(0, os.path.join(_root, "packages", "pycafe", "src"))
sys.path.insert(0, os.path.join(_root, "packages", "pycafe-vibro", "src"))

# -- Project information -----------------------------------------------------

project = "pyCAFE"
author = "Daniele Fabbri"
copyright = "2025, Daniele Fabbri"
release = "1.0.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}

napoleon_numpy_docstring = True
napoleon_google_docstring = False

templates_path = ["_templates"]
exclude_patterns = []

source_suffix = {
    ".rst": "restructuredtext",
}

master_doc = "index"
language = "en"

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "navigation_depth": 3,
}

html_static_path = ["_static"]
html_logo = "_static/logo.png"


# -- Extension configuration -------------------------------------------------

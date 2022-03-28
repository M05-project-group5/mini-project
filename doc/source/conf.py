# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

# -- Add current project where running from
sys.path.append(os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = 'Mini-Project'
copyright = '2022, CA, MC'
author = 'CA, MC'

# The full version, including alpha/beta/rc tags
release = '2021-02-22'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

import sphinx_rtd_theme

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.ifconfig",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.graphviz",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_rtd_theme",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "1.3"

nitpicky = False

# Ignores stuff we can't easily resolve on other project's sphinx manuals
nitpick_ignore = []

# Allows the user to override warnings from a separate file
if os.path.exists("nitpick-exceptions.txt"):
    for line in open("nitpick-exceptions.txt"):
        if line.strip() == "" or line.startswith("#"):
            continue
        dtype, target = line.split(None, 1)
        target = target.strip()
        try:  # python 2.x
            target = unicode(target)
        except NameError:
            pass
        nitpick_ignore.append((dtype, target))

# Always includes todos
todo_include_todos = True

# Generates auto-summary automatically
autosummary_generate = True

# Create numbers on figures with captions
numfig = True

# If we are on OSX, the 'dvipng' path maybe different
dvipng_osx = "/opt/local/libexec/texlive/binaries/dvipng"
if os.path.exists(dvipng_osx):
    pngmath_dvipng = dvipng_osx

# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Output file base name for HTML help builder.
htmlhelp_basename = "_doc"


# -- Post configuration --------------------------------------------------------

# Default processing flags for sphinx
autoclass_content = "class"
autodoc_member_order = "bysource"
autodoc_default_flags = [
    "members",
    "undoc-members",
    "show-inheritance",
]

intersphinx_mapping = dict(
    python=('https://docs.python.org/3', None),
    numpy=("https://numpy.org/doc/stable/", None),
    scipy=("https://docs.scipy.org/doc/scipy/reference", None),
)

# -- Extension configuration -------------------------------------------------
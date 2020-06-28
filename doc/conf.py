import os
import sys

sys.path.insert(0, os.path.abspath(".."))
import stan

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = ".rst"
master_doc = "index"
project = u"pystan"
copyright = u"2019, pystan Developers"


intersphinx_mapping = {
    "python": ("http://python.readthedocs.io/en/latest/", None),
}

version = release = stan.__version__

################################################################################
# theme configuration
################################################################################

# on_rtd is whether we are on readthedocs.org
on_rtd = os.environ.get("READTHEDOCS", None) == "True"

if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme

    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

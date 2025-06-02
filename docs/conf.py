# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import shutil
import sys

sys.path.append("../src")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "automated-scoring"
copyright = "2025, Paul Nuehrenberg"
author = "Paul Nuehrenberg"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "jupyter_sphinx",
    "nbsphinx",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_default_options = {"ignore-module-all": True}
autodoc_mock_imports = ["loguru"]

add_module_names = False
autodoc_typehints = "description"
typehints_defaults = "comma"
always_use_bars_union = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "numba": ("https://numba.readthedocs.io/en/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "optuna": ("https://optuna.readthedocs.io/en/stable", None),
    "mpi4py": ("https://mpi4py.readthedocs.io/en/stable", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "shibuya"
html_static_path = ["_static"]


def crawl_source_shorten_titles(path):
    # see https://groups.google.com/g/sphinx-users/c/x590XdgOx1M
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if os.path.isdir(file_path):
            crawl_source_shorten_titles(file_path)
        else:
            _, extension = os.path.splitext(file_path)
            if extension == ".rst":
                with open(file_path, "r") as file:
                    lines = file.readlines()
                if "=" not in lines[1]:
                    # not a title
                    continue
                lines[0] = (
                    lines[0]
                    .split(".")[-1]
                    .replace("\\_", " ")
                    .replace("package", "")
                    .replace("module", "")
                )
                lines[1] = ("=" * (len(lines[0]) - 1)) + "\n"
                with open(file_path, "w") as file:
                    file.writelines(lines)


crawl_source_shorten_titles("source")


shutil.copy2("../examples/CALMS21/minimal_example.ipynb", "source")
shutil.copy2("../examples/CALMS21/results_and_figures.ipynb", "source")
shutil.copy2("../examples/CALMS21/postprocessing_parameters.ipynb", "source")

# sphinx-apidoc -f -e -o source/ ../src/vassi && make clean && make html

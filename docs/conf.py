# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from datetime import date

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import mapc_sim

project = 'mapc_sim'
copyright = (f'{date.today().year}, Maksymilian Wojnar,Wojciech Ciężobka,'
             f'Katarzyna Kosek-Szott,Krzysztof Rusek,Szymon Szott')
author = ('Maksymilian Wojnar,Wojciech Ciężobka,Katarzyna Kosek-Szott,'
          'Krzysztof Rusek,Szymon Szott')
version = 'latest'

sys.path.insert(0, os.path.abspath('..'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.autodoc.typehints',
              'sphinx.ext.intersphinx',
              'sphinx.ext.mathjax',
              'sphinx.ext.napoleon',
              ]

templates_path = ['_templates']
exclude_patterns = []
napoleon_preprocess_types=True

#html_theme = 'sphinx_rtd_theme'
#html_theme = 'sphinx_book_theme'
github_url = "https://github.com/ml4wifi-devs/mapc-sim"

html_static_path = ['_static']
source_suffix = ['.rst', '.md', '.ipynb']

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"


html_theme_options = {
    'show_powered_by': False,
    'github_user': 'ml4wifi-devs',
    'github_repo': 'mapc-sim',
    'github_banner': True,
    'show_related': False,
    'note_bg': '#FFF59C'
}
# sphinx-build -b html docs docs/build/html
#  open docs/build/html/index.html
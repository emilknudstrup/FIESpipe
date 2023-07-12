# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('./../../src'))
sys.path.insert(0, os.path.abspath('./../../examples/basic'))
sys.path.insert(0, os.path.abspath('./../../examples/broad'))
sys.path.insert(0, os.path.abspath('./../../examples/chisq'))
sys.path.insert(0, os.path.abspath('./../../examples/tempmatch'))
sys.path.insert(0, os.path.abspath('./../../examples/sidx'))
sys.path.insert(0, os.path.abspath('./../../examples/thar'))

# -- Project information -----------------------------------------------------

project = 'FIESpipe'
copyright = '2023, Emil Knudstrup, Marcus Marcussen, Hans Kjeldsen, Rasmus Handberg'
author = 'Emil Knudstrup, Marcus Marcussen, Hans Kjeldsen, Rasmus Handberg'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
'sphinx.ext.autodoc', 
'sphinx.ext.coverage', 
'sphinx.ext.todo', 
'sphinx.ext.napoleon',
'nbsphinx',
'sphinxcontrib.bibtex',
'sphinx.ext.graphviz'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# graphviz config
#graphviz_output_format = "svg"


# Bibliography
bibtex_bibfiles = ['refs.bib']
from dataclasses import dataclass, field
import sphinxcontrib.bibtex.plugin

from sphinxcontrib.bibtex.style.referencing import BracketStyle
from sphinxcontrib.bibtex.style.referencing.author_year \
    import AuthorYearReferenceStyle


def bracket_style() -> BracketStyle:
    return BracketStyle(
        left='(',
        right=')',
    )


@dataclass
class MyReferenceStyle(AuthorYearReferenceStyle):
    bracket_parenthetical: BracketStyle = field(default_factory=bracket_style)
    bracket_textual: BracketStyle = field(default_factory=bracket_style)
    bracket_author: BracketStyle = field(default_factory=bracket_style)
    bracket_label: BracketStyle = field(default_factory=bracket_style)
    bracket_year: BracketStyle = field(default_factory=bracket_style)


sphinxcontrib.bibtex.plugin.register_plugin(
    'sphinxcontrib.bibtex.style.referencing',
    'author_year_round', MyReferenceStyle)
    
   
bibtex_reference_style = 'author_year_round'

# Workaround to install and execute git-lfs on Read the Docs
# Needed to display images and gif
import os
if not os.path.exists('./git-lfs'):
    os.system('wget https://github.com/git-lfs/git-lfs/releases/download/v2.7.1/git-lfs-linux-amd64-v2.7.1.tar.gz')
    os.system('tar xvfz git-lfs-linux-amd64-v2.7.1.tar.gz')
    os.system('./git-lfs install')  # make lfs available in current repository
    os.system('./git-lfs fetch')  # download content from remote
    os.system('./git-lfs checkout')  # make local files to have the real content on them

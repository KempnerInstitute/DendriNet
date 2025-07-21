# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Dendritic Modeling Framework'
copyright = '2024, Kempner Institute for the Study of Natural and Artificial Intelligence'
author = 'Kempner Institute Team'
release = '0.1.0'
version = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Core Sphinx extensions
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.githubpages',
    
    # Additional extensions for enhanced documentation
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.graphviz',    # For drawing graphs
    'sphinx_copybutton',      # Copy button for code blocks
    'sphinx_design',          # Enhanced design elements
    # 'myst_parser',            # Markdown support - disabled until myst_parser issue is resolved
    'nbsphinx',              # Jupyter notebook support
]

# Source file parsers - only RST for now
source_suffix = '.rst'

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Autosummary settings
autosummary_generate = True
autosummary_generate_overwrite = True

# Exclude patterns
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'md_backup', 'convert_md_to_rst.py', 'figures/md_backup']

# Napoleon settings for Google/NumPy docstring support
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

# Math settings
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
    }
}

# MyST parser settings - disabled
# myst_enable_extensions = [
#     "colon_fence",
#     "deflist",
#     "dollarmath",
#     "fieldlist", 
#     "html_admonition",
#     "html_image",
#     "linkify",
#     "replacements",
#     "smartquotes",
#     "strikethrough",
#     "substitution",
#     "tasklist",
# ]

# NBSphinx settings for Jupyter notebooks
nbsphinx_allow_errors = True
nbsphinx_execute = 'never'  # Don't execute notebooks during build
nbsphinx_kernel_name = 'python3'

# Copy button settings
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True
copybutton_remove_prompts = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# Notebook execution settings
nbsphinx_execute = 'never'  # Don't execute notebooks during build
nbsphinx_allow_errors = True

# Cross-reference configuration
nitpicky = True  # Warn about broken cross-references

# Language and locale
language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Use the modern PyData Sphinx theme
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# Theme options
html_theme_options = {
    "github_url": "https://github.com/KempnerInstitute/dendritic-modeling",
    "twitter_url": "https://twitter.com/KempnerInstitute",
    "search_bar_text": "Search framework docs...",
    "navbar_end": ["search-field", "navbar-icon-links"],
    "show_toc_level": 2,
    "collapse_navigation": False,
    "navigation_depth": 3,
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navbar_align": "left",
    "navbar_center": ["navbar-nav"],
    "search_bar_text": "Search the docs...",
    "navigation_with_keys": True,
    "show_nav_level": 1,
    "collapse_navigation": False,
    "navigation_depth": 4,
    "header_links_before_dropdown": 7,
    
    # Custom footer
    "footer_items": ["copyright", "sphinx-version", "theme-version"],
    
    # Logo and favicon
    "logo": {
        "image_light": "_static/logo_light.png",
        "image_dark": "_static/logo_dark.png",
        "text": "Dendritic Modeling"
    },
    
    # Custom CSS
    "pygments_light_style": "github-light",
    "pygments_dark_style": "github-dark",
}

# Edit page configuration
html_context = {
    "github_user": "KempnerInstitute",
    "github_repo": "dendritic-modeling",
    "github_version": "main",
    "doc_path": "docs",
}

# Custom CSS files
html_css_files = [
    'custom.css',
]

# Custom JavaScript files
html_js_files = [
    'custom.js',
]

# Additional HTML settings
html_title = f"{project} v{version}"
html_short_title = "Dendritic Modeling"
html_favicon = '_static/favicon.ico'
html_last_updated_fmt = '%b %d, %Y'
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

# Sidebar settings
html_sidebars = {
    "**": ["search-field", "sidebar-nav-bs"],
    "index": [],
}

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': r'''
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{graphicx}
''',
}

latex_documents = [
    ('index', 'dendritic-modeling.tex', 'Dendritic Modeling Framework Documentation',
     'Kempner Institute Team', 'manual'),
]

# -- Options for EPUB output -------------------------------------------------

epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_exclude_files = ['search.html']

# -- Extension configuration -------------------------------------------------

# Todo extension
todo_include_todos = True
todo_emit_warnings = True

# Coverage extension
coverage_skip_undoc_in_source = True
coverage_ignore_modules = [
    "dendritic_modeling.tests",
]

# Doctest extension
doctest_global_setup = '''
import numpy as np
import torch
import dendritic_modeling
'''

# -- Custom configuration ----------------------------------------------------

# Custom roles and directives
def setup(app):
    """Custom Sphinx setup function."""
    # Add custom CSS
    app.add_css_file('custom.css')
    
    # Add custom directives if needed
    pass

# Source suffix - RST only
# source_suffix = {
#     '.rst': None,
#     '.md': 'myst_parser',
# }

# Master document
master_doc = 'index'

# Suppress warnings for external links
suppress_warnings = ['image.nonlocal_uri']

# Figure numbering
numfig = True
numfig_format = {
    'figure': 'Figure %s',
    'table': 'Table %s',
    'code-block': 'Code Block %s'
}

# Cross-reference settings
default_role = None

# Linkcheck settings
linkcheck_ignore = [
    r'http://localhost:\d+/',
    r'https://github.com/.*/blob/.*',  # Skip blob links that may be transient
]

# -- Custom processing -------------------------------------------------------

def skip_member(app, what, name, obj, skip, options):
    """Custom function to skip certain members during autodoc."""
    # Skip private methods and attributes starting with underscore
    # unless they are special methods like __init__
    if name.startswith('_') and not name.startswith('__'):
        return True
    return skip

def setup_autodoc(app):
    """Set up autodoc customizations."""
    app.connect('autodoc-skip-member', skip_member)

# Connect the setup function
def setup(app):
    setup_autodoc(app)
    app.add_css_file('custom.css')

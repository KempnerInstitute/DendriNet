"""
Configuration file for the Sphinx documentation builder.
"""

import os
import sys
from datetime import datetime

# Add the source directory to the path so that autodoc can find the modules
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
project = 'Dendritic Modeling'
copyright = 'MIT License'
author = 'Kempner Institute at Harvard University'

# The full version, including alpha/beta/rc tags
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'sphinx_rtd_theme',
    'sphinx.ext.githubpages',
    'sphinxcontrib.bibtex',
]

# Configure bibtex extension
bibtex_bibfiles = ['refs.bib']
bibtex_default_style = 'unsrt'

# Configure autosectionlabel extension
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2

# Configure intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static', 'source/_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'css/custom.css',
]

# -- Napoleon settings -------------------------------------------------------
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

# -- Setup for execution before the build ------------------------------------
def setup(app):
    try:
        # Try to run the script to generate placeholder images
        import subprocess
        import sys
        
        script_path = os.path.join(os.path.dirname(__file__), 'make.py')
        if os.path.exists(script_path):
            print("Generating placeholder images...")
            subprocess.run([sys.executable, script_path], check=True)
            print("Placeholder images generated.")
        else:
            print(f"Warning: Could not find {script_path}")
    except Exception as e:
        print(f"Warning: Failed to generate placeholder images: {e}")
        
    # Create _static/css directory if it doesn't exist
    css_dir = os.path.join(os.path.dirname(__file__), '_static', 'css')
    os.makedirs(css_dir, exist_ok=True)
    
    # Create a custom CSS file
    css_path = os.path.join(css_dir, 'custom.css')
    with open(css_path, 'w') as f:
        f.write("""
/* Custom styles for dendritic-modeling docs */
.wy-nav-content {
    max-width: 1200px;
}

/* Improve code block styling */
.highlight {
    background: #f8f8f8;
}

/* Improve table styling */
table.docutils {
    width: 100%;
    margin-bottom: 1em;
}

/* Better spacing for parameters */
.field-list dt {
    padding-top: 8px;
}
        """)

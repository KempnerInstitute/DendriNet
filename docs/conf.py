# Configuration file for the Sphinx documentation builder.

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The API reference documents the source tree next to these pages, not an
# installed copy of the package.
sys.path.insert(0, str(ROOT / "src"))

# -- Project information -----------------------------------------------------
project = "DendriNet"
copyright = "2024-2026, Kempner Institute"
author = "Kempner Institute Team"
release = re.search(
    r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.MULTILINE
).group(1)

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = f"{project} v{release}"

# -- Extension configuration -------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from sagari import __version__  # noqa: E402

# the example notebooks live in examples/notebooks; copy them in so Sphinx can render them (already executed)
shutil.rmtree(HERE / "notebooks", ignore_errors=True)
shutil.copytree(HERE.parent / "examples" / "notebooks", HERE / "notebooks",
                ignore=shutil.ignore_patterns("*_campaign", "cluster_campaign", ".ipynb_checkpoints"))

project = "SAGARI"
author = "Thomas K. Waters"
copyright = "2026, Thomas K. Waters"
release = version = __version__
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode", "sphinx.ext.mathjax",
              "myst_nb", "sphinx_copybutton"]
nb_execution_mode = "off"            # the notebooks are stored executed
myst_enable_extensions = ["dollarmath", "colon_fence", "deflist"]
myst_heading_anchors = 3
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
html_theme = "sphinx_rtd_theme"
html_title = f"SAGARI {version}"
exclude_patterns = ["_build", "notebooks/*_campaign", "notebooks/cluster_campaign"]
suppress_warnings = ["mystnb.unknown_mime_type"]

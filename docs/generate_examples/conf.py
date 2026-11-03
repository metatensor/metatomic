# Pseudo-sphinx configuration to run sphinx-gallery as a command line tool

import os

from chemiscope.sphinx import ChemiscopeScraper
from sphinx_gallery.sorting import FileNameSortKey


extensions = [
    "sphinx_gallery.gen_gallery",
]

HERE = os.path.dirname(__file__)
ROOT = os.path.realpath(os.path.join(HERE, "..", ".."))

EXAMPLES = ["c", "torch", "jax", "ase", "torchsim"]

try:
    # JaxModel needs a metatensor-core which can store JAX arrays and use TensorMap
    # with JAX transformations (metatensor/metatensor#1207), which is not released yet.
    # Without it, the JAX tutorials are rendered without being executed.
    import metatensor._jax  # noqa: F401

    RUN_JAX_EXAMPLES = True
except ImportError:
    RUN_JAX_EXAMPLES = False

sphinx_gallery_conf = {
    "filename_pattern": r"\.py" if RUN_JAX_EXAMPLES else r"^(?!.*[/\\]jax[/\\]).*\.py",
    "copyfile_regex": r".*\.(example|mts|xyz)",
    "examples_dirs": [os.path.join(ROOT, "examples", e) for e in EXAMPLES],
    "gallery_dirs": [
        os.path.join(ROOT, "docs", "src", "examples", e) for e in EXAMPLES
    ],
    "example_extensions": {".py", ".c", ".cpp"},
    "matplotlib_animations": False,
    "image_scrapers": ("matplotlib", ChemiscopeScraper()),
    "remove_config_comments": True,
    "within_subsection_order": FileNameSortKey,
    "default_thumb_file": os.path.join(ROOT, "docs", "logo", "metatomic-512.png"),
    "reference_url": {"metatomic": None},
}

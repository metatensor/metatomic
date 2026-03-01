"""Sphinx configuration for metatomic-torchsim documentation."""

from datetime import datetime


project = "metatomic-torchsim"
author = "the metatomic developers"
copyright = f"{datetime.now().date().year}, {author}"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "autoapi.extension",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
suppress_warnings = ["autoapi.python_import_resolution"]

# -- AutoAPI -----------------------------------------------------------------

autoapi_dirs = ["../metatomic/torchsim"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]
autoapi_python_class_content = "both"
autoapi_member_order = "bysource"
autoapi_keep_files = False
autoapi_add_toctree_entry = False

# -- MyST --------------------------------------------------------------------

myst_enable_extensions = ["colon_fence", "fieldlist"]

# -- Autodoc mocking ---------------------------------------------------------

autodoc_mock_imports = [
    "torch",
    "torch_sim",
    "metatensor",
    "metatomic",
    "vesin",
    "nvalchemiops",
]

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "metatomic": ("https://docs.metatensor.org/metatomic/latest/", None),
    "metatensor": ("https://docs.metatensor.org/latest/", None),
    "ase": ("https://ase-lib.org/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "shibuya"
html_static_path = ["_static"]

html_context = {
    "source_type": "github",
    "source_user": "metatensor",
    "source_repo": "metatomic",
    "source_version": "main",
    "source_docs_path": "/python/metatomic_torchsim/docs/",
}

html_theme_options = {
    "github_url": "https://github.com/metatensor/metatomic",
    "accent_color": "teal",
    "dark_code": True,
    "globaltoc_expand_depth": 1,
    "nav_links": [
        {
            "title": "Ecosystem",
            "children": [
                {
                    "title": "metatensor",
                    "url": "https://docs.metatensor.org",
                    "summary": "Data storage for atomistic ML",
                    "external": True,
                },
                {
                    "title": "metatomic",
                    "url": "https://docs.metatensor.org/metatomic/latest/",
                    "summary": "Atomistic models with metatensor",
                    "external": True,
                },
                {
                    "title": "metatrain",
                    "url": "https://metatensor.github.io/metatrain/latest/",
                    "summary": "Training framework for atomistic ML",
                    "external": True,
                },
                {
                    "title": "torch-sim",
                    "url": "https://torchsim.github.io/torch-sim/",
                    "summary": "Differentiable molecular dynamics in PyTorch",
                    "external": True,
                },
            ],
        },
        {
            "title": "PyPI",
            "url": "https://pypi.org/project/metatomic-torchsim/",
            "external": True,
        },
    ],
}

html_sidebars = {
    "**": [
        "sidebars/localtoc.html",
        "sidebars/repo-stats.html",
        "sidebars/edit-this-page.html",
    ],
}

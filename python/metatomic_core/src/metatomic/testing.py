"""Helpers for testing simulation-engine integrations with metatomic."""

import os
import sys

from . import utils as _utils


def plugins_directory():
    """Directory containing installed metatomic test plugins.

    On Unix this is ``<prefix>/libexec/metatomic``; on Windows
    ``<prefix>/bin`` (next to ``metatomic.dll``).
    """
    if sys.platform.startswith("win"):
        return os.path.join(_utils._installation_prefix, "bin")
    return os.path.join(_utils._installation_prefix, "libexec", "metatomic")


def plugin_path(name):
    """Absolute path of an installed metatomic test plugin.

    ``name`` is the plugin library stem (for example ``"lj-plugin"``). Plugins
    are shared libraries named ``{name}.so`` on every platform.

    Simulation engines can resolve the path in their own test suites::

        python -c "import metatomic; print(metatomic.testing.plugin_path('lj-plugin'))"

    Load it with :c:func:`mta_load_plugin`, then call :c:func:`mta_load_model`
    with the model name and plugin name documented for that plugin.
    """
    path = os.path.join(plugins_directory(), f"{name}.so")
    if os.path.isfile(path):
        return path

    raise FileNotFoundError(
        f"metatomic test plugin '{name}' not found at '{path}'; reinstall "
        "metatomic-core to restore it."
    )

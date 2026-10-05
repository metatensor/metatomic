"""Helpers for testing simulation-engine integrations with metatomic."""

import os
import sys


_HERE = os.path.dirname(os.path.abspath(__file__))


def _install_prefix():
    """Install prefix that contains the test plugins.

    An external library records ``EXTERNAL_METATOMIC_PREFIX``. A bundled build
    installs plugins inside this package.
    """
    try:
        from ._external import EXTERNAL_METATOMIC_PREFIX
    except ImportError:
        return _HERE
    return EXTERNAL_METATOMIC_PREFIX


def plugins_directory():
    """Directory containing installed metatomic test plugins"""
    if sys.platform == "win32":
        return os.path.join(_install_prefix(), "bin")
    elif sys.platform.startswith("linux") or sys.platform == "darwin":
        return os.path.join(_install_prefix(), "libexec", "metatomic")
    else:
        raise RuntimeError(f"unsupported platform '{sys.platform}'")


_PLUGINS_FILES = {
    "lj": "metatomic-lj-plugin.so",
}


def plugin_path(name):
    """Absolute path of an installed metatomic test plugin.

    ``name`` is the plugin name, e.g. ``"lj"`` for the Lennard-Jones plugin.

    Simulation engines can resolve the path in their own test suites::

        python -c "import metatomic; print(metatomic.testing.plugin_path('lj'))"

    Load it with :c:func:`mta_load_plugin`, then call :c:func:`mta_load_model`
    with the model name and plugin name documented for that plugin.
    """
    if name not in _PLUGINS_FILES:
        raise ValueError(f"Unknown plugin name: '{name}'")

    path = os.path.join(plugins_directory(), _PLUGINS_FILES[name])
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"metatomic test plugin '{name}' not found at '{path}'; reinstall "
            "metatomic-core to restore it."
        )

    return path

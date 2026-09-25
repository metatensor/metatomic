"""Helpers for testing simulation-engine integrations with metatomic."""

import os

from . import utils as _utils


def _plugins_subdir():
    """Install subdirectory of test plugins, relative to the metatomic prefix.

    Comes from the CMake install layout (``CMAKE_INSTALL_BINDIR`` on Windows,
    ``CMAKE_INSTALL_LIBEXECDIR/metatomic`` elsewhere), recorded at build time.
    """
    try:
        from ._plugins import PLUGINS_SUBDIR

        return PLUGINS_SUBDIR
    except ImportError:
        pass

    try:
        from ._external import EXTERNAL_METATOMIC_PLUGINS_SUBDIR

        return EXTERNAL_METATOMIC_PLUGINS_SUBDIR
    except ImportError:
        pass

    raise FileNotFoundError(
        "metatomic test plugin install path is unknown; reinstall "
        "metatomic-core to restore generated install metadata."
    )


def plugins_directory():
    """Directory containing installed metatomic test plugins.

    The location follows the CMake install layout used when metatomic was
    built (typically ``<prefix>/<libexecdir>/metatomic`` on Unix, or
    ``<prefix>/<bindir>`` on Windows).
    """
    return os.path.join(_utils._installation_prefix, _plugins_subdir())


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

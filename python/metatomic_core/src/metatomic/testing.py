"""Helpers for testing simulation-engine integrations with metatomic."""

import os


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


def _plugins_subdir():
    """Install subdirectory of test plugins, relative to the metatomic prefix.

    Comes from the CMake install layout (``CMAKE_INSTALL_BINDIR`` on Windows,
    ``CMAKE_INSTALL_LIBEXECDIR/metatomic`` elsewhere), recorded at build time.
    """
    try:
        from ._external import EXTERNAL_METATOMIC_PLUGINS_SUBDIR
    except ImportError as error:
        raise FileNotFoundError(
            "metatomic test plugin install path is unknown; reinstall "
            "metatomic-core to restore generated install metadata."
        ) from error
    return EXTERNAL_METATOMIC_PLUGINS_SUBDIR


def plugins_directory():
    """Directory containing installed metatomic test plugins.

    The location follows the CMake install layout used when metatomic was
    built (typically ``<prefix>/<libexecdir>/metatomic`` on Unix, or
    ``<prefix>/<bindir>`` on Windows).
    """
    return os.path.join(_install_prefix(), _plugins_subdir())


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

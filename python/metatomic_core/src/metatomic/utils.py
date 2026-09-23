import os
import sys


_HERE = os.path.dirname(os.path.abspath(__file__))


try:
    from ._external import EXTERNAL_METATOMIC_PREFIX

    cmake_prefix_path = EXTERNAL_METATOMIC_PREFIX
    """
    Path containing the CMake configuration files for the underlying C library
    """
    _installation_prefix = EXTERNAL_METATOMIC_PREFIX

except ImportError:
    cmake_prefix_path = os.path.join(_HERE, "lib", "cmake")
    """
    Path containing the CMake configuration files for the underlying C library
    """
    _installation_prefix = _HERE


def lj_plugin_path():
    """Absolute path of the shifted Lennard-Jones test plugin.

    Simulation engines can load this shared library in their own test suites::

        python -c "import metatomic; print(metatomic.utils.lj_plugin_path())"

    Load it with :c:func:`mta_load_plugin`, then
    ``mta_load_model("lennard-jones", options, "lj-plugin")``.

    The options are a JSON object. ``sigma``, ``epsilon``, and ``cutoff`` may
    be numbers or strings; ``atomic_type`` may be an integer array or a
    comma-separated string. Supported keys are ``sigma``, ``epsilon``,
    ``cutoff``, ``atomic_type``, ``length_unit``, and ``energy_unit``.
    """
    # Mirrors lj-plugin/CMakeLists.txt (libexec on Unix, bin/ on Windows).
    if sys.platform.startswith("win"):
        path = os.path.join(_installation_prefix, "bin", "lj-plugin.so")
    else:
        path = os.path.join(
            _installation_prefix, "libexec", "metatomic", "lj-plugin.so"
        )
    if os.path.isfile(path):
        return path

    raise FileNotFoundError(
        f"Lennard-Jones test plugin not found at '{path}'; reinstall "
        "metatomic-core to restore it."
    )

import ctypes
import os
import sys


_HERE = os.path.dirname(os.path.abspath(__file__))


def unit_conversion_factor(from_unit: str, to_unit: str) -> float:
    """Return the conversion factor between compatible units.

    Unit expressions follow :ref:`units`. If either unit is empty, return 1.0.
    Incompatible or unknown units raise :py:class:`metatomic.MetatomicError`.
    """
    from ._c_lib import _get_library

    factor = ctypes.c_double()
    _get_library().mta_unit_conversion_factor(
        from_unit.encode("utf8"), to_unit.encode("utf8"), ctypes.byref(factor)
    )
    return factor.value


try:
    from ._external import EXTERNAL_METATOMIC_PREFIX

    cmake_prefix_path = EXTERNAL_METATOMIC_PREFIX
    """
    Path containing the CMake configuration files for the underlying C library
    """

except ImportError:
    cmake_prefix_path = os.path.join(_HERE, "lib", "cmake")
    """
    Path containing the CMake configuration files for the underlying C library
    """


if sys.platform == "win32":
    python_plugin_path = os.path.join(_HERE, "bin", "metatomic-python-plugin.so")
else:
    python_plugin_path = os.path.join(
        _HERE, "libexec", "metatomic", "metatomic-python-plugin.so"
    )
"""
Path to the metatomic plugin loading models defined in Python scripts. Load it
with :c:func:`mta_load_plugin`, then load a Python script defining a ``model``
variable with :c:func:`mta_load_model`.
"""

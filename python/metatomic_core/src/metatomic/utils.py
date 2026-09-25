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

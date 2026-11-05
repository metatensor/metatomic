import importlib
import os
import re
import sys
from collections import namedtuple

import torch


Version = namedtuple("Version", ["major", "minor", "patch"])


def parse_version(version):
    match = re.match(r"(\d+)\.(\d+)\.(\d+).*", version)
    if match:
        return Version(*map(int, match.groups()))
    else:
        raise ValueError("Invalid version string format")


def version_compatible(actual, required):
    actual = parse_version(actual)
    required = parse_version(required)

    if actual.major != required.major:
        return False
    elif actual.minor != required.minor:
        return False
    else:
        return True


_HERE = os.path.dirname(__file__)
_TORCH_VERSION = parse_version(torch.__version__)
install_prefix = os.path.join(
    _HERE, f"torch-{_TORCH_VERSION.major}.{_TORCH_VERSION.minor}"
)


external_path = os.path.join(install_prefix, "_external.py")
if os.path.exists(external_path):
    spec = importlib.util.spec_from_file_location("_external", external_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _prefix = module.EXTERNAL_METATOMIC_TORCH_PREFIX
    cmake_prefix_path = _prefix
    """
    Path containing the CMake configuration files for the underlying C++ library
    """
else:
    _prefix = install_prefix
    cmake_prefix_path = os.path.join(install_prefix, "lib", "cmake")
    """
    Path containing the CMake configuration files for the underlying C++ library
    """

if sys.platform == "win32":
    plugin_path = os.path.join(_prefix, "bin", "metatomic-torch-plugin.so")
else:
    plugin_path = os.path.join(
        _prefix, "libexec", "metatomic", "metatomic-torch-plugin.so"
    )
"""
Path to the metatomic plugin loading models exported with
:py:class:`metatomic.torch.AtomisticModel`. Load it with
:py:func:`metatomic.load_plugin`, then load the exported model with
:py:func:`metatomic.load_model`.
"""

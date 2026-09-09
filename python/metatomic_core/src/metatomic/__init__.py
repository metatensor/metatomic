# The metatomic python package is distributed in multiple pieces, each with its own
# version number and dependency management:
#
# - the `metatomic-core` distribution contains the Python bindings to the
#   metatomic-core C API in the `metatomic` python package.
# - the `metatomic-torch` distribution contains the TorchScript bindings to the C API.

from . import utils  # noqa: F401
from ._status import MetatomicError
from ._version import __version__  # noqa: F401


# pretend the classes are defined in the top-level module for better error messages
MetatomicError.__module__ = __name__

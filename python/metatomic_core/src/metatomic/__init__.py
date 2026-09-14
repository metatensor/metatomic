from . import utils  # noqa: F401
from ._capabilities import ModelCapabilities
from ._metadata import ModelMetadata, References
from ._quantity import Quantity
from ._status import MetatomicError
from ._system import PairListOptions
from ._version import __version__  # noqa: F401


# pretend the classes are defined in the top-level module for better error messages
MetatomicError.__module__ = __name__
ModelCapabilities.__module__ = __name__
ModelMetadata.__module__ = __name__
PairListOptions.__module__ = __name__
Quantity.__module__ = __name__
References.__module__ = __name__

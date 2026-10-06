# ruff: noqa: F401

from . import (
    testing,
    utils,
)
from ._capabilities import ModelCapabilities
from ._metadata import ModelMetadata, References
from ._model import BaseModel, ExternalModel, execute_model
from ._quantity import Quantity
from ._status import MetatomicError
from ._system import PairListOptions, System
from ._version import __version__


# pretend the classes are defined in the top-level module for better error messages
BaseModel.__module__ = __name__
ExternalModel.__module__ = __name__
MetatomicError.__module__ = __name__
ModelCapabilities.__module__ = __name__
ModelMetadata.__module__ = __name__
PairListOptions.__module__ = __name__
System.__module__ = __name__
Quantity.__module__ = __name__
References.__module__ = __name__

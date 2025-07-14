import importlib
import os
import sys

import torch

from . import utils  # noqa: F401
from ._c_lib import _load_library
from .version import __version__  # noqa: F401


if os.environ.get("METATOMIC_IMPORT_FOR_SPHINX", "0") != "0":
    from .documentation import (
        ModelCapabilities,
        ModelEvaluationOptions,
        ModelMetadata,
        ModelOutput,
        NeighborListOptions,
        System,
        check_atomistic_model,
        load_model_extensions,
        read_model_metadata,
        register_autograd_neighbors,
        unit_conversion_factor,
    )

else:
    _load_library()

    System = torch.classes.metatomic.System
    NeighborListOptions = torch.classes.metatomic.NeighborListOptions

    ModelOutput = torch.classes.metatomic.ModelOutput
    ModelEvaluationOptions = torch.classes.metatomic.ModelEvaluationOptions
    ModelCapabilities = torch.classes.metatomic.ModelCapabilities
    ModelMetadata = torch.classes.metatomic.ModelMetadata

    read_model_metadata = torch.ops.metatomic.read_model_metadata
    load_model_extensions = torch.ops.metatomic.load_model_extensions
    check_atomistic_model = torch.ops.metatomic.check_atomistic_model

    register_autograd_neighbors = torch.ops.metatomic.register_autograd_neighbors
    unit_conversion_factor = torch.ops.metatomic.unit_conversion_factor

from .io import load_system, save  # noqa: F401
from .model import (  # noqa: F401
    AtomisticModel,
    ModelInterface,
    is_atomistic_model,
    load_atomistic_model,
)
from .systems_to_torch import systems_to_torch  # noqa: F401


# Define the attributes to be lazily imported.
# XXX(rg): This is a stripped version of lazy-loader [1,2]
#          Consider growing a dependency on it
# [1]: https://scientific-python.org/specs/spec-0001/
# [2]: https://pypi.org/project/lazy-loader/
# The key is the name exposed to the user.
# The value is a tuple: (relative_module_path, attribute_name_in_module)
# If attribute_name_in_module is None, the entire module is returned.
_lazy_imports = {
    # Exposes `metatomic.torch.ase_calculator` as `metatomic.torch.ase`
    "ase": (".ase_calculator", None),
    # Exposes `MetatomicCalculator` from the submodule as `MetatomicAseCalculator`
    "MetatomicAseCalculator": (".ase_calculator", "MetatomicCalculator"),
}

# The public API of this module includes the lazy-loaded names.
# This is used by `help()` and introspecting tools.
__all__ = list(_lazy_imports.keys())


def __getattr__(name: str):
    """
    Lazily import attributes upon first access.

    This function is called by the Python interpreter when an attribute is
    accessed on this module that doesn't already exist.
    """
    if name in _lazy_imports:
        # Get the import details from our mapping
        module_path, attribute_name = _lazy_imports[name]

        # Perform the actual import (lazy)
        module = importlib.import_module(module_path, __name__)

        if attribute_name is None:
            # For the "ase" key, we want the entire module
            value = module
        else:
            # For "MetatomicAseCalculator", get the class from the module
            value = getattr(module, attribute_name)

        # VERY IMPORTANT: Cache the result in the module's dictionary.
        # This ensures __getattr__ is only called once for this attribute.
        # Subsequent access will be fast and hit the cached value directly.
        # Better than mucking with globals()!!
        setattr(sys.modules[__name__], name, value)

        return value

    # For any other attribute, raise the standard error.
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """
    Expose the lazy-loaded attributes to `dir()` and IDEs.

    This helps with code completion and introspection.
    """
    return __all__

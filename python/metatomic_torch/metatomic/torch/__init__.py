import os

import torch

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


def __getattr__(name):
    # lazy import for ase_calculator, making it accessible as
    # ``metatomic.torch.ase_calculator`` without requiring a separate import from
    # ``metatomic.torch``, but only importing the code when actually required.
    if name == "ase_calculator":
        import metatomic.torch.ase_calculator

        return metatomic.torch.ase_calculator
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

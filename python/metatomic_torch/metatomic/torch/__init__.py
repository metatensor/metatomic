import os
from typing import TYPE_CHECKING

import torch

from ._c_lib import _load_library
from .version import __version__  # noqa: F401


if os.environ.get("METATOMIC_IMPORT_FOR_SPHINX", "0") != "0" or TYPE_CHECKING:
    from .documentation import (
        ModelCapabilities,
        ModelEvaluationOptions,
        ModelMetadata,
        ModelOutput,
        NeighborListOptions,
        System,
        check_atomistic_model,
        load_model_extensions,
        pick_device,
        pick_output,
        read_model_metadata,
        register_autograd_neighbors,
        unit_conversion_factor,
    )

    _check_outputs = None

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
    _check_outputs = torch.ops.metatomic._check_outputs

    register_autograd_neighbors = torch.ops.metatomic.register_autograd_neighbors

    _unit_conversion_factor_v2 = torch.ops.metatomic.unit_conversion_factor_v2
    _unit_conversion_factor_v1 = torch.ops.metatomic.unit_conversion_factor

    def unit_conversion_factor(*args, **kwargs):
        """Unit conversion factor supporting both 2-arg and 3-arg signatures.

        2-arg: ``unit_conversion_factor(from_unit, to_unit)``
        3-arg (deprecated): ``unit_conversion_factor(quantity, from_unit, to_unit)``
        """
        import warnings

        if len(args) == 2 and not kwargs:
            return _unit_conversion_factor_v2(args[0], args[1])
        elif len(args) == 3 and not kwargs:
            warnings.warn(
                "the 3-argument unit_conversion_factor(quantity, from, to) is "
                "deprecated; use the 2-argument form instead",
                DeprecationWarning,
                stacklevel=2,
            )
            return _unit_conversion_factor_v2(args[1], args[2])
        elif "from_unit" in kwargs and "to_unit" in kwargs:
            if "quantity" in kwargs:
                warnings.warn(
                    "the 3-argument unit_conversion_factor(quantity, from, to)"
                    " is deprecated; use the 2-argument form instead",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return _unit_conversion_factor_v2(kwargs["from_unit"], kwargs["to_unit"])
        else:
            raise TypeError(
                "unit_conversion_factor() expects 2 or 3 positional arguments"
            )

    pick_device = torch.ops.metatomic.pick_device
    pick_output = torch.ops.metatomic.pick_output

from .model import (  # noqa: F401
    AtomisticModel,
    ModelInterface,
    is_atomistic_model,
    load_atomistic_model,
)
from .serialization import (  # noqa: F401
    load_system,
    load_system_buffer,
    save,
    save_buffer,
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

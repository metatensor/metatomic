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
        unit_dimension_for_quantity,
    )

    _check_quantities = None

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
    _check_quantities = torch.ops.metatomic._check_quantities

    register_autograd_neighbors = torch.ops.metatomic.register_autograd_neighbors

    unit_conversion_factor = torch.ops.metatomic.unit_conversion_factor
    unit_dimension_for_quantity = torch.ops.metatomic.unit_dimension_for_quantity

    pick_device = torch.ops.metatomic.pick_device
    pick_output = torch.ops.metatomic.pick_output

from . import ase_calculator  # noqa: F401
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

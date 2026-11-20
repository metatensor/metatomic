import os
from typing import TYPE_CHECKING

import torch


if os.environ.get("METATOMIC_IMPORT_FOR_SPHINX", "0") != "0" or TYPE_CHECKING:
    from ._documentation import (
        LegacyModelCapabilities,
        ModelEvaluationOptions,
        ModelMetadata,
        ModelOutput,
        check_atomistic_model,
        load_model_extensions,
        pick_device,
        pick_output,
        read_model_metadata,
        unit_dimension_for_quantity,
    )

    _check_quantities = None

else:
    ModelOutput = torch.classes.metatomic.ModelOutput
    ModelEvaluationOptions = torch.classes.metatomic.ModelEvaluationOptions
    LegacyModelCapabilities = torch.classes.metatomic.ModelCapabilities
    ModelMetadata = torch.classes.metatomic.ModelMetadata

    read_model_metadata = torch.ops.metatomic.read_model_metadata
    load_model_extensions = torch.ops.metatomic.load_model_extensions
    check_atomistic_model = torch.ops.metatomic.check_atomistic_model
    _check_quantities = torch.ops.metatomic._check_quantities

    unit_dimension_for_quantity = torch.ops.metatomic.unit_dimension_for_quantity

    pick_device = torch.ops.metatomic.pick_device
    pick_output = torch.ops.metatomic.pick_output


from ._model import (  # noqa: F401
    AtomisticModel,
    ModelInterface,
    is_atomistic_model,
    load_atomistic_model,
)

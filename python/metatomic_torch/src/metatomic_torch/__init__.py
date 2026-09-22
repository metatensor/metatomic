import importlib.abc
import importlib.util
import os
import sys
from typing import TYPE_CHECKING

import torch

import metatomic


sys.modules["metatomic.torch"] = sys.modules[__name__]
if not hasattr(metatomic, "torch"):
    metatomic.torch = sys.modules[__name__]


class _MetatomicTorchAliasLoader(importlib.abc.Loader):
    def __init__(self, canonical):
        self._canonical = canonical

    def create_module(self, spec):
        return importlib.import_module(self._canonical)

    def exec_module(self, module):
        pass


class _MetatomicTorchAliasFinder(importlib.abc.MetaPathFinder):
    """Resolve ``metatomic.torch.<x>`` to the same module as ``metatomic_torch.<x>``.

    ``metatomic.torch`` is an alias of the ``metatomic_torch`` package, but Python
    otherwise loads submodules under the two names as distinct module objects (it
    names a submodule after the import path, not after the package's ``__name__``).
    This finder makes both namespaces refer to the exact same module object, so e.g.
    ``metatomic.torch.AtomisticModel`` and ``metatomic_torch.model.AtomisticModel``
    are identical rather than two independent copies of the same class.
    """

    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith("metatomic.torch."):
            return None

        canonical = "metatomic_torch" + fullname[len("metatomic.torch") :]
        try:
            canonical_spec = importlib.util.find_spec(canonical)
        except ModuleNotFoundError:
            return None

        if canonical_spec is None:
            return None

        spec = importlib.util.spec_from_loader(
            fullname,
            _MetatomicTorchAliasLoader(canonical),
            origin=canonical_spec.origin,
        )
        if canonical_spec.submodule_search_locations:
            spec.submodule_search_locations = list(
                canonical_spec.submodule_search_locations
            )
        return spec


sys.meta_path.insert(0, _MetatomicTorchAliasFinder())


from ._c_lib import _load_library  # noqa: E402
from .version import __version__  # noqa: F401, E402


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

from . import (  # noqa: F401, E402
    ase_calculator,
    o3,
)
from .model import (  # noqa: F401, E402
    AtomisticModel,
    ModelInterface,
    is_atomistic_model,
    load_atomistic_model,
)
from .serialization import (  # noqa: F401, E402
    load_system,
    load_system_buffer,
    save,
    save_buffer,
)
from .systems_to_torch import systems_to_torch  # noqa: F401, E402

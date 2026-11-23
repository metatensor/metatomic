"""
Run models written for the ``metatomic.torch`` API
(:py:class:`metatomic.torch.AtomisticModel`) with :py:func:`metatomic.execute_model`,
by wrapping them in a :py:class:`metatomic.TorchModel`.
"""

import collections
import ctypes
import os
from typing import Optional

import metatensor as mts
import torch
from metatensor._c_api import mts_labels_t

from metatomic_torch import ModelOutput, is_atomistic_model, load_atomistic_model
from metatomic_torch import System as TorchSystem
from metatomic_torch._quantities import current_quantity_name

from .._capabilities import ModelCapabilities
from .._metadata import ModelMetadata
from .._quantity import Quantity
from .._system import PairListOptions, System
from ._torch import TorchModel


# legacy device names that are spelled differently in metatomic
_RENAMED_DEVICES = {"mps": "metal"}

# metatomic-torch also accepted the deprecated name of these outputs as the name of
# their properties, metatomic-core only accepts the current one
_DEPRECATED_PROPERTIES = {
    "non_conservative_force": "non_conservative_forces",
    "position": "positions",
    "momentum": "momenta",
    "mass": "masses",
    "velocity": "velocities",
}


class LegacyModel(TorchModel):
    """
    Wrap a :py:class:`metatomic.torch.AtomisticModel` (eager or TorchScript, e.g.
    loaded from a ``.pt`` file) to run it with :py:func:`metatomic.execute_model`.

    Systems, pair lists, custom inputs and ``selected_atoms`` are converted to their
    ``metatomic.torch`` / ``metatensor.torch`` equivalent before calling the model,
    and outputs are converted back, keeping the autograd graph so that
    :py:class:`metatomic.TorchModel` can compute positions and strain gradients. The
    ``explicit_gradients`` of the legacy model are declared as explicit gradients, and
    only the remaining ones are computed with autograd.

    Outputs declared with ``sample_kind="atom"`` are also declared with
    ``sample_kind="system"``, since the legacy API allowed requesting both. Outputs
    declared with a deprecated name (e.g. ``non_conservative_forces``) are declared
    with the current name (``non_conservative_force``), and the model is asked for
    the name it declared. Properties using the deprecated name are renamed as well.

    .. code-block:: python

        import metatomic

        model = metatomic.LegacyModel("model.pt", extensions_directory="extensions/")
        energy = metatomic.Quantity(
            name="energy", unit="eV", sample_kind="system", gradients=["positions"]
        )
        # The engine creates systems and attaches the requested pair lists.
        # systems = ...
        # (result,) = metatomic.execute_model(model, systems, [energy])
    """

    def __init__(self, model, extensions_directory: Optional[str] = None):
        """
        :param model: the :py:class:`metatomic.torch.AtomisticModel` to wrap, or a
            path to an exported model
        :param extensions_directory: directory containing TorchScript extensions
            required when loading an exported model
        """
        super().__init__()
        if isinstance(model, (str, os.PathLike)):
            model = load_atomistic_model(os.fspath(model), extensions_directory)

        if not is_atomistic_model(model):
            raise TypeError(f"`model` must be an AtomisticModel, got {type(model)}")

        # Call the inner module directly: execute_model handles unit conversion
        # and consistency checks, which AtomisticModel would repeat.
        self.model = model.module

        # the legacy metadata has the same fields, but is a TorchScript class
        metadata = model.metadata()
        self._metadata = ModelMetadata(
            name=metadata.name,
            description=metadata.description,
            authors=metadata.authors,
            references=metadata.references,
            extra=metadata.extra,
        )

        capabilities = model.capabilities()
        if capabilities.length_unit == "":
            raise ValueError(
                "this model does not declare a length unit, and can not be used "
                "with LegacyModel"
            )

        # AtomisticModel also lists outputs under their deprecated or current names,
        # only keep the names declared by the model, indexed by their current name. A
        # deprecated name is ignored if the model also declares the current one.
        self._legacy_names = {}
        for name in model._model_capabilities_outputs_names:
            current = current_quantity_name(name)
            if current == name or current not in self._legacy_names:
                self._legacy_names[current] = name

        declared = {
            name: capabilities.outputs[legacy_name]
            for name, legacy_name in self._legacy_names.items()
        }
        self._capabilities = ModelCapabilities(
            atomic_types=list(capabilities.atomic_types),
            interaction_range=capabilities.interaction_range,
            length_unit=capabilities.length_unit,
            supported_devices=[
                _RENAMED_DEVICES.get(device, device)
                for device in capabilities.supported_devices
            ],
            dtype=capabilities.dtype,
            outputs=[
                Quantity(
                    name=name,
                    unit=output.unit,
                    sample_kind=sample_kind,
                    description=output.description,
                    gradients=output.explicit_gradients,
                )
                for name, output in declared.items()
                # the legacy API allowed requesting per-atom outputs per-system
                for sample_kind in (
                    ["atom", "system"]
                    if output.sample_kind == "atom"
                    else [output.sample_kind]
                )
            ],
        )

        self._neighbor_lists = model.requested_neighbor_lists()
        self._pair_lists = [
            PairListOptions(
                cutoff=options.cutoff,
                full_list=options.full_list,
                strict=options.strict,
                requestors=options.requestors(),
            )
            for options in self._neighbor_lists
        ]
        self._inputs = [
            Quantity(
                name=name,
                unit=output.unit,
                sample_kind=output.sample_kind,
                description=output.description,
            )
            for name, output in model.requested_inputs(use_new_names=True).items()
        ]

    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def metadata(self) -> ModelMetadata:
        return self._metadata

    def requested_pair_lists(self, requested_outputs) -> list[PairListOptions]:
        return self._pair_lists

    def requested_inputs(self, requested_outputs) -> list[Quantity]:
        return self._inputs

    def _modules_with(self, method: str):
        # Requests were collected from AtomisticModel during initialization.
        # Legacy sub-modules use a different requested_inputs signature, so
        # TorchModel must not collect their requests again.
        yield type(self).__name__, self

    def forward(
        self,
        systems: list[System],
        selected_atoms: Optional[mts.Labels],
        outputs: list[Quantity],
    ) -> list[mts.TensorMap]:
        device = systems[0].positions.device if systems else torch.device("cpu")
        torch_systems = [self._torch_system(system) for system in systems]
        torch_selected = None
        if selected_atoms is not None:
            torch_selected = _convert_labels(selected_atoms, mts.torch, device)

        legacy_names = [self._legacy_names.get(o.name, o.name) for o in outputs]

        # the legacy API is keyed by name, so the same name requested with different
        # sample kinds needs separate calls
        results = [None] * len(outputs)
        for batch in _unique_name_batches(outputs):
            # like AtomisticModel, give the requests to the module as-is: it computes
            # outputs in the declared units, and execute_model converts them
            legacy_outputs = {
                legacy_names[i]: ModelOutput(
                    unit=outputs[i].unit,
                    sample_kind=outputs[i].sample_kind,
                    explicit_gradients=outputs[i].gradients,
                    description=outputs[i].description or "",
                )
                for i in batch
            }
            values = self.model(torch_systems, legacy_outputs, torch_selected)
            for i in batch:
                tensor = _convert_tensor(values[legacy_names[i]], mts)
                results[i] = _current_properties(tensor, outputs[i].name)

        return results

    def _torch_system(self, system: System) -> TorchSystem:
        result = TorchSystem(system.types, system.positions, system.cell, system.pbc)

        for torch_options, options in zip(
            self._neighbor_lists, self._pair_lists, strict=True
        ):
            # the pair vectors are already registered with autograd by TorchModel
            pairs = _convert_block(system.pairs(options), mts.torch)
            result.add_pairs(torch_options, pairs)

        for name in system.known_custom_data():
            result.add_custom_data(
                name, _convert_tensor(system.custom_data(name), mts.torch)
            )

        return result


def _unique_name_batches(outputs: list[Quantity]) -> list[list[int]]:
    """Split the indices of ``outputs`` in batches where all names are unique"""
    counts = collections.Counter()
    batches = collections.defaultdict(list)
    for i, output in enumerate(outputs):
        # the n-th request for a given name goes in the n-th batch
        batches[counts[output.name]].append(i)
        counts[output.name] += 1
    return list(batches.values())


# Conversions between metatensor (core) and metatensor.torch, without copying any
# data. ``mts_module`` is the module to convert to, either ``metatensor`` or
# ``metatensor.torch``.


def _convert_labels(labels, mts_module, device):
    ptr = labels.release()
    if mts_module is mts.torch:
        ptr = ctypes.cast(ptr, ctypes.c_void_p).value
    else:
        ptr = ctypes.cast(ptr, ctypes.POINTER(mts_labels_t))

    labels = mts_module.Labels.unsafe_from_ptr(ptr, array_backend="torch")
    return labels.to(device=device)


def _convert_block(block, mts_module):
    # The two bindings read each other's arrays through DLPack, which does not keep the
    # autograd graph, so blocks are re-created around the same values instead of sharing
    # the C objects.
    device = block.values.device
    result = mts_module.TensorBlock(
        values=block.values,
        samples=_convert_labels(block.samples, mts_module, device),
        components=[_convert_labels(c, mts_module, device) for c in block.components],
        properties=_convert_labels(block.properties, mts_module, device),
    )
    for parameter, gradient in block.gradients():
        result.add_gradient(parameter, _convert_block(gradient, mts_module))
    return result


def _convert_tensor(tensor, mts_module):
    blocks = [_convert_block(block, mts_module) for block in tensor.blocks()]
    device = blocks[0].values.device if blocks else tensor.device
    keys = _convert_labels(tensor.keys, mts_module, device)
    return mts_module.TensorMap(keys, blocks)


def _current_properties(tensor: mts.TensorMap, name: str) -> mts.TensorMap:
    base = name.split("/")[0]
    deprecated = _DEPRECATED_PROPERTIES.get(base)
    if tensor.property_names != [deprecated]:
        return tensor
    return mts.rename_dimension(tensor, "properties", deprecated, base)

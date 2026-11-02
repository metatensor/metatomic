"""
Run models written for the ``metatomic.torch`` API (:py:class:`AtomisticModel`) with
:py:func:`metatomic.execute_model`, by wrapping them in a
:py:class:`metatomic.TorchModel`.
"""

import copy
import ctypes
import os
from typing import Optional

import metatensor as mts
import torch

from metatomic import (
    ModelCapabilities,
    ModelMetadata,
    PairListOptions,
    Quantity,
    System,
    TorchModel,
)

from . import ModelOutput
from . import System as LegacySystem
from .model import is_atomistic_model, load_atomistic_model


# legacy device names that are spelled differently in metatomic
_RENAMED_DEVICES = {"mps": "metal"}


class LegacyModel(TorchModel):
    """
    Wrap an :py:class:`AtomisticModel` (eager or TorchScript, e.g. loaded from a
    ``.pt`` file) to run it with :py:func:`metatomic.execute_model`.

    Systems, pair lists, custom inputs and ``selected_atoms`` are converted to their
    ``metatomic.torch`` / ``metatensor.torch`` equivalent before calling the model,
    and outputs are converted back, keeping the autograd graph so that
    :py:class:`metatomic.TorchModel` can compute positions and strain gradients. The
    ``explicit_gradients`` of the legacy model are declared as explicit gradients, and
    only the remaining ones are computed with autograd.

    Outputs declared with ``sample_kind="atom"`` are also declared with
    ``sample_kind="system"``, since the legacy API allowed requesting both. Deprecated
    names are not supported.

    .. code-block:: python

        import metatomic
        from metatomic_torch import LegacyModel

        model = LegacyModel("model.pt", extensions_directory="extensions/")
        energy = metatomic.Quantity(
            name="energy", unit="eV", sample_kind="system", gradients=["positions"]
        )
        # The engine creates systems and attaches the requested pair lists.
        # systems = ...
        # (result,) = metatomic.execute_model(model, systems, [energy])
    """

    def __init__(self, model, extensions_directory: Optional[str] = None):
        """
        :param model: the :py:class:`AtomisticModel` to wrap, or a path to an
            exported model
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
        self._legacy_metadata = model.metadata()

        capabilities = model.capabilities()
        if capabilities.length_unit == "":
            raise ValueError(
                "this model does not declare a length unit, and can not be used "
                "with LegacyModel"
            )

        # AtomisticModel also lists outputs under their deprecated or current names,
        # only keep the names declared by the model
        self._declared = {
            name: capabilities.outputs[name]
            for name in model._model_capabilities_outputs_names
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
                _quantity(name, output, sample_kind)
                for name, output in self._declared.items()
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
            _quantity(name, output, gradients=[])
            for name, output in model.requested_inputs(use_new_names=True).items()
        ]

    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def metadata(self) -> ModelMetadata:
        metadata = self._legacy_metadata
        return ModelMetadata(
            name=metadata.name,
            description=metadata.description,
            authors=list(metadata.authors),
            references={k: list(v) for k, v in metadata.references.items() if v},
            extra=dict(metadata.extra),
        )

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
        legacy_systems = [self._legacy_system(system) for system in systems]
        legacy_selected = (
            None if selected_atoms is None else _labels_to_torch(selected_atoms, device)
        )

        # the legacy API is keyed by name, so the same name requested with different
        # sample kinds needs separate calls
        results = [None] * len(outputs)
        for batch in _unique_name_batches(outputs):
            legacy_outputs = {
                outputs[i].name: self._legacy_output(outputs[i]) for i in batch
            }
            values = self.model(legacy_systems, legacy_outputs, legacy_selected)
            for i in batch:
                results[i] = _tensor_to_core(values[outputs[i].name])

        return results

    def _legacy_output(self, request: Quantity) -> ModelOutput:
        # The inner module receives requests in its declared units;
        # execute_model converts its outputs to the engine's requested units.
        declared = self._declared[request.name]
        return ModelOutput(
            unit=declared.unit,
            sample_kind=request.sample_kind,
            explicit_gradients=request.gradients,
            description=request.description or "",
        )

    def _legacy_system(self, system: System) -> LegacySystem:
        legacy = LegacySystem(system.types, system.positions, system.cell, system.pbc)
        device = system.positions.device

        for legacy_options, options in zip(
            self._neighbor_lists, self._pair_lists, strict=True
        ):
            # the pair vectors are already registered with autograd by TorchModel
            pairs = _block_to_torch(system.pairs(options), device)
            legacy.add_pairs(legacy_options, pairs)

        for name in system.known_custom_data():
            legacy.add_custom_data(
                name, _tensor_to_torch(system.custom_data(name), device)
            )

        return legacy


def _quantity(name, output, sample_kind=None, gradients=None) -> Quantity:
    return Quantity(
        name=name,
        unit=output.unit,
        sample_kind=output.sample_kind if sample_kind is None else sample_kind,
        description=output.description,
        gradients=list(output.explicit_gradients) if gradients is None else gradients,
    )


def _unique_name_batches(outputs: list[Quantity]) -> list[list[int]]:
    batches: list[list[int]] = []
    for i, output in enumerate(outputs):
        batch = next(
            (b for b in batches if all(outputs[j].name != output.name for j in b)),
            None,
        )
        if batch is None:
            batches.append([i])
        else:
            batch.append(i)
    return batches


# Conversions between metatensor (core) and metatensor.torch. The two bindings can
# not read each other's arrays, so blocks are re-created around the same values,
# which keeps the autograd graph in both directions.


def _labels_to_torch(labels: mts.Labels, device) -> mts.torch.Labels:
    # labels are immutable and reference counted: share the C object instead of
    # copying and re-validating the values
    ptr = ctypes.cast(copy.copy(labels).release(), ctypes.c_void_p).value
    return mts.torch.Labels.unsafe_from_ptr(ptr).to(device)


def _labels_to_core(labels: mts.torch.Labels) -> mts.Labels:
    return mts.Labels(labels.names, labels.values.cpu().numpy())


def _block_to_torch(block: mts.TensorBlock, device) -> mts.torch.TensorBlock:
    result = mts.torch.TensorBlock(
        values=block.values,
        samples=_labels_to_torch(block.samples, device),
        components=[_labels_to_torch(c, device) for c in block.components],
        properties=_labels_to_torch(block.properties, device),
    )
    for parameter, gradient in block.gradients():
        result.add_gradient(parameter, _block_to_torch(gradient, device))
    return result


def _block_to_core(block: mts.torch.TensorBlock) -> mts.TensorBlock:
    result = mts.TensorBlock(
        values=block.values,
        samples=_labels_to_core(block.samples),
        components=[_labels_to_core(c) for c in block.components],
        properties=_labels_to_core(block.properties),
    )
    for parameter, gradient in block.gradients():
        result.add_gradient(parameter, _block_to_core(gradient))
    return result


def _tensor_to_torch(tensor: mts.TensorMap, device) -> mts.torch.TensorMap:
    return mts.torch.TensorMap(
        _labels_to_torch(tensor.keys, device),
        [_block_to_torch(block, device) for block in tensor.blocks()],
    )


def _tensor_to_core(tensor: mts.torch.TensorMap) -> mts.TensorMap:
    return mts.TensorMap(
        _labels_to_core(tensor.keys),
        [_block_to_core(block) for block in tensor.blocks()],
    )

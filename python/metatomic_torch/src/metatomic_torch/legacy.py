"""
Run models written for the ``metatomic.torch`` API (:py:class:`AtomisticModel`) with
:py:func:`metatomic.execute_model`, by wrapping them in a
:py:class:`metatomic.TorchModel`.
"""

import os
import sys
import warnings
from typing import Optional

import metatensor
import metatensor.torch as mts_torch
import numpy as np
import torch

from metatomic import (
    ModelCapabilities,
    ModelMetadata,
    PairListOptions,
    Quantity,
    System,
    TorchModel,
)

from . import ModelOutput, unit_conversion_factor
from . import System as LegacySystem
from ._quantities import current_quantity_name
from .model import is_atomistic_model, load_atomistic_model


# legacy device names that are spelled differently in metatomic
_DEVICES = {"mps": "metal"}


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

    def __init__(
        self,
        model,
        length_unit: Optional[str] = None,
        *,
        extensions_directory=None,
    ):
        """
        :param model: the :py:class:`AtomisticModel` to wrap, or a path to an
            exported model
        :param extensions_directory: directory containing TorchScript extensions
            required when loading an exported model
        :param length_unit: length unit of the model, for models declaring an empty
            ``length_unit`` (which meant "no unit conversion"). Defaults to the unit
            declared by the model, or ``"angstrom"`` if it is empty.
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
        if length_unit is None:
            length_unit = capabilities.length_unit
            if length_unit == "":
                warnings.warn(
                    "this model does not declare a length unit, assuming 'angstrom'. "
                    "Pass `length_unit` to LegacyModel to silence this warning",
                    stacklevel=2,
                )
                length_unit = "angstrom"

        elif (
            capabilities.length_unit
            and unit_conversion_factor(capabilities.length_unit, length_unit) != 1.0
        ):
            raise ValueError(
                "`length_unit` can only override an empty model length unit; "
                f"the model declares {capabilities.length_unit!r}"
            )

        self._declared = {
            name: output
            for name, output in capabilities.outputs.items()
            # AtomisticModel lists both deprecated and current names, keep the latter
            if current_quantity_name(name) == name
        }
        self._capabilities = ModelCapabilities(
            atomic_types=list(capabilities.atomic_types),
            # metatomic can not send `inf` through JSON yet, use the largest float
            interaction_range=min(capabilities.interaction_range, sys.float_info.max),
            length_unit=length_unit,
            supported_devices=[
                _DEVICES.get(device, device)
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
        selected_atoms: Optional[metatensor.Labels],
        outputs: list[Quantity],
    ) -> list[metatensor.TensorMap]:
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
            legacy.add_neighbor_list(legacy_options, pairs)

        for name in system.known_custom_data():
            legacy.add_data(name, _tensor_to_torch(system.custom_data(name), device))

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


# Conversions between metatensor (core) and metatensor.torch. The values are passed
# through as-is, so the autograd graph is kept in both directions.


def _labels_to_torch(labels: metatensor.Labels, device) -> mts_torch.Labels:
    values = torch.tensor(np.asarray(labels.values), dtype=torch.int32, device=device)
    return mts_torch.Labels(list(labels.names), values)


def _labels_to_core(labels: mts_torch.Labels) -> metatensor.Labels:
    return metatensor.Labels(labels.names, labels.values.cpu().numpy())


def _block_to_torch(block: metatensor.TensorBlock, device) -> mts_torch.TensorBlock:
    result = mts_torch.TensorBlock(
        values=block.values,
        samples=_labels_to_torch(block.samples, device),
        components=[_labels_to_torch(c, device) for c in block.components],
        properties=_labels_to_torch(block.properties, device),
    )
    for parameter, gradient in block.gradients():
        result.add_gradient(parameter, _block_to_torch(gradient, device))
    return result


def _block_to_core(block: mts_torch.TensorBlock) -> metatensor.TensorBlock:
    result = metatensor.TensorBlock(
        values=block.values,
        samples=_labels_to_core(block.samples),
        components=[_labels_to_core(c) for c in block.components],
        properties=_labels_to_core(block.properties),
    )
    for parameter, gradient in block.gradients():
        result.add_gradient(parameter, _block_to_core(gradient))
    return result


def _tensor_to_torch(tensor: metatensor.TensorMap, device) -> mts_torch.TensorMap:
    return mts_torch.TensorMap(
        _labels_to_torch(tensor.keys, device),
        [_block_to_torch(block, device) for block in tensor.blocks()],
    )


def _tensor_to_core(tensor: mts_torch.TensorMap) -> metatensor.TensorMap:
    return metatensor.TensorMap(
        _labels_to_core(tensor.keys),
        [_block_to_core(block) for block in tensor.blocks()],
    )

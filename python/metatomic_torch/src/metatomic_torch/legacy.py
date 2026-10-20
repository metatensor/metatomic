"""
Run models written for the ``metatomic.torch`` API (:py:class:`AtomisticModel`) with
:py:func:`metatomic.execute_model`, by wrapping them in a
:py:class:`metatomic.TorchModel`.
"""

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

from . import ModelEvaluationOptions, ModelOutput, unit_conversion_factor
from . import System as LegacySystem
from ._quantities import current_quantity_name, deprecated_quantity_name
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
    output and input names are exposed with their current name.

    .. code-block:: python

        import metatomic
        from metatomic_torch.legacy import LegacyModel, system_with_pairs

        model = LegacyModel.load("model.pt", extensions_directory="extensions/")
        energy = metatomic.Quantity(
            name="energy", unit="eV", sample_kind="system", gradients=["positions"]
        )
        system = system_with_pairs(
            types,
            positions,
            cell,
            pbc,
            model.requested_pair_lists([energy]),
            length_unit=model.capabilities().length_unit,
            model_length_unit=model.capabilities().length_unit,
        )
        (result,) = metatomic.execute_model(model, [system], [energy])
    """

    def __init__(
        self,
        model,
        check_consistency: bool = False,
        length_unit: Optional[str] = None,
    ):
        """
        :param model: the :py:class:`AtomisticModel` to wrap
        :param check_consistency: run the legacy consistency checks inside the
            :py:class:`AtomisticModel` as well. The checks of
            :py:func:`metatomic.execute_model` are controlled separately.
        :param length_unit: length unit of the model, for models declaring an empty
            ``length_unit`` (which meant "no unit conversion"). Defaults to the unit
            declared by the model, or ``"angstrom"`` if it is empty.
        """
        super().__init__()
        if not is_atomistic_model(model):
            raise TypeError(f"`model` must be an AtomisticModel, got {type(model)}")

        self.model = model
        self.check_consistency = check_consistency

        capabilities = model.capabilities()
        # the model already works in this unit, never ask it to convert anything
        self._legacy_length_unit = capabilities.length_unit
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

    @classmethod
    def load(cls, path, extensions_directory=None, **kwargs) -> "LegacyModel":
        """
        Load a model exported with :py:meth:`AtomisticModel.save` and wrap it.

        :param path: path to the exported model
        :param extensions_directory: directory containing the TorchScript extensions
            required by the model, if any
        :param kwargs: forwarded to :py:class:`LegacyModel`
        """
        return cls(load_atomistic_model(path, extensions_directory), **kwargs)

    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def metadata(self) -> ModelMetadata:
        metadata = self.model.metadata()
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
        # AtomisticModel already collected the requests of all its sub-modules, and
        # its `requested_inputs(use_new_names: bool)` (as well as the one of legacy
        # sub-modules) does not follow the signature expected by TorchModel, so we
        # must not recurse into it.
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
            options = ModelEvaluationOptions(
                length_unit=self._legacy_length_unit,
                outputs={
                    outputs[i].name: self._legacy_output(outputs[i]) for i in batch
                },
                selected_atoms=legacy_selected,
            )
            values = self.model(legacy_systems, options, self.check_consistency)
            for i in batch:
                results[i] = _rename_properties(
                    _tensor_to_core(values[outputs[i].name]), outputs[i].name
                )

        return results

    def _legacy_output(self, request: Quantity) -> ModelOutput:
        # the systems are already in the model units, and the outputs are converted
        # by `execute_model`: request the declared unit so nothing is converted twice
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


def system_with_pairs(
    types,
    positions,
    cell,
    pbc,
    pair_lists: list[PairListOptions],
    length_unit: str = "angstrom",
    model_length_unit: Optional[str] = None,
) -> System:
    """
    Create a :py:class:`metatomic.System` from NumPy arrays and attach a pair list,
    computed with `vesin <https://luthaf.fr/vesin/>`_, for each of ``pair_lists``
    (typically ``model.requested_pair_lists(outputs)``).

    The cutoffs in ``pair_lists`` are in the model unit: give ``model_length_unit``
    when it differs from ``length_unit``, the unit of ``positions`` and ``cell``.
    The dtype of ``positions`` is used for all the data. From :py:class:`ase.Atoms`:

    .. code-block:: python

        system_with_pairs(
            atoms.numbers,
            atoms.positions,
            atoms.cell[:],
            atoms.pbc,
            model.requested_pair_lists(outputs),
            model_length_unit=model.capabilities().length_unit,
        )
    """
    import vesin

    types = np.asarray(types, dtype=np.int32)
    positions = np.asarray(positions)
    cell = np.asarray(cell, dtype=positions.dtype)
    pbc = np.asarray(pbc, dtype=bool)

    factor = 1.0
    if model_length_unit is not None:
        factor = unit_conversion_factor(model_length_unit, length_unit)

    system = System(length_unit, types, positions, cell, pbc)
    for options in pair_lists:
        i, j, shifts, vectors = vesin.NeighborList(
            cutoff=options.cutoff * factor, full_list=options.full_list
        ).compute(
            points=positions.astype(np.float64),
            box=cell.astype(np.float64),
            periodic=pbc,
            quantities="ijSD",
        )
        samples = np.column_stack([i, j, shifts]).astype(np.int32)
        system.add_pairs(
            options,
            metatensor.TensorBlock(
                values=vectors.astype(positions.dtype).reshape(-1, 3, 1),
                samples=metatensor.Labels(
                    [
                        "first_atom",
                        "second_atom",
                        "cell_shift_a",
                        "cell_shift_b",
                        "cell_shift_c",
                    ],
                    samples.reshape(-1, 5),
                ),
                components=[metatensor.Labels.range("xyz", 3)],
                properties=metatensor.Labels.range("distance", 1),
            ),
        )
    return system


def _quantity(name, output, sample_kind=None, gradients=None) -> Quantity:
    return Quantity(
        name=name,
        unit=output.unit,
        sample_kind=output.sample_kind if sample_kind is None else sample_kind,
        description=output.description,
        gradients=list(output.explicit_gradients) if gradients is None else gradients,
    )


def _rename_properties(tensor: metatensor.TensorMap, name: str):
    # models using a deprecated name (e.g. `non_conservative_forces`) also used it
    # for the properties, while metatomic checks for the current name
    current = name.split("/")[0]
    deprecated = deprecated_quantity_name(current)
    if deprecated != current and deprecated in tensor.property_names:
        return metatensor.rename_dimension(tensor, "properties", deprecated, current)
    return tensor


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

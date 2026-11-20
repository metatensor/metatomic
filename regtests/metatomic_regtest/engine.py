"""A minimal engine, used to run the frozen models of the regression tests"""

import json
import os
from collections import namedtuple
from typing import Dict, List

import metatensor
import metatensor_torch as mts
import numpy as np
import torch
import vesin
from metatensor_torch import Labels, TensorBlock, TensorMap
from vesin.metatomic import neighbor_lists_for_model

import metatomic
from metatomic_torch import (
    AtomisticModel,
    ModelEvaluationOptions,
    ModelOutput,
    System,
    load_atomistic_model,
)
from metatomic_torch.utils import legacy_plugin_path

from .models import model_path


Input = namedtuple(
    "Input",
    [
        "path",
        "directory",
        "model",
        "length_unit",
        "systems",
        "selected_atoms",
        "outputs",
    ],
)

STR_TO_DTYPE = {"float32": torch.float32, "float64": torch.float64}

PAIRS_SAMPLES = [
    "first_atom",
    "second_atom",
    "cell_shift_a",
    "cell_shift_b",
    "cell_shift_c",
]


def load_input(reference: str) -> Input:
    """
    Read the system, requested outputs, and expected values of the outputs for the given
    reference.
    """
    with open(reference) as fd:
        data = json.load(fd)

    selected_atoms = data.get("selected_atoms")
    if selected_atoms is not None:
        selected_atoms = Labels(
            ["system", "atom"], torch.tensor(selected_atoms, dtype=torch.int32)
        )

    return Input(
        path=reference,
        directory=os.path.dirname(reference),
        model=data["model"],
        length_unit=data["length_unit"],
        systems=data["systems"],
        selected_atoms=selected_atoms,
        outputs=data["outputs"],
    )


def load_expected(input: Input) -> Dict[str, TensorMap]:
    """Read the reference outputs stored next to the given input."""
    expected = {}
    for name, output in input.outputs.items():
        expected[name] = mts.load(os.path.join(input.directory, output["reference"]))

    return expected


def load_model(reference: str) -> AtomisticModel:
    """
    Download & cache the required model for the given reference and then load it.
    """
    return load_atomistic_model(model_path(reference))


def _build_systems(input: Input, dtype: torch.dtype):
    """
    Create the :py:class:`System` for this input, tracking the tensors we'll later need
    to extract gradients from.

    Returns the systems, the per-system positions, and the per-system strain. Each
    system gets its own strain tensor: a single shared one would accumulate the
    gradients of all the systems into a single value.
    """
    gradients = set()
    for output in input.outputs.values():
        gradients.update(output.get("gradients", []))

    systems = []
    all_positions = []
    all_strains = []
    for data in input.systems:
        types = torch.tensor(data["types"], dtype=torch.int32)
        positions = torch.tensor(data["positions"], dtype=dtype)
        cell = torch.tensor(data["cell"], dtype=dtype)
        pbc = torch.tensor(data["pbc"], dtype=torch.bool)

        if "positions" in gradients:
            positions.requires_grad_(True)
        all_positions.append(positions)

        if "strain" in gradients:
            strain = torch.eye(3, requires_grad=True, dtype=dtype)
            positions = positions @ strain
            if "positions" in gradients:
                # `positions` is no longer a leaf of the autograd graph
                all_positions[-1].retain_grad()
            cell = cell @ strain
            all_strains.append(strain)

        systems.append(System(types, positions, cell, pbc))

    return systems, all_positions, all_strains


def _input_block(
    input: Input, name: str, sample_kind: str, data: dict, n_atoms: int, dtype
) -> TensorBlock:
    """
    Build the ``name`` input that a model requests with ``sample_kind``, taking it from
    the ``inputs`` of the system ``data`` in ``input.json``.

    The model says whether it wants one value per atom or one per system, and the shape
    of the data in ``input.json`` says whether that value is a scalar or a vector.
    """
    inputs = data.get("inputs", {})
    if name not in inputs:
        raise ValueError(
            f"the model requires the '{name}' input, but a system in "
            f'{input.path} does not provide it; add it to the "inputs" of '
            f"that system"
        )

    values = torch.tensor(inputs[name], dtype=dtype)

    if sample_kind == "atom":
        count = n_atoms
        samples = Labels(
            ["system", "atom"],
            torch.tensor([[0, i] for i in range(count)], dtype=torch.int32),
        )
        scalar = values.dim() == 1
        vector = values.dim() == 2 and values.shape[1] == 3
        expected = "one value or one vector per atom"
    elif sample_kind == "system":
        count = 1
        samples = Labels("system", torch.tensor([[0]]))
        scalar = values.dim() == 0
        vector = values.dim() == 1 and values.shape[0] == 3
        expected = "one value or one vector for the whole system"
    else:
        raise ValueError(
            f"the model asks for the '{name}' input with sample kind "
            f"'{sample_kind}', which this engine does not know how to build"
        )

    if scalar:
        components = []
        values = values.reshape(count, 1)
    elif vector:
        components = [Labels("xyz", torch.tensor([[0], [1], [2]]))]
        values = values.reshape(count, 3, 1)
    else:
        raise ValueError(
            f"can not make the '{name}' input from data of shape "
            f"{tuple(values.shape)}: expected {expected}"
        )

    return TensorBlock(
        values=values,
        samples=samples,
        components=components,
        properties=Labels(name, torch.tensor([[0]])),
    )


def _add_model_inputs(
    input: Input, model: AtomisticModel, systems: List[System], dtype: torch.dtype
):
    """
    Attach the extra data the model declares through ``requested_inputs()``, taking it
    from the ``inputs`` of each system in ``input.json``.
    """
    requested = model.requested_inputs(use_new_names=True)

    for system, data in zip(systems, input.systems, strict=True):
        for name, output in requested.items():
            block = _input_block(
                input, name, output.sample_kind, data, len(system), dtype
            )
            system.add_data(name, TensorMap(Labels.single(), [block]))


def _gradient_block(
    block: TensorBlock,
    name: str,
    values: List[torch.Tensor],
    components: List[Labels],
) -> TensorBlock:
    """
    Build the gradient :py:class:`TensorBlock` called ``name`` for ``block``, from the
    per-sample gradient ``values``.
    """
    if name == "positions":
        samples = []
        for sample, gradient in enumerate(values):
            for atom in range(gradient.shape[0]):
                samples.append([sample, atom])

        samples = Labels(["sample", "atom"], torch.tensor(samples, dtype=torch.int32))
    else:
        assert name == "strain"
        samples = Labels(
            ["sample"],
            torch.arange(len(values), dtype=torch.int32).reshape(-1, 1),
        )

    return TensorBlock(
        values=torch.concatenate(values),
        samples=samples,
        components=components,
        properties=block.properties,
    )


def _compute_gradients(
    output: TensorMap,
    gradients: List[str],
    all_positions: List[torch.Tensor],
    all_strains: List[torch.Tensor],
) -> TensorMap:
    """
    Compute the requested gradients of ``output`` with autograd, and return a new
    :py:class:`TensorMap` with them attached.
    """
    if len(output) != 1:
        raise ValueError(
            "can only compute gradients for a TensorMap with a single block, "
            f"got {len(output)} blocks"
        )

    block = output.block()
    if block.samples.names != ["system"]:
        raise ValueError(
            "can only compute gradients for outputs with one sample per system, got "
            f"samples named {block.samples.names}"
        )

    n_systems = len(all_positions)
    n_properties = len(block.properties)

    xyz = Labels("xyz", torch.tensor([[0], [1], [2]], dtype=torch.int32))
    xyz_1 = Labels("xyz_1", torch.tensor([[0], [1], [2]], dtype=torch.int32))
    xyz_2 = Labels("xyz_2", torch.tensor([[0], [1], [2]], dtype=torch.int32))

    # one gradient per property, since autograd only differentiates scalars
    positions_gradients = [[] for _ in range(n_systems)]
    strain_gradients = [[] for _ in range(n_systems)]
    for property in range(n_properties):
        wrt = []
        if "positions" in gradients:
            wrt += all_positions
        if "strain" in gradients:
            wrt += all_strains

        computed = torch.autograd.grad(
            outputs=block.values[:, property].sum(),
            inputs=wrt,
            retain_graph=property != n_properties - 1,
        )

        computed = list(computed)
        if "positions" in gradients:
            for system in range(n_systems):
                positions_gradients[system].append(computed.pop(0))
        if "strain" in gradients:
            for system in range(n_systems):
                strain_gradients[system].append(computed.pop(0))

    new_block = TensorBlock(
        values=block.values.detach(),
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )

    # keep any gradient the model computed itself
    for name, gradient in block.gradients():
        new_block.add_gradient(
            name,
            TensorBlock(
                values=gradient.values.detach(),
                samples=gradient.samples,
                components=gradient.components,
                properties=gradient.properties,
            ),
        )

    if "positions" in gradients:
        # (n_atoms, 3) per property => (n_atoms, 3, n_properties)
        values = [
            torch.stack(per_property, dim=-1).detach()
            for per_property in positions_gradients
        ]
        new_block.add_gradient(
            "positions", _gradient_block(block, "positions", values, [xyz])
        )

    if "strain" in gradients:
        # (3, 3) per property => (1, 3, 3, n_properties)
        values = [
            torch.stack(per_property, dim=-1).reshape(1, 3, 3, n_properties).detach()
            for per_property in strain_gradients
        ]
        new_block.add_gradient(
            "strain", _gradient_block(block, "strain", values, [xyz_1, xyz_2])
        )

    return TensorMap(output.keys, [new_block])


def _detach(tensor: TensorMap) -> TensorMap:
    blocks = []
    for block in tensor.blocks():
        new_block = TensorBlock(
            values=block.values.detach(),
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )
        for name, gradient in block.gradients():
            new_block.add_gradient(
                name,
                TensorBlock(
                    values=gradient.values.detach(),
                    samples=gradient.samples,
                    components=gradient.components,
                    properties=gradient.properties,
                ),
            )
        blocks.append(new_block)

    return TensorMap(tensor.keys, blocks)


def run_model(model: AtomisticModel, input: Input) -> Dict[str, TensorMap]:
    capabilities = model.capabilities()
    dtype = STR_TO_DTYPE[capabilities.dtype]

    systems, all_positions, all_strains = _build_systems(input, dtype)

    for calculator in neighbor_lists_for_model(input.length_unit, model, skin=0.0):
        calculator.add_neighbor_list(systems)

    _add_model_inputs(input, model, systems, dtype)

    options = ModelEvaluationOptions(
        length_unit=input.length_unit,
        outputs={
            name: ModelOutput(
                unit=output.get("unit", ""),
                sample_kind=output.get("sample_kind", "system"),
                explicit_gradients=[],
            )
            for name, output in input.outputs.items()
        },
        selected_atoms=input.selected_atoms,
    )

    outputs = model(systems=systems, options=options, check_consistency=True)

    results = {}
    for name, output in input.outputs.items():
        with_autograd = [gradient for gradient in output.get("gradients", [])]

        if with_autograd:
            results[name] = _compute_gradients(
                outputs[name], with_autograd, all_positions, all_strains
            )
        else:
            results[name] = _detach(outputs[name])

    return results


# The same models, loaded and executed through the metatomic-torch legacy plugin with
# the metatomic-core API, like an engine using the metatomic C or C++ API would.


def load_plugin_model(reference: str) -> metatomic.BaseModel:
    """
    Download & cache the required model for the given reference and then load it with
    the metatomic-torch legacy plugin.
    """
    metatomic.load_plugin(legacy_plugin_path)
    # the outputs of the plugin are stored in C++ arrays, which metatensor-core reads
    # through DLPack since 0.2.7, and only through registered wrappers before
    if not hasattr(metatensor.TensorBlock, "values_dlpack"):
        for origin in [
            "metatensor_torch::TorchDataArray",
            "metatensor::SimpleDataArray",
        ]:
            metatensor.register_external_data_wrapper(
                origin, metatensor.ExternalCpuArray
            )

    # the plugin recognizes the files exported by AtomisticModel by itself
    return metatomic.load_model(model_path(reference))


def _labels_to_core(labels: Labels) -> metatensor.Labels:
    return metatensor.Labels(labels.names, labels.values.numpy())


def _labels_to_torch(labels: metatensor.Labels) -> Labels:
    return Labels(labels.names, torch.tensor(np.asarray(labels.values)))


def _values_to_torch(values) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu()
    return torch.tensor(np.asarray(values))


def _block_to_torch(block: metatensor.TensorBlock) -> TensorBlock:
    """Convert a block of the plugin's outputs to metatensor-torch"""
    result = TensorBlock(
        values=_values_to_torch(block.values),
        samples=_labels_to_torch(block.samples),
        components=[_labels_to_torch(c) for c in block.components],
        properties=_labels_to_torch(block.properties),
    )

    for parameter, gradient in block.gradients():
        gradient = _block_to_torch(gradient)
        if parameter == "positions":
            # metatomic-core gives the system of each atom in the gradient samples,
            # the references use the ("sample", "atom") samples of metatomic-torch
            gradient = TensorBlock(
                values=gradient.values,
                samples=Labels(["sample", "atom"], gradient.samples.values[:, [0, 2]]),
                components=gradient.components,
                properties=gradient.properties,
            )
        result.add_gradient(parameter, gradient)

    return result


def _plugin_systems(
    input: Input, model: metatomic.BaseModel, requests: List[metatomic.Quantity]
) -> List[metatomic.System]:
    """Create the metatomic-core systems for this input, with the pairs and inputs
    requested by ``model``"""
    capabilities = model.capabilities()
    dtype = STR_TO_DTYPE[capabilities.dtype]
    factor = metatomic.unit_conversion_factor(
        capabilities.length_unit, input.length_unit
    )

    systems = []
    for data in input.systems:
        positions = torch.tensor(data["positions"], dtype=dtype).numpy()
        cell = torch.tensor(data["cell"], dtype=dtype).numpy()
        pbc = np.array(data["pbc"], dtype=bool)
        system = metatomic.System(
            input.length_unit,
            np.array(data["types"], dtype=np.int32),
            positions,
            cell,
            pbc,
        )

        # the cutoffs of the pair lists are in the length unit of the model
        for options in model.requested_pair_lists(requests):
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
                    samples=metatensor.Labels(PAIRS_SAMPLES, samples.reshape(-1, 5)),
                    components=[metatensor.Labels.range("xyz", 3)],
                    properties=metatensor.Labels.range("distance", 1),
                ),
            )

        for quantity in model.requested_inputs(requests):
            block = _input_block(
                input, quantity.name, quantity.sample_kind, data, len(positions), dtype
            )
            tensor = metatensor.TensorMap(
                metatensor.Labels.single(),
                [
                    metatensor.TensorBlock(
                        values=block.values.numpy(),
                        samples=_labels_to_core(block.samples),
                        components=[_labels_to_core(c) for c in block.components],
                        properties=_labels_to_core(block.properties),
                    )
                ],
            )
            # the values in input.json are in the unit requested by the model
            tensor.set_info("unit", quantity.unit)
            system.add_custom_data(quantity.name, tensor)

        systems.append(system)

    return systems


def run_plugin_model(model: metatomic.BaseModel, input: Input) -> Dict[str, TensorMap]:
    """
    Run a model loaded with :py:func:`load_plugin_model` with ``execute_model``, and
    convert the outputs to metatensor-torch to compare them with the references.
    """
    requests = [
        metatomic.Quantity(
            name=name,
            unit=output.get("unit", ""),
            sample_kind=output.get("sample_kind", "system"),
            gradients=output.get("gradients", []),
        )
        for name, output in input.outputs.items()
    ]

    selected_atoms = None
    if input.selected_atoms is not None:
        selected_atoms = _labels_to_core(input.selected_atoms)

    outputs = metatomic.execute_model(
        model,
        _plugin_systems(input, model, requests),
        requests,
        selected_atoms=selected_atoms,
        check_consistency=True,
    )

    results = {}
    for request, output in zip(requests, outputs, strict=True):
        results[request.name] = TensorMap(
            _labels_to_torch(output.keys),
            [_block_to_torch(block) for block in output.blocks()],
        )
    return results

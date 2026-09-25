"""A minimal engine, used to run the frozen models of the regression tests"""

import json
import os
from collections import namedtuple
from typing import Dict, List

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from vesin.metatomic import neighbor_lists_for_model

from metatomic.torch import (
    AtomisticModel,
    ModelEvaluationOptions,
    ModelOutput,
    System,
    load_atomistic_model,
)

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
    import metatensor.torch as mts

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


def _add_model_inputs(
    input: Input, model: AtomisticModel, systems: List[System], dtype: torch.dtype
):
    """
    Attach the extra data the model declares through ``requested_inputs()``, taking it
    from the ``inputs`` of each system in ``input.json``.

    The model says whether it wants one value per atom or one per system, and the shape
    of the data in ``input.json`` says whether that value is a scalar or a vector.
    """
    requested = model.requested_inputs(use_new_names=True)

    for system, data in zip(systems, input.systems, strict=True):
        inputs = data.get("inputs", {})
        for name in requested.keys():
            if name not in inputs:
                raise ValueError(
                    f"the model requires the '{name}' input, but a system in "
                    f'{input.path} does not provide it; add it to the "inputs" of '
                    f"that system"
                )

            values = torch.tensor(inputs[name], dtype=dtype)
            sample_kind = requested[name].sample_kind

            if sample_kind == "atom":
                count = len(system)
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

            block = TensorBlock(
                values=values,
                samples=samples,
                components=components,
                properties=Labels(name, torch.tensor([[0]])),
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

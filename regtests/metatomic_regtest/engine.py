"""
A minimal engine, used to run the frozen models of the regression tests. It loads them
with the metatomic-torch legacy plugin and runs them with the metatomic-core API, like
an engine using the metatomic C or C++ API would.
"""

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

import metatomic
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

STR_TO_DTYPE = {"float32": np.float32, "float64": np.float64}


def load_input(reference: str) -> Input:
    """
    Read the system, requested outputs, and expected values of the outputs for the given
    reference.
    """
    with open(reference) as fd:
        data = json.load(fd)

    selected_atoms = data.get("selected_atoms")
    if selected_atoms is not None:
        selected_atoms = metatensor.Labels(
            ["system", "atom"], np.array(selected_atoms, dtype=np.int32)
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


def load_model(reference: str) -> metatomic.BaseModel:
    """
    Download & cache the required model for the given reference and then load it with
    the metatomic-torch legacy plugin.
    """
    metatomic.load_plugin(legacy_plugin_path)

    # the plugin recognizes the files exported by AtomisticModel by itself
    return metatomic.load_model(model_path(reference))


def _input_block(
    input: Input, name: str, sample_kind: str, data: dict, n_atoms: int, dtype
) -> metatensor.TensorBlock:
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

    values = np.array(inputs[name], dtype=dtype)

    if sample_kind == "atom":
        count = n_atoms
        samples = metatensor.Labels(
            ["system", "atom"],
            np.array([[0, i] for i in range(count)], dtype=np.int32),
        )
        scalar = values.ndim == 1
        vector = values.ndim == 2 and values.shape[1] == 3
        expected = "one value or one vector per atom"
    elif sample_kind == "system":
        count = 1
        samples = metatensor.Labels("system", np.array([[0]], dtype=np.int32))
        scalar = values.ndim == 0
        vector = values.ndim == 1 and values.shape[0] == 3
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
        components = [metatensor.Labels.range("xyz", 3)]
        values = values.reshape(count, 3, 1)
    else:
        raise ValueError(
            f"can not make the '{name}' input from data of shape "
            f"{tuple(values.shape)}: expected {expected}"
        )

    return metatensor.TensorBlock(
        values=values,
        samples=samples,
        components=components,
        properties=metatensor.Labels(name, np.array([[0]], dtype=np.int32)),
    )


def _build_systems(
    input: Input, model: metatomic.BaseModel, requests: List[metatomic.Quantity]
) -> List[metatomic.System]:
    """Create the systems for this input, with the pairs and inputs requested by
    ``model``"""
    capabilities = model.capabilities()
    dtype = STR_TO_DTYPE[capabilities.dtype]
    factor = metatomic.unit_conversion_factor(
        capabilities.length_unit, input.length_unit
    )

    systems = []
    for data in input.systems:
        positions = np.array(data["positions"], dtype=dtype)
        cell = np.array(data["cell"], dtype=dtype)
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
                    values=vectors.astype(dtype).reshape(-1, 3, 1),
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

        for quantity in model.requested_inputs(requests):
            block = _input_block(
                input, quantity.name, quantity.sample_kind, data, len(positions), dtype
            )
            tensor = metatensor.TensorMap(metatensor.Labels.single(), [block])
            # the values in input.json are in the unit requested by the model
            tensor.set_info("unit", quantity.unit)
            system.add_custom_data(quantity.name, tensor)

        systems.append(system)

    return systems


def _labels_to_torch(labels: metatensor.Labels) -> Labels:
    return Labels(labels.names, torch.tensor(np.asarray(labels.values)))


def _values_to_torch(values) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu()
    return torch.tensor(np.asarray(values))


def _block_to_torch(block: metatensor.TensorBlock) -> TensorBlock:
    """Convert a block of the outputs to metatensor-torch, like the references"""
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


def run_model(model: metatomic.BaseModel, input: Input) -> Dict[str, TensorMap]:
    """
    Run a model loaded with :py:func:`load_model` with ``execute_model``, and convert
    the outputs to metatensor-torch to compare them with the references.
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

    outputs = metatomic.execute_model(
        model,
        _build_systems(input, model, requests),
        requests,
        selected_atoms=input.selected_atoms,
        check_consistency=True,
    )

    results = {}
    for request, output in zip(requests, outputs, strict=True):
        results[request.name] = TensorMap(
            _labels_to_torch(output.keys),
            [_block_to_torch(block) for block in output.blocks()],
        )
    return results

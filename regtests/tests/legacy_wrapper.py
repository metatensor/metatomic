"""
Run models exported with ``metatomic.torch`` through ``metatomic_torch.legacy``'s
``LegacyModel`` and :py:func:`metatomic.execute_model`, and check that they produce the
same outputs as the stored references and as the legacy engine.
"""

import glob
import os
import sys

import metatensor
import numpy as np
import pytest
import torch
import vesin
from metatomic_torch._quantities import current_quantity_name

import metatomic
from metatomic.torch import load_atomistic_model


LegacyModel = pytest.importorskip("metatomic_torch.legacy").LegacyModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = sorted(glob.glob(os.path.join(ROOT, "references", "*", "input.json")))
CASE_IDS = [os.path.basename(os.path.dirname(case)) for case in CASES]

PAIR_SAMPLES = [
    "first_atom",
    "second_atom",
    "cell_shift_a",
    "cell_shift_b",
    "cell_shift_c",
]

sys.path.append(ROOT)

import metatomic_regtest  # noqa E402
from metatomic_regtest.engine import Input  # noqa E402


def _numpy(array):
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _pairs(data, options, dtype):
    """Pair list for one system of ``input.json``, as a metatensor block"""
    positions = np.array(data["positions"], dtype=np.float64)
    cell = np.array(data["cell"], dtype=np.float64)
    pbc = np.array(data["pbc"], dtype=bool)

    calculator = vesin.NeighborList(cutoff=options.cutoff, full_list=options.full_list)
    i, j, S, D = calculator.compute(positions, cell, pbc, quantities="ijSD")

    samples = np.concatenate([i[:, None], j[:, None], S], axis=1).astype(np.int32)
    return metatensor.TensorBlock(
        values=torch.tensor(D, dtype=dtype).reshape(-1, 3, 1),
        samples=metatensor.Labels(PAIR_SAMPLES, samples),
        components=[metatensor.Labels.range("xyz", 3)],
        properties=metatensor.Labels.range("distance", 1),
    )


def _custom_data(name, quantity, values, n_atoms, dtype):
    """
    Custom data for one system, following ``metatomic_regtest.engine._add_model_inputs``
    """
    values = torch.tensor(values, dtype=dtype)
    if quantity.sample_kind == "atom":
        count = n_atoms
        samples = metatensor.Labels(
            ["system", "atom"], np.array([[0, a] for a in range(n_atoms)], np.int32)
        )
    else:
        assert quantity.sample_kind == "system"
        count = 1
        samples = metatensor.Labels(["system"], np.array([[0]], np.int32))

    vector = values.numel() == 3 * count
    block = metatensor.TensorBlock(
        values=values.reshape(count, 3, 1) if vector else values.reshape(count, 1),
        samples=samples,
        components=[metatensor.Labels.range("xyz", 3)] if vector else [],
        properties=metatensor.Labels.range(name, 1),
    )
    return metatensor.TensorMap(metatensor.Labels.range("_", 1), [block])


def run_wrapped(model: LegacyModel, input: Input) -> dict:
    """
    Equivalent of :py:func:`metatomic_regtest.run_model`, using
    :py:func:`metatomic.execute_model` and letting ``model`` compute the gradients.
    """
    quantities = [
        metatomic.Quantity(
            name=name,
            unit=output.get("unit", ""),
            sample_kind=output.get("sample_kind", "system"),
            gradients=output.get("gradients", []),
        )
        for name, output in input.outputs.items()
    ]

    dtype = getattr(torch, model.capabilities().dtype)
    pair_lists = model.requested_pair_lists(quantities)
    inputs = model.requested_inputs(quantities)

    systems = []
    for data in input.systems:
        types = np.array(data["types"], dtype=np.int32)
        # pairs and custom data must be added before accessing the system arrays
        pairs = [(options, _pairs(data, options, dtype)) for options in pair_lists]
        custom = [
            (q.name, _custom_data(q.name, q, data["inputs"][q.name], len(types), dtype))
            for q in inputs
        ]

        system = metatomic.System(
            input.length_unit,
            torch.tensor(types),
            torch.tensor(data["positions"], dtype=dtype),
            torch.tensor(data["cell"], dtype=dtype),
            torch.tensor(data["pbc"], dtype=torch.bool),
        )
        for options, block in pairs:
            system.add_pairs(options, block)
        for name, tensor in custom:
            system.add_custom_data(name, tensor)
        systems.append(system)

    selected_atoms = input.selected_atoms
    if selected_atoms is not None:
        selected_atoms = metatensor.Labels(
            selected_atoms.names, _numpy(selected_atoms.values)
        )

    outputs = metatomic.execute_model(
        model,
        systems,
        quantities,
        selected_atoms=selected_atoms,
        check_consistency=True,
    )
    return dict(zip(input.outputs.keys(), outputs, strict=True))


def _sorted_values(block, names):
    """
    Values of ``block``, sorted by the ``names`` dimensions of the samples. Some models
    shuffle their samples, and the reference gradients do not have a ``system``
    dimension, so only these dimensions are used.
    """
    columns = [list(block.samples.names).index(name) for name in names]
    samples = _numpy(block.samples.values)[:, columns]
    order = np.lexsort(samples.T[::-1])
    return samples[order], _numpy(block.values)[order]


def compare(actual, expected) -> list:
    """
    Pair the values of ``actual`` and ``expected`` for every block and gradient, as
    ``(part, values, reference)``, after checking that their metadata match.
    """
    assert actual.keys.names == expected.keys.names
    np.testing.assert_equal(_numpy(actual.keys.values), _numpy(expected.keys.values))

    results = []
    for i, expected_block in enumerate(expected.blocks()):
        actual_block = actual.block(i)
        assert set(actual_block.gradients_list()) == set(
            expected_block.gradients_list()
        )
        parts = [("values", actual_block, expected_block)]
        for parameter in expected_block.gradients_list():
            parts.append(
                (
                    parameter,
                    actual_block.gradient(parameter),
                    expected_block.gradient(parameter),
                )
            )

        for part, block, reference in parts:
            part = f"block {i} {part}"
            assert block.properties.names == [
                current_quantity_name(name) for name in reference.properties.names
            ], part
            assert len(block.components) == len(reference.components), part
            for component, expected_component in zip(
                block.components, reference.components, strict=True
            ):
                assert component.names == expected_component.names, part
                np.testing.assert_equal(
                    _numpy(component.values),
                    _numpy(expected_component.values),
                    err_msg=part,
                )
            np.testing.assert_equal(
                _numpy(block.properties.values),
                _numpy(reference.properties.values),
                err_msg=part,
            )
            names = list(reference.samples.names)
            if block.samples.names != names:
                # Legacy position gradients omit the system column. Core adds it
                # explicitly; check it against the referenced parent sample.
                assert names == ["sample", "atom"], part
                assert block.samples.names == ["sample", "system", "atom"], part
                samples_array = _numpy(block.samples.values)
                parent_systems = _numpy(actual_block.samples.values)[
                    :, list(actual_block.samples.names).index("system")
                ]
                np.testing.assert_equal(
                    samples_array[:, 1],
                    parent_systems[samples_array[:, 0]],
                    err_msg=part,
                )
            samples, values = _sorted_values(block, names)
            reference_samples, reference = _sorted_values(reference, names)
            np.testing.assert_equal(samples, reference_samples, err_msg=part)
            results.append((part, values, reference.astype(values.dtype)))

    return results


def assert_close(actual, expected, rtol, atol, context):
    for part, values, reference in compare(actual, expected):
        np.testing.assert_allclose(
            values, reference, rtol=rtol, atol=atol, err_msg=f"{context}: {part}"
        )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_regression(case):
    input = metatomic_regtest.load_input(case)
    model = LegacyModel(metatomic_regtest.load_model(input.model))
    outputs = run_wrapped(model, input)

    for name, expected in metatomic_regtest.load_expected(input).items():
        rtol, atol = input.outputs[name]["rtol"], input.outputs[name]["atol"]
        assert_close(outputs[name], expected, rtol, atol, f"{input.model} {name}")


def _argon(n_cells: int, seed: int = 0):
    """Argon FCC crystal, with a small perturbation of the positions"""
    a = 5.26
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    grid = np.stack(np.meshgrid(*[np.arange(n_cells)] * 3, indexing="ij"), -1)
    positions = a * (grid.reshape(-1, 1, 3) + basis).reshape(-1, 3)
    positions += np.random.default_rng(seed).normal(scale=0.1, size=positions.shape)
    return {
        "types": [18] * len(positions),
        "positions": positions.tolist(),
        "cell": (a * n_cells * np.eye(3)).tolist(),
        "pbc": [True, True, True],
    }


@pytest.mark.parametrize("saved", [False, True], ids=["eager", "saved"])
@pytest.mark.parametrize("with_extension", [False, True], ids=["pure", "extension"])
def test_lj(saved, with_extension, tmp_path):
    lj_test = pytest.importorskip("metatomic_lj_test")

    atomistic = lj_test.lennard_jones_model(
        atomic_type=18,
        cutoff=7.0,
        epsilon=0.01,
        sigma=3.405,
        length_unit="angstrom",
        energy_unit="eV",
        with_extension=with_extension,
    )

    if saved:
        path = str(tmp_path / "lj-extension.pt")
        extensions = str(tmp_path / "extensions")
        atomistic.save(path, collect_extensions=extensions)
        atomistic = load_atomistic_model(path, extensions_directory=extensions)
        model = LegacyModel.load(path, extensions_directory=extensions)
    else:
        model = LegacyModel(atomistic)

    input = Input(
        path="",
        directory="",
        model="lj-extension",
        length_unit="angstrom",
        # Pure LJ stacks atom arrays even for system outputs, so all systems
        # must have the same atom count. The extension supports unequal counts.
        systems=[_argon(2), _argon(3) if with_extension else _argon(2, seed=1)],
        selected_atoms=None,
        outputs={
            "energy": {"unit": "eV", "gradients": ["positions", "strain"]},
            "energy/doubled": {"unit": "eV", "gradients": ["positions"]},
        },
    )
    if not with_extension:
        input.outputs.update(
            {
                "non_conservative_force": {
                    "unit": "eV/angstrom",
                    "sample_kind": "atom",
                },
                "non_conservative_stress": {"unit": "eV/angstrom^3"},
            }
        )

    per_atom = Input(
        **{
            **input._asdict(),
            # the extension model can only compute per-atom energies for systems with
            # the same number of atoms
            "systems": [_argon(2), _argon(2, seed=1)],
            "outputs": {"energy": {"unit": "eV", "sample_kind": "atom"}},
        }
    )

    for case in [input, per_atom]:
        # The pure LJ model adds random noise to its non-conservative outputs.
        torch.manual_seed(0)
        outputs = run_wrapped(model, case)
        for name in case.outputs:
            # the reference engine frees the graph after the gradients of each output,
            # and "energy" and "energy/doubled" share the same graph: only compute the
            # gradients of one output at the time
            single = {
                other: output if other == name else {**output, "gradients": []}
                for other, output in case.outputs.items()
            }
            single = Input(**{**case._asdict(), "outputs": single})
            torch.manual_seed(0)
            expected = metatomic_regtest.run_model(atomistic, single)[name]
            assert_close(outputs[name], expected, 1e-10, 1e-10, name)

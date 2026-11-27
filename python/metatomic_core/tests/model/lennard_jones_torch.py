import itertools
import textwrap

import metatensor as mts
import numpy as np
import pytest
import torch
from metatensor import Labels, TensorBlock

from metatomic import (
    PairListOptions,
    Quantity,
    System,
    execute_model,
    load_model,
)
from metatomic.testing import LennardJonesModel, TorchLennardJonesModel

from .._tests_utils import can_use_mps_backend


CUTOFF = 3.4
SIGMA = 1.5
EPSILON = 23.0

PAIRS_OPTIONS = PairListOptions(cutoff=CUTOFF, full_list=False, strict=True)

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")
if can_use_mps_backend():
    DEVICES.append("mps")


def compute_pairs(positions, cell, pbc):
    """Brute-force half pair list."""
    ranges = [range(-2, 3) if periodic else range(1) for periodic in pbc]

    samples = []
    vectors = []
    for i, j in itertools.combinations_with_replacement(range(len(positions)), 2):
        for shift in itertools.product(*ranges):
            if i == j and shift <= tuple(-s for s in shift):
                continue

            vector = positions[j] - positions[i] + np.array(shift) @ cell
            if np.linalg.norm(vector) < CUTOFF:
                samples.append([i, j, *shift])
                vectors.append(vector)

    return np.array(samples, dtype=np.int32), np.array(vectors).reshape(-1, 3)


def make_system(positions, cell, pbc):
    system = System(
        np.full(len(positions), 12, dtype=np.int32),
        positions,
        cell,
        np.array(pbc),
        length_unit="angstrom",
    )

    samples, vectors = compute_pairs(positions, cell, pbc)
    names = [
        "first_atom",
        "second_atom",
        "cell_shift_a",
        "cell_shift_b",
        "cell_shift_c",
    ]

    system.add_pairs(
        PairListOptions(cutoff=CUTOFF, full_list=False, strict=True),
        TensorBlock(
            values=vectors.reshape(-1, 3, 1).copy(),
            samples=Labels(names, samples),
            components=[Labels.range("xyz", 3)],
            properties=Labels.range("distance", 1),
        ),
    )

    return system


@pytest.fixture
def systems():
    positions = 1.8 * np.array(
        list(itertools.product(range(2), repeat=3)), dtype=np.float64
    ) + 0.1 * np.random.default_rng(0).standard_normal((8, 3))
    cubic = make_system(positions, cell=3.6 * np.eye(3), pbc=[True, True, True])

    positions = np.array(
        [[0.0, 0.0, 0.0], [1.6, 0.0, 0.0], [0.0, 1.7, 0.2], [5.0, 5.0, 5.0]],
        dtype=np.float64,
    )
    isolated = make_system(positions, cell=np.zeros((3, 3)), pbc=[False, False, False])

    return [cubic, isolated]


def torch_lj_model(dtype="float64", seed=None, supported_devices=None):
    return TorchLennardJonesModel(
        atomic_type=12,
        cutoff=CUTOFF,
        epsilon=EPSILON,
        sigma=SIGMA,
        length_unit="angstrom",
        energy_unit="eV",
        dtype=dtype,
        supported_devices=supported_devices,
        seed=seed,
    )


def numpy_lj_model(seed=None):
    return LennardJonesModel(
        atomic_type=12,
        cutoff=CUTOFF,
        epsilon=EPSILON,
        sigma=SIGMA,
        length_unit="angstrom",
        energy_unit="eV",
        seed=seed,
    )


def test_capabilities():
    model = torch_lj_model()
    capabilities = model.capabilities()

    assert capabilities.atomic_types == [12]
    assert capabilities.interaction_range == CUTOFF
    assert capabilities.length_unit == "angstrom"
    assert capabilities.dtype == "float64"
    assert capabilities.supported_devices == ["cpu", "cuda", "rocm"]

    outputs = {(o.name, o.sample_kind): o for o in capabilities.outputs}
    for variant in ["", "/doubled"]:
        for name in ["energy", "energy_ensemble", "energy_uncertainty"]:
            assert outputs[(name + variant, "system")].gradients == [
                "positions",
                "strain",
            ]
            assert outputs[(name + variant, "atom")].gradients == []

        assert (
            outputs[("non_conservative_force" + variant, "atom")].unit == "eV/angstrom"
        )
        assert (
            outputs[("non_conservative_stress" + variant, "system")].unit
            == "eV/angstrom^3"
        )

    assert len(outputs) == 16
    assert model.metadata().name == "Test Lennard-Jones (PyTorch)"

    model_f32 = torch_lj_model(dtype="float32")
    assert model_f32.capabilities().supported_devices == [
        "cpu",
        "cuda",
        "rocm",
        "metal",
    ]

    message = "dtype must be one of \\['float32', 'float64'\\], got 'float16'"
    with pytest.raises(ValueError, match=message):
        torch_lj_model(dtype="float16")


def test_requested_pair_lists(systems):
    model = torch_lj_model()

    outputs = model.capabilities().outputs
    assert model.requested_pair_lists(outputs) == [PAIRS_OPTIONS]
    assert model.requested_inputs(outputs) == []

    for name in ["energy", "energy_ensemble/doubled"]:
        output = Quantity(name=name, unit="", sample_kind="system")
        assert model.requested_pair_lists([output]) == [PAIRS_OPTIONS]

    output = Quantity(name="non_conservative_force", unit="", sample_kind="atom")
    assert model.requested_pair_lists([output]) == [PAIRS_OPTIONS]

    outputs_no_pairs = [
        Quantity(name="energy_uncertainty", unit="", sample_kind="system"),
        Quantity(name="energy_uncertainty/doubled", unit="", sample_kind="atom"),
        Quantity(name="non_conservative_stress", unit="", sample_kind="system"),
    ]
    assert model.requested_pair_lists(outputs_no_pairs) == []

    systems_no_pairs = [
        System(
            system.types.copy(),
            system.positions.copy(),
            system.cell.copy(),
            system.pbc.copy(),
            length_unit="angstrom",
        )
        for system in systems
    ]
    execute_model(model, systems_no_pairs, outputs_no_pairs, check_consistency=True)


def test_matches_numpy_model(systems):
    torch_model = torch_lj_model(seed=0)
    np_model = numpy_lj_model(seed=0)

    outputs = [
        Quantity(
            name="energy",
            unit="eV",
            sample_kind="system",
            gradients=["positions", "strain"],
        ),
        Quantity(name="energy", unit="eV", sample_kind="atom"),
        Quantity(
            name="energy_ensemble",
            unit="eV",
            sample_kind="system",
            gradients=["positions", "strain"],
        ),
        Quantity(name="energy_ensemble", unit="eV", sample_kind="atom"),
        Quantity(
            name="energy_uncertainty",
            unit="eV",
            sample_kind="system",
            gradients=["positions", "strain"],
        ),
        Quantity(name="energy_uncertainty", unit="eV", sample_kind="atom"),
    ]
    torch_results = execute_model(torch_model, systems, outputs, check_consistency=True)
    np_results = execute_model(np_model, systems, outputs, check_consistency=True)

    for request, t_res, n_res in zip(outputs, torch_results, np_results, strict=True):
        t_block = t_res.block()
        n_block = n_res.block()
        if request.sample_kind == "atom":
            t_sorted = mts.sort_block(t_block, axes="samples")
            n_sorted = mts.sort_block(n_block, axes="samples")
            # Samples are shuffled when selected_atoms is None
            assert not t_block.samples == t_sorted.samples
            np.testing.assert_allclose(t_sorted.values.numpy(), n_sorted.values)
        else:
            np.testing.assert_allclose(t_block.values.numpy(), n_block.values)
            for parameter in request.gradients:
                np.testing.assert_allclose(
                    t_block.gradient(parameter).values.numpy(),
                    n_block.gradient(parameter).values,
                    rtol=1e-6,
                    atol=1e-10,
                )


def test_selected_atoms(systems):
    torch_model = torch_lj_model()
    np_model = numpy_lj_model()

    outputs = [
        Quantity(
            name="energy",
            unit="eV",
            sample_kind="system",
            gradients=["positions", "strain"],
        ),
        Quantity(name="energy", unit="eV", sample_kind="atom"),
    ]
    selected = Labels(
        ["system", "atom"],
        np.array([[0, 3], [1, 1], [0, 0], [0, 6], [1, 2]], dtype=np.int32),
    )

    torch_results = execute_model(
        torch_model, systems, outputs, selected_atoms=selected, check_consistency=True
    )
    np_results = execute_model(
        np_model, systems, outputs, selected_atoms=selected, check_consistency=True
    )

    # Per-atom sample order matches selected_atoms
    np.testing.assert_equal(
        torch_results[1].block().samples.values.numpy(), selected.values
    )
    np.testing.assert_allclose(
        torch_results[1].block().values.numpy(), np_results[1].block().values
    )

    t_energy = torch_results[0].block()
    n_energy = np_results[0].block()
    np.testing.assert_allclose(t_energy.values.numpy(), n_energy.values)
    for parameter in ["positions", "strain"]:
        np.testing.assert_allclose(
            t_energy.gradient(parameter).values.numpy(),
            n_energy.gradient(parameter).values,
            rtol=1e-6,
            atol=1e-10,
        )


def test_non_conservative_and_doubled(systems):
    model = torch_lj_model(seed=42)
    outputs = model.capabilities().outputs
    results = execute_model(model, systems, outputs, check_consistency=True)
    by_key = {
        (output.name, output.sample_kind): result
        for output, result in zip(outputs, results, strict=True)
    }

    for (name, sample_kind), result in by_key.items():
        if name.endswith("/doubled"):
            continue
        block = result.block()
        doubled = by_key[(name + "/doubled", sample_kind)].block()

        assert doubled.samples == block.samples
        np.testing.assert_allclose(
            doubled.values.numpy(), 2.0 * block.values.numpy(), rtol=1e-12
        )
        assert doubled.gradients_list() == block.gradients_list()
        for parameter, gradient in block.gradients():
            np.testing.assert_allclose(
                doubled.gradient(parameter).values.numpy(),
                2.0 * gradient.values.numpy(),
                rtol=1e-12,
            )

    # Check non_conservative_force is close to the conservative LJ forces
    nc_force = mts.sort_block(
        by_key[("non_conservative_force", "atom")].block(), axes="samples"
    )
    forces = -by_key[("energy", "system")].block().gradient("positions").values.numpy()
    assert np.all(nc_force.values.numpy() != forces)
    for system_i in range(2):
        mask = nc_force.samples.values.numpy()[:, 0] == system_i
        noise = 0.3 * np.mean(np.abs(forces[mask]))
        assert np.all(np.abs(nc_force.values.numpy()[mask] - forces[mask]) < 5 * noise)


@pytest.mark.parametrize("device", DEVICES)
def test_device(systems, device):
    model = torch_lj_model(dtype="float32", seed=0)
    np_model = numpy_lj_model(seed=0)

    request = Quantity(
        name="energy",
        unit="eV",
        sample_kind="system",
        gradients=["positions", "strain"],
    )
    expected = execute_model(np_model, [systems[0]], [request])[0].block()

    system = systems[0]
    system.arrays_backend = "torch"
    on_device = system.to(device=device, dtype=torch.float32)

    (result,) = execute_model(model, [on_device], [request], check_consistency=True)
    block = result.block()
    assert block.values.device.type == device
    np.testing.assert_allclose(block.values.cpu().numpy(), expected.values, rtol=1e-5)
    for parameter in ["positions", "strain"]:
        gradient = block.gradient(parameter)
        assert gradient.values.device.type == device
        np.testing.assert_allclose(
            gradient.values.cpu().numpy(),
            expected.gradient(parameter).values,
            rtol=1e-4,
            atol=1e-4,
        )


def test_python_plugin(tmp_path, systems):
    path = tmp_path / "lennard-jones-torch.py"
    path.write_text(
        textwrap.dedent(f"""
        from metatomic.testing import TorchLennardJonesModel
        model = TorchLennardJonesModel(
            atomic_type=12, cutoff={CUTOFF}, epsilon={EPSILON}, sigma={SIGMA},
            length_unit='angstrom', energy_unit='eV',
        )""")
    )

    model = load_model(str(path), plugin_name="python")
    output = Quantity(
        name="energy",
        unit="eV",
        sample_kind="system",
        gradients=["positions", "strain"],
    )
    results = execute_model(model, systems, [output], check_consistency=True)
    expected = execute_model(
        numpy_lj_model(), systems, [output], check_consistency=True
    )
    np.testing.assert_allclose(results[0].block().values, expected[0].block().values)

import itertools
import textwrap

import metatensor as mts
import numpy as np
import pytest
from metatensor import Labels, TensorBlock

from metatomic import (
    PairListOptions,
    Quantity,
    System,
    execute_model,
    load_model,
)
from metatomic.testing import LennardJonesModel


CUTOFF = 3.4
SIGMA = 1.5
EPSILON = 23.0

PAIRS_OPTIONS = PairListOptions(cutoff=CUTOFF, full_list=False, strict=True)


def compute_pairs(positions, cell, pbc):
    """Brute-force half pair list"""
    ranges = [range(-2, 3) if periodic else range(1) for periodic in pbc]

    samples = []
    vectors = []
    for i, j in itertools.combinations_with_replacement(range(len(positions)), 2):
        for shift in itertools.product(*ranges):
            if i == j and shift <= tuple(-s for s in shift):
                # only include one of the (i, i, S) and (i, i, -S) pairs, and skip
                # (i, i, 0)
                continue

            vector = positions[j] - positions[i] + np.array(shift) @ cell
            if np.linalg.norm(vector) < CUTOFF:
                samples.append([i, j, *shift])
                vectors.append(vector)

    return np.array(samples, dtype=np.int32), np.array(vectors).reshape(-1, 3)


def make_system(positions, cell, pbc):
    system = System(
        "angstrom",
        np.full(len(positions), 12, dtype=np.int32),
        positions,
        cell,
        np.array(pbc),
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
    # perturbed simple cubic lattice in a periodic cell
    positions = 1.8 * np.array(
        list(itertools.product(range(2), repeat=3)), dtype=np.float64
    ) + 0.1 * np.random.default_rng(0).standard_normal((8, 3))
    cubic = make_system(positions, cell=3.6 * np.eye(3), pbc=[True, True, True])

    # non-periodic cluster containing an isolated atom
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.6, 0.0, 0.0], [0.0, 1.7, 0.2], [5.0, 5.0, 5.0]],
        dtype=np.float64,
    )
    isolated = make_system(positions, cell=np.zeros((3, 3)), pbc=[False, False, False])

    return [cubic, isolated]


def lj_model(seed=None):
    return LennardJonesModel(
        atomic_type=12,
        cutoff=CUTOFF,
        epsilon=EPSILON,
        sigma=SIGMA,
        length_unit="angstrom",
        energy_unit="eV",
        seed=seed,
    )


def reference_energies(system):
    """Per-atom energies, computed with a simple loop over pairs"""
    shift = 4 * EPSILON * ((SIGMA / CUTOFF) ** 12 - (SIGMA / CUTOFF) ** 6)

    energies = np.zeros(len(system.positions))
    samples, vectors = compute_pairs(system.positions, system.cell, system.pbc)
    for (i, j, *_), vector in zip(samples, vectors, strict=True):
        r = np.linalg.norm(vector)
        energy = 4 * EPSILON * ((SIGMA / r) ** 12 - (SIGMA / r) ** 6) - shift
        energies[i] += 0.5 * energy
        energies[j] += 0.5 * energy

    return energies


def test_capabilities():
    model = lj_model()
    capabilities = model.capabilities()

    assert capabilities.atomic_types == [12]
    assert capabilities.interaction_range == CUTOFF
    assert capabilities.length_unit == "angstrom"
    assert capabilities.dtype == "float64"

    outputs = {(o.name, o.sample_kind): o for o in capabilities.outputs}
    for variant in ["", "/doubled"]:
        for name in ["energy", "energy_ensemble", "energy_uncertainty"]:
            assert outputs[(name + variant, "system")].gradients == [
                "positions",
                "strain",
            ]
            assert outputs[(name + variant, "atom")].gradients == []

        output = outputs[("non_conservative_force" + variant, "atom")]
        assert output.unit == "eV/angstrom"

        output = outputs[("non_conservative_stress" + variant, "system")]
        assert output.unit == "eV/angstrom^3"

    assert len(outputs) == 16

    assert model.metadata().name == "Test Lennard-Jones (NumPy)"


def test_requested_pair_lists(systems):
    model = lj_model()

    outputs = model.capabilities().outputs
    assert model.requested_pair_lists(outputs) == [PAIRS_OPTIONS]
    assert model.requested_inputs(outputs) == []

    for name in ["energy", "energy_ensemble/doubled"]:
        output = Quantity(name=name, unit="", sample_kind="system")
        assert model.requested_pair_lists([output]) == [PAIRS_OPTIONS]

    output = Quantity(name="non_conservative_force", unit="", sample_kind="atom")
    assert model.requested_pair_lists([output]) == [PAIRS_OPTIONS]

    # these outputs do not need pairs
    outputs = [
        Quantity(name="energy_uncertainty", unit="", sample_kind="system"),
        Quantity(name="energy_uncertainty/doubled", unit="", sample_kind="atom"),
        Quantity(name="non_conservative_stress", unit="", sample_kind="system"),
    ]
    assert model.requested_pair_lists(outputs) == []

    # the model runs without the pairs
    systems_no_pairs = [
        System(
            "angstrom",
            system.types.copy(),
            system.positions.copy(),
            system.cell.copy(),
            system.pbc.copy(),
        )
        for system in systems
    ]

    for system in systems_no_pairs:
        assert len(system.known_pairs()) == 0

    execute_model(model, systems_no_pairs, outputs, check_consistency=True)


def test_energy(systems):
    model = lj_model()

    outputs = [
        Quantity(name="energy", unit="eV", sample_kind="system"),
        Quantity(name="energy", unit="eV", sample_kind="atom"),
    ]
    energy, energy_per_atom = execute_model(
        model, systems, outputs, check_consistency=True
    )

    expected = [reference_energies(s) for s in systems]
    assert np.all(expected[0] != 0.0)
    assert expected[1][3] == 0.0

    block = energy.block()
    assert block.samples == Labels.range("system", 2)
    assert block.properties == Labels.range("energy", 1)
    assert block.gradients_list() == []
    np.testing.assert_allclose(block.values, [[e.sum()] for e in expected])

    block = energy_per_atom.block()
    sorted = mts.sort_block(block, axes="samples")
    np.testing.assert_equal(
        sorted.samples.values, [[0, a] for a in range(8)] + [[1, a] for a in range(4)]
    )
    np.testing.assert_allclose(sorted.values, np.concatenate(expected).reshape(-1, 1))

    # check that the samples are shuffled
    assert block.samples.names == sorted.samples.names
    assert not block.samples == sorted.samples


def test_energy_gradients(systems):
    model = lj_model()
    output = Quantity(
        name="energy",
        unit="eV",
        sample_kind="system",
        gradients=["positions", "strain"],
    )

    system = systems[0]

    results = execute_model(model, [system], [output], check_consistency=True)
    block = results[0].block()
    assert block.gradients_list() == ["positions", "strain"]

    def energy(positions, cell):
        return reference_energies(make_system(positions, cell, pbc=system.pbc)).sum()

    delta = 1e-6

    gradient = block.gradient("positions")
    assert gradient.samples == Labels(
        ["sample", "system", "atom"], np.array([[0, 0, a] for a in range(8)])
    )
    assert gradient.components == [Labels.range("xyz", 3)]
    expected = np.zeros((8, 3))
    for atom, xyz in itertools.product(range(8), range(3)):
        displaced = system.positions.copy()
        displaced[atom, xyz] += delta
        energy_plus = energy(displaced, system.cell)
        displaced[atom, xyz] -= 2 * delta
        energy_minus = energy(displaced, system.cell)
        expected[atom, xyz] = (energy_plus - energy_minus) / (2 * delta)

    np.testing.assert_allclose(gradient.values.reshape(8, 3), expected, rtol=1e-6)
    # sum of forces is zero
    np.testing.assert_allclose(gradient.values.sum(axis=0), 0.0, atol=1e-10)

    gradient = block.gradient("strain")
    assert gradient.samples == Labels.range("sample", 1)
    assert gradient.components == [Labels.range("xyz_1", 3), Labels.range("xyz_2", 3)]
    expected = np.zeros((3, 3))
    for a, b in itertools.product(range(3), range(3)):
        strain = np.eye(3)
        strain[a, b] += delta
        energy_plus = energy(system.positions @ strain, system.cell @ strain)
        strain[a, b] -= 2 * delta
        energy_minus = energy(system.positions @ strain, system.cell @ strain)
        expected[a, b] = (energy_plus - energy_minus) / (2 * delta)

    np.testing.assert_allclose(gradient.values.reshape(3, 3), expected, rtol=1e-6)


def test_selected_atoms(systems):
    model = lj_model()

    outputs = [
        Quantity(
            name="energy",
            unit="eV",
            sample_kind="system",
            gradients=["positions", "strain"],
        ),
        Quantity(name="energy", unit="eV", sample_kind="atom"),
    ]
    complete = execute_model(model, systems, outputs, check_consistency=True)

    expected = [reference_energies(s) for s in systems]

    # split all atoms in two sets, the sum of the outputs for both sets should be the
    # same as the outputs for all atoms
    first = np.array([[0, 3], [1, 1], [0, 0], [0, 6], [1, 2]], dtype=np.int32)
    second = np.array(
        [[0, 1], [0, 2], [0, 4], [0, 5], [0, 7], [1, 0], [1, 3]], dtype=np.int32
    )

    energy = 0.0
    positions_gradients = 0.0
    strain_gradients = 0.0
    for selected in [first, second]:
        results = execute_model(
            model,
            systems,
            outputs,
            selected_atoms=Labels(["system", "atom"], selected),
            check_consistency=True,
        )

        # the samples follow the order of `selected_atoms`
        block = results[1].block()
        np.testing.assert_equal(block.samples.values, selected)
        np.testing.assert_allclose(
            block.values, [[expected[s][a]] for s, a in selected]
        )

        block = results[0].block()
        assert block.samples == Labels.range("system", 2)
        energy += block.values
        positions_gradients += block.gradient("positions").values
        strain_gradients += block.gradient("strain").values

    block = complete[0].block()
    np.testing.assert_allclose(energy, block.values)
    np.testing.assert_allclose(
        positions_gradients, block.gradient("positions").values, atol=1e-10
    )
    np.testing.assert_allclose(
        strain_gradients, block.gradient("strain").values, atol=1e-10
    )

    # only systems containing selected atoms are included in per-system outputs
    selected_atoms = Labels(["system", "atom"], np.array([[1, 0], [1, 1]]))
    outputs = [
        Quantity(
            name="energy",
            unit="eV",
            sample_kind="system",
            gradients=["positions", "strain"],
        ),
        Quantity(
            name="energy_uncertainty",
            unit="eV",
            sample_kind="system",
        ),
        Quantity(
            name="non_conservative_stress",
            unit="eV/angstrom^3",
            sample_kind="system",
        ),
    ]
    results = execute_model(
        model,
        systems,
        outputs,
        selected_atoms=selected_atoms,
        check_consistency=True,
    )

    for result in results:
        assert result.block().samples == Labels(["system"], np.array([[1]]))

    block = results[0].block()
    np.testing.assert_allclose(block.values, [[expected[1][0] + expected[1][1]]])
    assert block.gradient("positions").samples == Labels(
        ["sample", "system", "atom"], np.array([[0, 1, a] for a in range(4)])
    )


def test_ensemble(systems):
    model = lj_model()

    outputs = [
        Quantity(
            name="energy",
            unit="eV",
            sample_kind="system",
            gradients=["positions", "strain"],
        ),
        Quantity(
            name="energy_ensemble",
            unit="eV",
            sample_kind="system",
            gradients=["positions", "strain"],
        ),
        Quantity(name="energy", unit="eV", sample_kind="atom"),
        Quantity(name="energy_ensemble", unit="eV", sample_kind="atom"),
    ]
    energy, ensemble, energy_per_atom, ensemble_per_atom = execute_model(
        model, systems, outputs, check_consistency=True
    )

    energy = energy.block()
    ensemble = ensemble.block()
    assert ensemble.properties == Labels.range("energy", 16)
    assert ensemble.samples == energy.samples
    np.testing.assert_equal(ensemble.values, np.tile(energy.values, (1, 16)))
    for parameter in ["positions", "strain"]:
        gradient = ensemble.gradient(parameter)
        assert gradient.samples == energy.gradient(parameter).samples
        np.testing.assert_equal(
            gradient.values,
            np.repeat(energy.gradient(parameter).values, 16, axis=-1),
        )

    sorted_ensemble = mts.sort_block(ensemble_per_atom.block(), axes="samples")
    sorted_energy = mts.sort_block(energy_per_atom.block(), axes="samples")
    assert sorted_ensemble.samples == sorted_energy.samples
    np.testing.assert_equal(
        sorted_ensemble.values, np.tile(sorted_energy.values, (1, 16))
    )


def test_uncertainty(systems):
    model = lj_model()

    outputs = [
        Quantity(
            name="energy_uncertainty",
            unit="eV",
            sample_kind="system",
            gradients=["positions", "strain"],
        ),
        Quantity(name="energy_uncertainty", unit="eV", sample_kind="atom"),
    ]
    uncertainty, uncertainty_per_atom = execute_model(
        model, systems, outputs, check_consistency=True
    )

    block = uncertainty.block()
    assert block.properties == Labels.range("energy", 1)
    np.testing.assert_allclose(block.values, [[0.001 * 8**2], [0.001 * 4**2]])
    assert block.gradient("positions").values.shape == (12, 3, 1)
    np.testing.assert_equal(block.gradient("positions").values, 0.0)
    assert block.gradient("strain").values.shape == (2, 3, 3, 1)
    np.testing.assert_equal(block.gradient("strain").values, 0.0)

    block = uncertainty_per_atom.block()
    samples = np.asarray(block.samples.values)
    np.testing.assert_allclose(
        block.values[:, 0], np.where(samples[:, 0] == 0, 0.001 * 8**2, 0.001 * 4**2)
    )


def test_non_conservative(systems):
    model = lj_model(seed=0)

    outputs = [
        Quantity(
            name="energy", unit="eV", sample_kind="system", gradients=["positions"]
        ),
        Quantity(name="non_conservative_force", unit="eV/A", sample_kind="atom"),
        Quantity(name="non_conservative_stress", unit="eV/A^3", sample_kind="system"),
    ]
    energy, force, stress = execute_model(
        model, systems, outputs, check_consistency=True
    )

    block = force.block()
    assert block.components == [Labels.range("xyz", 3)]
    assert block.properties == Labels.range("non_conservative_force", 1)

    # the forces are close to the Lennard-Jones forces
    sorted_block = mts.sort_block(block, axes="samples")
    forces = -energy.block().gradient("positions").values
    assert np.all(sorted_block.values != forces)
    for system in range(2):
        mask = sorted_block.samples.values[:, 0] == system
        noise = 0.3 * np.mean(np.abs(forces[mask]))
        assert np.all(np.abs(sorted_block.values[mask] - forces[mask]) < 5 * noise)

    block = stress.block()
    assert block.samples == Labels.range("system", 2)
    assert block.components == [Labels.range("xyz_1", 3), Labels.range("xyz_2", 3)]
    assert block.properties == Labels.range("non_conservative_stress", 1)
    assert block.values.shape == (2, 3, 3, 1)


@pytest.mark.parametrize("seed", [None, 42])
def test_doubled_variant(seed, systems):
    model = lj_model(seed=seed)

    # request all outputs, including the `<name>/doubled` variants
    outputs = model.capabilities().outputs
    results = execute_model(model, systems, outputs, check_consistency=True)
    results = {
        (output.name, output.sample_kind): result
        for output, result in zip(outputs, results, strict=True)
    }

    for (name, sample_kind), result in results.items():
        if name.endswith("/doubled"):
            continue

        block = result.block()
        doubled = results[(name + "/doubled", sample_kind)].block()

        assert doubled.samples == block.samples
        np.testing.assert_equal(doubled.values, 2.0 * block.values)
        assert doubled.gradients_list() == block.gradients_list()
        for parameter, gradient in block.gradients():
            np.testing.assert_equal(
                doubled.gradient(parameter).values, 2.0 * gradient.values
            )


def test_errors(systems):
    model = lj_model()

    message = "this model can not compute 'energy' with sample kind 'atom_pair'"
    with pytest.raises(ValueError, match=message):
        execute_model(
            model,
            systems,
            [Quantity(name="energy", unit="eV", sample_kind="atom_pair")],
        )

    message = (
        "this model can not compute explicit gradients of 'energy' "
        "with respect to 'positions'"
    )
    with pytest.raises(ValueError, match=message):
        execute_model(
            model,
            systems,
            [
                Quantity(
                    name="energy",
                    unit="eV",
                    sample_kind="atom",
                    gradients=["positions"],
                )
            ],
        )


def test_python_plugin(tmp_path, systems):
    path = tmp_path / "lennard-jones.py"
    path.write_text(
        textwrap.dedent(f"""
        from metatomic.testing import LennardJonesModel
        model = LennardJonesModel(
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
    expected = execute_model(lj_model(), systems, [output], check_consistency=True)

    np.testing.assert_allclose(results[0].block().values, expected[0].block().values)

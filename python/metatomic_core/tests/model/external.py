import metatensor
import numpy as np
import pytest
from metatensor import Labels, TensorBlock

import metatomic as mta
from metatomic import (
    ExternalModel,
    PairListOptions,
    Quantity,
    System,
    execute_model,
    load_model,
    load_plugin,
)


PLUGIN = "metatomic-lj-plugin"
MODEL = "metatomic-lj-model"

OPTIONS = {
    "sigma": "1",
    "epsilon": "1",
    "atomic_type": "1",
    "cutoff": "3.0",
    "length_unit": "A",
    "energy_unit": "kJ/mol",
}


@pytest.fixture(scope="module", autouse=True)
def lj_plugin():
    load_plugin(mta.testing.plugin_path("lj"))

    # Register a data wrapper for C++ arrays, this should be moved to metatensor
    metatensor.register_external_data_wrapper(
        "metatensor::SimpleDataArray", metatensor.ExternalCpuArray
    )


def lj_system(distance):
    """System with 2 atoms separated by ``distance``, and the corresponding pairs"""
    system = System(
        "Angstrom",
        np.array([1, 1], dtype=np.int32),
        np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=np.float64),
        np.zeros((3, 3), dtype=np.float64),
        np.array([True, True, True]),
    )

    pairs = TensorBlock(
        values=np.array([[[distance], [0.0], [0.0]]], dtype=np.float64),
        samples=Labels(
            [
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            np.array([[0, 1, 0, 0, 0]], dtype=np.int32),
        ),
        components=[Labels.range("xyz", 3)],
        properties=Labels.range("distance", 1),
    )
    options = PairListOptions(cutoff=3.0, full_list=False, strict=False)
    system.add_pairs(options, pairs)

    return system


def lj_energy(distance, sigma=1.0, epsilon=1.0, cutoff=3.0):
    sigma_r_6 = (sigma / distance) ** 6
    sigma_cutoff_6 = (sigma / cutoff) ** 6
    return (
        4.0
        * epsilon
        * (
            sigma_r_6 * sigma_r_6
            - sigma_r_6
            - sigma_cutoff_6 * sigma_cutoff_6
            + sigma_cutoff_6
        )
    )


def lj_force(distance):
    step = 1e-6
    return (lj_energy(distance + step) - lj_energy(distance - step)) / (2.0 * step)


GLOBAL_ENERGY = Quantity(
    name="energy", unit="kJ/mol", sample_kind="system", gradients=["positions"]
)
PER_ATOM_ENERGY = Quantity(name="energy", unit="kJ/mol", sample_kind="atom")


def test_external_model():
    model = load_model(MODEL, OPTIONS)
    assert isinstance(model, ExternalModel)

    capabilities = model.capabilities()
    assert capabilities.atomic_types == [1]
    assert capabilities.interaction_range == 3.0
    assert capabilities.length_unit == "A"
    assert capabilities.supported_devices == ["cpu"]
    assert capabilities.dtype == "float64"
    assert capabilities.outputs == [GLOBAL_ENERGY, PER_ATOM_ENERGY]

    metadata = model.metadata()
    assert metadata.name == "Lennard-Jones test model"
    assert metadata.authors == ["metatomic authors"]

    assert model.requested_pair_lists() == [
        PairListOptions(cutoff=3.0, full_list=False, strict=False)
    ]
    assert model.requested_inputs() == []

    message = "ExternalModel.execute_inner\\(\\) should never be called directly"
    with pytest.raises(RuntimeError, match=message):
        model.execute_inner([lj_system(2.0)], None, [GLOBAL_ENERGY])


def test_energy_and_forces():
    model = load_model(MODEL, OPTIONS)
    systems = [lj_system(2.0), lj_system(1.2)]

    # execute model twice to make sure the model remains valid
    for _ in range(2):
        results = execute_model(
            model,
            systems,
            [GLOBAL_ENERGY, PER_ATOM_ENERGY],
            check_consistency=True,
        )
        assert len(results) == 2

        # global energy
        block = results[0].block(0)
        assert block.samples == Labels(["system"], np.array([[0], [1]]))
        np.testing.assert_allclose(
            block.values[:, 0], [lj_energy(2.0), lj_energy(1.2)], rtol=1e-12
        )

        # forces from the global energy
        gradient = block.gradient("positions")
        assert gradient.samples == Labels(
            ["sample", "system", "atom"],
            np.array([[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]]),
        )
        np.testing.assert_allclose(
            gradient.values[:, 0, 0],
            [-lj_force(2.0), lj_force(2.0), -lj_force(1.2), lj_force(1.2)],
            rtol=1e-8,
        )

        # per-atom energy
        block = results[1].block(0)
        assert block.samples == Labels(
            ["system", "atom"], np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        )
        np.testing.assert_allclose(
            block.values[:, 0],
            0.5 * np.array([lj_energy(2.0)] * 2 + [lj_energy(1.2)] * 2),
            rtol=1e-12,
        )


def test_selected_atoms():
    model = load_model(MODEL, OPTIONS)
    systems = [lj_system(2.0), lj_system(1.2)]

    # select only one atom in each system
    selected_atoms = Labels(["system", "atom"], np.array([[0, 0], [1, 1]]))

    results = execute_model(
        model,
        systems,
        [GLOBAL_ENERGY, PER_ATOM_ENERGY],
        selected_atoms=selected_atoms,
        check_consistency=True,
    )
    assert len(results) == 2

    # global energy, only containing half of the pair energy
    block = results[0].block(0)
    assert block.samples == Labels(["system"], np.array([[0], [1]]))
    np.testing.assert_allclose(
        block.values[:, 0], 0.5 * np.array([lj_energy(2.0), lj_energy(1.2)]), rtol=1e-12
    )

    # forces from the global energy, including on the unselected atoms
    gradient = block.gradient("positions")
    assert gradient.samples == Labels(
        ["sample", "system", "atom"],
        np.array([[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]]),
    )
    np.testing.assert_allclose(
        gradient.values[:, 0, 0],
        0.5 * np.array([-lj_force(2.0), lj_force(2.0), -lj_force(1.2), lj_force(1.2)]),
        rtol=1e-8,
    )

    # per-atom energy, only for the selected atoms
    block = results[1].block(0)
    assert block.samples == selected_atoms
    np.testing.assert_allclose(
        block.values[:, 0], 0.5 * np.array([lj_energy(2.0), lj_energy(1.2)]), rtol=1e-12
    )

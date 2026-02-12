import metatomic_lj_test
import numpy as np
import pytest
import torch
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    System,
)
from metatomic.torch.ase_calculator import MetatomicCalculator
from metatomic.torch.heat_flux import (
    HardyHeatFluxWrapper,
    HeatFluxWrapper,
    check_collisions,
    collisions_to_replicas,
    generate_replica_atoms,
    unfold_system,
    wrap_positions,
)


@pytest.fixture
def model():
    return metatomic_lj_test.lennard_jones_model(
        atomic_type=18,
        cutoff=7.0,
        sigma=3.405,
        epsilon=0.01032,
        length_unit="Angstrom",
        energy_unit="eV",
        with_extension=False,
    )


@pytest.fixture
def atoms():
    cell = np.array([[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]])
    positions = np.array([[3.0, 3.0, 3.0]])
    atoms = Atoms(f"Ar", scaled_positions=positions, cell=cell, pbc=True).repeat(
        (2, 2, 2)
    )
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=300, rng=np.random.default_rng(42)
    )
    return atoms


def _make_scalar_tensormap(values: torch.Tensor, property_name: str) -> TensorMap:
    block = TensorBlock(
        values=values,
        samples=Labels(
            ["atoms"],
            torch.arange(values.shape[0], device=values.device).reshape(-1, 1),
        ),
        components=[],
        properties=Labels([property_name], torch.tensor([[0]], device=values.device)),
    )
    return TensorMap(Labels("_", torch.tensor([[0]], device=values.device)), [block])


def _make_velocity_tensormap(values: torch.Tensor) -> TensorMap:
    block = TensorBlock(
        values=values,
        samples=Labels(
            ["atoms"],
            torch.arange(values.shape[0], device=values.device).reshape(-1, 1),
        ),
        components=[
            Labels(
                ["xyz"],
                torch.arange(3, device=values.device).reshape(-1, 1),
            )
        ],
        properties=Labels(["velocity"], torch.tensor([[0]], device=values.device)),
    )
    return TensorMap(Labels("_", torch.tensor([[0]], device=values.device)), [block])


def _make_system_with_data(positions: torch.Tensor, cell: torch.Tensor) -> System:
    types = torch.tensor([1] * len(positions), dtype=torch.int32)
    system = System(
        types=types,
        positions=positions,
        cell=cell,
        pbc=torch.tensor([True, True, True]),
    )
    masses = torch.ones((len(positions), 1), dtype=positions.dtype)
    velocities = torch.zeros((len(positions), 3, 1), dtype=positions.dtype)
    system.add_data("masses", _make_scalar_tensormap(masses, "mass"))
    system.add_data("velocities", _make_velocity_tensormap(velocities))
    return system


def test_wrap_positions_cubic_matches_expected():
    cell = torch.eye(3) * 2.0
    positions = torch.tensor([[-0.1, 0.0, 0.0], [2.1, 1.0, -0.5]])
    wrapped = wrap_positions(positions, cell)
    expected = torch.tensor([[1.9, 0.0, 0.0], [0.1, 1.0, 1.5]])
    assert torch.allclose(wrapped, expected)


def test_check_collisions_cubic_axis_order():
    cell = torch.eye(3) * 2.0
    positions = torch.tensor([[0.1, 1.0, 1.9]])
    collisions, norm_coords = check_collisions(cell, positions, cutoff=0.2, skin=0.0)
    assert torch.allclose(norm_coords, positions)
    assert collisions.shape == (1, 6)
    assert collisions[0].tolist() == [True, False, False, False, False, True]


def test_generate_replica_atoms_cubic_offsets():
    types = torch.tensor([1])
    positions = torch.tensor([[0.1, 1.0, 1.0]])
    cell = torch.eye(3) * 2.0
    collisions = torch.tensor([[True, False, False, False, False, False]])
    replicas = collisions_to_replicas(collisions)
    replica_idx, replica_types, replica_positions = generate_replica_atoms(
        types, positions, cell, replicas
    )
    assert replica_idx.tolist() == [0]
    assert replica_types.tolist() == [1]
    assert torch.allclose(
        replica_positions, positions + torch.tensor([[2.0, 0.0, 0.0]])
    )


def test_wrap_positions_triclinic_fractional_bounds_and_shift():
    cell = torch.tensor(
        [
            [2.0, 0.3, 0.2],
            [0.1, 1.7, 0.4],
            [0.2, 0.5, 1.9],
        ]
    )
    positions = torch.tensor(
        [
            [-0.1, 0.0, 0.0],
            [2.1, 1.6, -0.5],
            [4.2, -0.2, 6.1],
        ]
    )
    inv_cell = cell.inverse()
    wrapped = wrap_positions(positions, cell)
    fractional_before = torch.einsum("iv,vk->ik", positions, inv_cell)
    fractional_after = torch.einsum("iv,vk->ik", wrapped, inv_cell)

    assert torch.all(fractional_after >= 0)
    assert torch.all(fractional_after < 1)

    delta_frac = fractional_after - fractional_before
    rounded = torch.round(delta_frac)
    assert torch.allclose(delta_frac, rounded, atol=1e-6, rtol=0)
    assert torch.allclose(rounded, -torch.floor(fractional_before), atol=1e-6, rtol=0)


def test_check_collisions_triclinic_targets():
    cell = torch.tensor(
        [
            [2.0, 0.3, 0.2],
            [0.1, 1.7, 0.4],
            [0.2, 0.5, 1.9],
        ]
    )
    cutoff = 0.2
    inv_cell = cell.inverse()
    recip = inv_cell.T
    norms = torch.linalg.norm(recip, dim=1)
    heights = 1.0 / norms
    norm_vectors = recip / norms[:, None]

    target = torch.stack(
        [
            torch.tensor([0.05, 0.6, 0.6]),
            torch.tensor([heights[0] - 0.05, 0.05, heights[2] - 0.1]),
            torch.tensor([0.3, heights[1] - 0.05, 0.1]),
        ]
    )
    positions = target @ torch.inverse(norm_vectors).T

    collisions, norm_coords = check_collisions(cell, positions, cutoff=cutoff, skin=0.0)
    assert torch.allclose(norm_coords, target, atol=1e-6, rtol=0)

    expected_low = target <= cutoff
    expected_high = target >= heights - cutoff
    expected = torch.hstack([expected_low, expected_high])
    expected = expected[:, [0, 3, 1, 4, 2, 5]]

    assert torch.equal(collisions, expected)


def test_check_collisions_raises_on_small_cell():
    cell = torch.eye(3) * 1.0
    positions = torch.zeros((1, 3))
    with pytest.raises(ValueError, match="Cell is too small"):
        check_collisions(cell, positions, cutoff=0.9, skin=0.2)


def test_collisions_to_replicas_combines_displacements():
    collisions = torch.tensor([[True, False, False, True, False, False]])
    replicas = collisions_to_replicas(collisions)
    assert replicas.shape == (1, 3, 3, 3)
    assert replicas[0, 0, 0, 0].item() is False

    nonzero = torch.nonzero(replicas, as_tuple=False)
    expected = {
        (0, 1, 0, 0),
        (0, 0, 2, 0),
        (0, 1, 2, 0),
    }
    assert {tuple(row.tolist()) for row in nonzero} == expected


def test_generate_replica_atoms_triclinic_offsets():
    cell = torch.tensor(
        [
            [2.0, 0.3, 0.2],
            [0.1, 1.7, 0.4],
            [0.2, 0.5, 1.9],
        ]
    )
    types = torch.tensor([1])
    positions = torch.tensor([[0.2, 0.4, 0.6]])
    collisions = torch.tensor([[True, False, True, False, True, False]])
    replicas = collisions_to_replicas(collisions)
    replica_idx, replica_types, replica_positions = generate_replica_atoms(
        types, positions, cell, replicas
    )

    assert replica_idx.tolist() == [0, 0, 0, 0, 0, 0, 0]
    assert replica_types.tolist() == [1, 1, 1, 1, 1, 1, 1]

    expected_offsets = [cell[0], cell[1], cell[2], cell[0] + cell[1], cell[0] + cell[2], cell[1] + cell[2], cell[0] + cell[1] + cell[2]]
    expected_positions = [positions[0] + offset for offset in expected_offsets]

    for expected in expected_positions:
        assert any(
            torch.allclose(expected, actual, atol=1e-6, rtol=0)
            for actual in replica_positions
        )


def test_unfold_system_adds_replica_and_data():
    cell = torch.eye(3) * 2.0
    positions = torch.tensor([[0.1, 1.0, 1.0]])
    system = _make_system_with_data(positions, cell)
    unfolded = unfold_system(system, cutoff=0.1)

    assert len(unfolded.positions) == 2
    assert torch.all(unfolded.pbc == torch.tensor([False, False, False]))
    assert torch.allclose(unfolded.cell, torch.zeros_like(unfolded.cell))

    masses = unfolded.get_data("masses").block().values
    velocities = unfolded.get_data("velocities").block().values
    assert masses.shape[0] == 2
    assert velocities.shape[0] == 2

    assert torch.allclose(unfolded.positions[0], positions[0])
    assert torch.allclose(
        unfolded.positions[1], positions[0] + torch.tensor([2.0, 0.0, 0.0])
    )


def test_heat_flux_wrapper_requested_inputs():
    class DummyCapabilities:
        def __init__(self):
            self.outputs = {"energy": ModelOutput(quantity="energy", unit="eV")}
            self.length_unit = "A"
            self.interaction_range = 1.0

    class DummyModel:
        def __init__(self):
            self._capabilities = DummyCapabilities()

        def capabilities(self):
            return self._capabilities

        def __call__(self, systems, options, check_consistency):
            results = {}
            if "energy" in options.outputs:
                values = torch.zeros(
                    (len(systems), 1), dtype=systems[0].positions.dtype
                )
                block = TensorBlock(
                    values=values,
                    samples=Labels(
                        ["system"],
                        torch.arange(len(systems), device=values.device).reshape(-1, 1),
                    ),
                    components=[],
                    properties=Labels(
                        ["energy"], torch.tensor([[0]], device=values.device)
                    ),
                )
                results["energy"] = TensorMap(
                    Labels("_", torch.tensor([[0]], device=values.device)), [block]
                )
            return results

    wrapper = HeatFluxWrapper(DummyModel())
    requested = wrapper.requested_inputs()
    assert set(requested.keys()) == {"masses", "velocities"}


def test_unfolded_energy_order_used_for_barycenter():
    class DummyCapabilities:
        def __init__(self):
            self.outputs = {"energy": ModelOutput(quantity="energy", unit="eV")}
            self.length_unit = "A"
            self.interaction_range = 1.0

    class DummyModel:
        def __init__(self):
            self._capabilities = DummyCapabilities()

        def capabilities(self):
            return self._capabilities

        def __call__(self, systems, options, check_consistency):
            system = systems[0]
            n_atoms = len(system.positions)
            values = torch.arange(
                n_atoms, dtype=system.positions.dtype, device=system.positions.device
            ).reshape(-1, 1)
            block = TensorBlock(
                values=values,
                samples=Labels(
                    ["atoms"],
                    torch.arange(n_atoms, device=values.device).reshape(-1, 1),
                ),
                components=[],
                properties=Labels(
                    ["energy"], torch.tensor([[0]], device=values.device)
                ),
            )
            return {
                "energy": TensorMap(
                    Labels("_", torch.tensor([[0]], device=values.device)), [block]
                )
            }

    cell = torch.eye(3) * 10.0
    positions = torch.tensor(
        [
            [0.05, 5.0, 5.0],  # near x_lo -> one replica
            [9.95, 5.5, 5.0],  # near x_hi -> one replica
            [0.05, 6.0, 5.5],  # near x_lo -> one replica
        ]
    )
    system = _make_system_with_data(positions, cell)
    unfolded = unfold_system(system, cutoff=0.1, skin=0.0)
    n_atoms = len(system.positions)
    assert len(unfolded.positions) == n_atoms * 2

    wrapper = HeatFluxWrapper(DummyModel())
    barycenter, atomic_e, total_e = wrapper.barycenter_and_atomic_energies(
        unfolded, n_atoms
    )

    expected_atomic_e = torch.arange(
        len(unfolded.positions),
        dtype=unfolded.positions.dtype,
        device=unfolded.positions.device,
    )
    expected_total_e = expected_atomic_e[:n_atoms].sum()
    expected_barycenter = torch.einsum(
        "i,ik->k", expected_atomic_e[:n_atoms], unfolded.positions[:n_atoms]
    )

    assert torch.allclose(atomic_e, expected_atomic_e)
    assert torch.allclose(total_e, expected_total_e)
    assert torch.allclose(barycenter, expected_barycenter)


def test_heat_flux_wrapper_forward_adds_output(monkeypatch):
    class DummyCapabilities:
        def __init__(self):
            self.outputs = {"energy": ModelOutput(quantity="energy", unit="eV")}
            self.length_unit = "A"
            self.interaction_range = 1.0

    class DummyModel:
        def __init__(self):
            self._capabilities = DummyCapabilities()

        def capabilities(self):
            return self._capabilities

        def __call__(self, systems, options, check_consistency):
            values = torch.zeros((len(systems), 1), dtype=systems[0].positions.dtype)
            block = TensorBlock(
                values=values,
                samples=Labels(
                    ["system"],
                    torch.arange(len(systems), device=values.device).reshape(-1, 1),
                ),
                components=[],
                properties=Labels(
                    ["energy"], torch.tensor([[0]], device=values.device)
                ),
            )
            return {
                "energy": TensorMap(
                    Labels("_", torch.tensor([[0]], device=values.device)), [block]
                )
            }

    def _fake_hf(self, system):
        return torch.tensor(
            [1.0, 2.0, 3.0], device=system.device, dtype=system.positions.dtype
        )

    wrapper = HeatFluxWrapper(DummyModel())
    monkeypatch.setattr(HeatFluxWrapper, "calc_unfolded_heat_flux", _fake_hf)

    cell = torch.eye(3)
    systems = [
        System(
            types=torch.tensor([1], dtype=torch.int32),
            positions=torch.zeros((1, 3)),
            cell=cell,
            pbc=torch.tensor([True, True, True]),
        ),
        System(
            types=torch.tensor([1], dtype=torch.int32),
            positions=torch.ones((1, 3)),
            cell=cell,
            pbc=torch.tensor([True, True, True]),
        ),
    ]

    outputs = {
        "energy": ModelOutput(quantity="energy", unit="eV"),
        "extra::heat_flux": ModelOutput(quantity="heat_flux", unit=""),
    }
    results = wrapper.forward(systems, outputs, None)
    assert "extra::heat_flux" in results
    hf_block = results["extra::heat_flux"].block()
    assert hf_block.values.shape == (2, 3, 1)
    assert torch.allclose(hf_block.values[:, :, 0], torch.tensor([[1.0, 2.0, 3.0]] * 2))


@pytest.mark.parametrize(
    "heat_flux,expected",
    [
        # (HardyHeatFluxWrapper, [[4.0898e-05], [-3.1652e-04], [-2.1660e-04]]),
        (HeatFluxWrapper, [[8.1053e-05], [-1.2710e-05], [-2.8778e-04]]),
    ],
)
def test_heat_flux_wrapper_calc_heat_flux(heat_flux, expected, model, atoms):
    metadata = ModelMetadata()
    wrapper = heat_flux(model.eval())
    cap = wrapper._model.capabilities()
    outputs = cap.outputs.copy()
    outputs["extra::heat_flux"] = ModelOutput(
        quantity="heat_flux",
        unit="",
        explicit_gradients=[],
        per_atom=False,
    )

    new_cap = ModelCapabilities(
        outputs=outputs,
        atomic_types=cap.atomic_types,
        interaction_range=cap.interaction_range,
        length_unit=cap.length_unit,
        supported_devices=cap.supported_devices,
        dtype=cap.dtype,
    )
    heat_model = AtomisticModel(wrapper.eval(), metadata, capabilities=new_cap).to(
        device="cpu"
    )
    calc = MetatomicCalculator(
        heat_model,
        device="cpu",
        additional_outputs={
            "extra::heat_flux": ModelOutput(
                quantity="heat_flux",
                unit="",
                explicit_gradients=[],
                per_atom=False,
            )
        },
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    assert "extra::heat_flux" in atoms.calc.additional_outputs
    results = atoms.calc.additional_outputs["extra::heat_flux"].block().values
    assert torch.allclose(
        results,
        torch.tensor(expected, dtype=results.dtype),
    )

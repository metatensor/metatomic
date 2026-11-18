from typing import Dict, List, Optional

import numpy as np
import pytest
import torch
import vesin.metatomic
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from metatensor.torch import Labels, TensorBlock, TensorMap

import metatomic_lj_test
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
    systems_to_torch,
    unit_conversion_factor,
)
from metatomic.torch.heat_flux import (
    HeatFlux,
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
def model_in_kcal_per_mol():
    return metatomic_lj_test.lennard_jones_model(
        atomic_type=18,
        cutoff=7.0,
        sigma=3.405,
        epsilon=0.2380,
        length_unit="Angstrom",
        energy_unit="kcal/mol",
        with_extension=False,
    )


@pytest.fixture
def system(request):
    if hasattr(request, "param") and request.param == "system_triclinic":
        cell = np.array([[6.0, 3.0, 1.0], [2.0, 6.0, 0.0], [0.0, 0.0, 6.0]])
        positions = np.array([[0.0, 0.0, 0.0]])
    else:
        cell = np.array([[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]])
        positions = np.array([[3.0, 3.0, 3.0]])
    atoms = Atoms("Ar", scaled_positions=positions, cell=cell, pbc=True).repeat(
        (2, 2, 2)
    )
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=300, rng=np.random.default_rng(42)
    )
    system = systems_to_torch(
        atoms,
        dtype=(torch.float64),
    )
    system.positions.requires_grad = True

    nl = vesin.metatomic.NeighborList(
        NeighborListOptions(cutoff=7.0, full_list=False, strict=True),
        length_unit="Angstrom",
        check_consistency=True,
    )

    nl.add_neighbor_list(system)

    # Add additional data for heat flux calculation
    n_atoms = len(atoms)
    masses = (
        torch.as_tensor(atoms.get_masses())
        .reshape(-1, 1)
        .to(device=system.device, dtype=system.positions.dtype)
    )
    velocities = (
        torch.as_tensor(atoms.get_velocities())
        .reshape(-1, 3, 1)
        .to(device=system.device, dtype=system.positions.dtype)
    )
    masses_block = TensorBlock(
        values=masses,
        samples=Labels(
            ["system", "atom"],
            torch.vstack([torch.full((n_atoms,), 0), torch.arange(n_atoms)]).T,
        ),
        components=[],
        properties=Labels(["mass"], torch.tensor([[0]])),
    )
    velocities_block = TensorBlock(
        values=velocities,
        samples=Labels(
            ["system", "atom"],
            torch.vstack([torch.full((n_atoms,), 0), torch.arange(n_atoms)]).T,
        ),
        components=[Labels(["xyz"], torch.arange(3).reshape(-1, 1))],
        properties=Labels(["velocity"], torch.tensor([[0]])),
    )
    masses_tensor = TensorMap(
        keys=Labels(["_"], torch.tensor([[0]])),
        blocks=[masses_block],
    )
    velocities_tensor = TensorMap(
        keys=Labels(["_"], torch.tensor([[0]])),
        blocks=[velocities_block],
    )

    masses_tensor.set_info("unit", "u")
    velocities_tensor.set_info("unit", "(eV/u)^(1/2)")
    system.add_data("mass", masses_tensor)
    system.add_data("velocity", velocities_tensor)
    return system


def test_heat_flux_wrapper_requested_inputs(model):
    wrapper = HeatFlux.wrap(model)
    requested = wrapper.requested_inputs(use_new_names=True)
    assert set(requested.keys()) == {"mass", "velocity"}


@pytest.mark.parametrize("script", [True, False])
@pytest.mark.parametrize(
    "system, variant, expected",
    [
        ("system", "/doubled", [[9.0147e-05], [-2.6166e-04], [-1.9002e-04]]),
        ("system", "", [[8.8238e-05], [-2.5559e-04], [-2.0570e-04]]),
        ("system_triclinic", "/doubled", [[1.0979e-04], [-2.7677e-04], [-1.7868e-04]]),
        ("system_triclinic", "", [[9.8061e-05], [-2.6314e-04], [-2.0004e-04]]),
    ],
    indirect=["system"],
)
def test_heat_flux(model, script, system, variant, expected):
    heat_flux_model = HeatFlux.wrap(model)

    evaluation_options = ModelEvaluationOptions(
        length_unit="Angstrom",
        outputs={
            "heat_flux" + variant: ModelOutput(unit="eV*A/fs", sample_kind="system")
        },
    )

    if script:
        heat_flux_model = torch.jit.script(heat_flux_model)

    results = heat_flux_model([system], evaluation_options, check_consistency=True)

    heat_flux = results["heat_flux" + variant].block().values
    assert torch.allclose(
        heat_flux,
        torch.tensor(expected, dtype=heat_flux.dtype),
    )


def test_multiple_outputs(model, system):
    heat_flux_model = HeatFlux.wrap(model)

    evaluation_options = ModelEvaluationOptions(
        length_unit="Angstrom",
        outputs={
            "heat_flux": ModelOutput(unit="eV*A/fs", sample_kind="system"),
            "heat_flux/doubled": ModelOutput(unit="eV*A/fs", sample_kind="system"),
        },
    )

    results = heat_flux_model([system], evaluation_options, check_consistency=True)

    heat_flux = results["heat_flux"].block().values
    expected = torch.tensor(
        [[8.8238e-05], [-2.5559e-04], [-2.0570e-04]], dtype=heat_flux.dtype
    )
    assert torch.allclose(heat_flux, expected)

    heat_flux = results["heat_flux/doubled"].block().values
    expected = torch.tensor(
        [[9.0147e-05], [-2.6166e-04], [-1.9002e-04]], dtype=heat_flux.dtype
    )
    assert torch.allclose(heat_flux, expected)


def test_input_energy_in_kcal_per_mol(model_in_kcal_per_mol, system):
    heat_flux_model = HeatFlux.wrap(model_in_kcal_per_mol)
    evaluation_options = ModelEvaluationOptions(
        length_unit="Angstrom",
        outputs={"heat_flux": ModelOutput(unit="eV*A/fs", sample_kind="system")},
    )
    results = heat_flux_model([system], evaluation_options, check_consistency=True)

    heat_flux = results["heat_flux"].block().values
    expected = torch.tensor(
        [[8.8238e-05], [-2.5559e-04], [-2.0570e-04]], dtype=heat_flux.dtype
    )
    assert torch.allclose(heat_flux, expected)


def test_output_unit_conversion(model, system):
    heat_flux_model = HeatFlux.wrap(model)
    evaluation_options = ModelEvaluationOptions(
        length_unit="Angstrom",
        outputs={"heat_flux": ModelOutput(unit="kcal/mol*A/ps", sample_kind="system")},
    )

    results = heat_flux_model([system], evaluation_options, check_consistency=True)
    heat_flux = results["heat_flux"].block().values

    expected = torch.tensor(
        [[8.8238e-05], [-2.5559e-04], [-2.0570e-04]], dtype=heat_flux.dtype
    )
    expected_converted = expected * unit_conversion_factor("eV*A/fs", "kcal/mol*A/ps")

    assert torch.allclose(heat_flux, expected_converted, rtol=1e-3)


class MessagePassingModel(torch.nn.Module):
    """
    A simple message passing model for testing the HeatFlux wrapper, especially for
    ensuring the consistency between the unfolded heat flux and the hop heat flux."""

    def __init__(
        self,
        rc: float,
        n_layers: int,
    ):
        super().__init__()
        self.rc = rc
        self.n_layers = n_layers

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [NeighborListOptions(self.rc, True, True)]

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        energy_blocks: List[TensorBlock] = []
        gather_blocks: List[TensorBlock] = []
        for i_system, system in enumerate(systems):
            positions = system.positions
            neighbor_list = system.get_neighbor_list(self.requested_neighbor_lists()[0])
            samples = neighbor_list.samples
            first = samples["first_atom"]
            second = samples["second_atom"]
            shifts = samples.values[:, 2:].to(positions.dtype)
            gather_samples = samples.insert(
                0, "system", torch.full((len(samples),), i_system, dtype=torch.int64)
            )

            positions_j = positions[second]
            tblock = TensorBlock(
                values=positions_j,
                samples=gather_samples,
                components=[],
                properties=Labels(
                    ["feature"], torch.tensor([[0], [1], [2]], dtype=torch.int64)
                ),
            )
            gather_blocks.append(tblock)
            distances = torch.linalg.norm(
                positions_j + shifts @ system.cell - positions[first], dim=1
            ).reshape(-1, 1)

            distances_vesin = torch.linalg.norm(
                neighbor_list.values.reshape(-1, 3), dim=1
            ).reshape(-1, 1)

            s = system.types.to(positions.dtype).reshape(-1, 1)
            for _ in range(self.n_layers):
                s_j = s[second]
                s = torch.nn.functional.silu(
                    s
                    + torch.zeros_like(s).index_add_(
                        0,
                        first,
                        s_j * (1 - distances / self.rc)
                        + (1 - distances_vesin / self.rc) ** 2,
                    )
                )
                tblock = TensorBlock(
                    values=s_j,
                    samples=gather_samples,
                    components=[],
                    properties=Labels(
                        "feature", torch.tensor([[0]], dtype=torch.int64)
                    ),
                )
                gather_blocks.append(tblock)

            energy_tblock = TensorBlock(
                values=s,
                samples=Labels(
                    ["system", "atom"],
                    torch.stack(
                        [
                            torch.full((len(s),), i_system, dtype=torch.int64),
                            torch.arange(len(s), dtype=torch.int64),
                        ],
                        dim=1,
                    ),
                ),
                components=[],
                properties=Labels("energy", torch.tensor([[0]], dtype=torch.int64)),
            )
            energy_blocks.append(energy_tblock)
        energy_tmap = TensorMap(
            Labels.range("system", len(energy_blocks)), energy_blocks
        )
        gather_tmap = TensorMap(
            Labels.range("gather", len(gather_blocks)), gather_blocks
        )

        return {"metatomic::gather": gather_tmap, "energy": energy_tmap}


def test_heat_flux_consistency(system):
    model = MessagePassingModel(rc=5.0, n_layers=2)
    model.train(False)
    capabilities = ModelCapabilities(
        length_unit="angstrom",
        interaction_range=model.rc * model.n_layers,
        supported_devices=["cpu"],
        dtype="float64",
        atomic_types=[1, 2, 3, 18],
        outputs={
            "test::nl": ModelOutput(),
            "metatomic::gather": ModelOutput(),
            "energy": ModelOutput(unit="eV"),
        },  #
    )
    atomistic = AtomisticModel(model, ModelMetadata(), capabilities)

    system.positions.requires_grad = True
    systems = [system]
    calculators = vesin.metatomic.neighbor_lists_for_model("angstrom", atomistic)
    for calculator in calculators:
        calculator.add_neighbor_list(systems)

    hf_calc = HeatFlux.wrap(atomistic)
    hop_results = hf_calc(
        systems,
        ModelEvaluationOptions(
            outputs={"heat_flux": ModelOutput(), "metatomic::gather": ModelOutput()}
        ),
        check_consistency=False,
    )
    unfolded_results = hf_calc(
        systems,
        ModelEvaluationOptions(outputs={"heat_flux": ModelOutput()}),
        check_consistency=True,
    )
    assert torch.allclose(
        hop_results["heat_flux"].block().values,
        unfolded_results["heat_flux"].block().values,
        rtol=1e-5,
        atol=1e-8,
    )

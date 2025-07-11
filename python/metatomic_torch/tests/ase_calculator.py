import glob
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional

import ase.build
import ase.calculators.lj
import ase.md
import ase.units
import metatomic_lj_test
import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)
from metatomic.torch.ase_calculator import (
    MetatomicCalculator,
    _compute_ase_neighbors,
    _full_3x3_to_voigt_6_stress,
)

from . import _tests_utils


STR_TO_DTYPE = {
    "float32": torch.float32,
    "float64": torch.float64,
}


CUTOFF = 5.0
SIGMA = 1.5808
EPSILON = 0.1729


@pytest.fixture
def model():
    return metatomic_lj_test.lennard_jones_model(
        atomic_type=28,
        cutoff=CUTOFF,
        sigma=SIGMA,
        epsilon=EPSILON,
        length_unit="Angstrom",
        energy_unit="eV",
        with_extension=False,
    )


@pytest.fixture
def model_different_units():
    return metatomic_lj_test.lennard_jones_model(
        atomic_type=28,
        cutoff=CUTOFF / ase.units.Bohr,
        sigma=SIGMA / ase.units.Bohr,
        epsilon=EPSILON / ase.units.kJ * ase.units.mol,
        length_unit="Bohr",
        energy_unit="kJ/mol",
        with_extension=False,
    )


@pytest.fixture
def atoms():
    np.random.seed(0xDEADBEEF)

    atoms = ase.build.make_supercell(
        ase.build.bulk("Ni", "fcc", a=3.6, cubic=True), 2 * np.eye(3)
    )
    atoms.positions += 0.2 * np.random.rand(*atoms.positions.shape)

    return atoms


def check_against_ase_lj(atoms, calculator):
    ref = atoms.copy()
    ref.calc = ase.calculators.lj.LennardJones(
        sigma=SIGMA, epsilon=EPSILON, rc=CUTOFF, ro=CUTOFF, smooth=False
    )

    atoms.calc = calculator

    assert np.allclose(ref.get_potential_energy(), atoms.get_potential_energy())
    assert np.allclose(ref.get_potential_energies(), atoms.get_potential_energies())
    assert np.allclose(ref.get_forces(), atoms.get_forces())
    assert np.allclose(ref.get_stress(), atoms.get_stress())


def test_python_model(model, model_different_units, atoms):
    check_against_ase_lj(atoms, MetatomicCalculator(model, check_consistency=True))
    check_against_ase_lj(
        atoms, MetatomicCalculator(model_different_units, check_consistency=True)
    )


def test_torch_script_model(model, model_different_units, atoms):
    model = torch.jit.script(model)
    check_against_ase_lj(atoms, MetatomicCalculator(model, check_consistency=True))

    model_different_units = torch.jit.script(model_different_units)
    check_against_ase_lj(
        atoms, MetatomicCalculator(model_different_units, check_consistency=True)
    )


def test_exported_model(tmpdir, model, model_different_units, atoms):
    path = os.path.join(tmpdir, "exported-model.pt")
    model.save(path)
    check_against_ase_lj(atoms, MetatomicCalculator(path, check_consistency=True))

    model_different_units.save(path)
    check_against_ase_lj(atoms, MetatomicCalculator(path, check_consistency=True))


@pytest.mark.parametrize("non_conservative", [True, False])
def test_get_properties(model, atoms, non_conservative):
    atoms.calc = MetatomicCalculator(
        model,
        check_consistency=True,
        non_conservative=non_conservative,
    )

    properties = atoms.get_properties(["energy", "energies", "forces", "stress"])

    assert np.all(properties["energies"] == atoms.get_potential_energies())
    assert np.all(properties["energy"] == atoms.get_potential_energy())
    assert np.all(properties["forces"] == atoms.get_forces())
    assert np.all(properties["stress"] == atoms.get_stress())

    # check that we can use all of the `.get_xxx` functions independantly
    atoms.calc = MetatomicCalculator(model, non_conservative=non_conservative)
    atoms.get_potential_energy()

    atoms.calc = MetatomicCalculator(model, non_conservative=non_conservative)
    atoms.get_potential_energies()

    atoms.calc = MetatomicCalculator(model, non_conservative=non_conservative)
    atoms.get_forces()

    atoms.calc = MetatomicCalculator(model, non_conservative=non_conservative)
    atoms.get_stress()


def test_run_model(tmpdir, model, atoms):
    ref = atoms.copy()
    ref.calc = ase.calculators.lj.LennardJones(
        sigma=SIGMA, epsilon=EPSILON, rc=CUTOFF, ro=CUTOFF, smooth=False
    )

    path = os.path.join(tmpdir, "exported-model.pt")
    model.save(path)
    calculator = MetatomicCalculator(path, check_consistency=True)

    first_mask = [a % 2 == 0 for a in range(len(atoms))]
    first_half = Labels(
        ["system", "atom"],
        torch.tensor([[0, a] for a in range(len(atoms)) if a % 2 == 0]),
    )

    second_mask = [a % 2 == 1 for a in range(len(atoms))]
    second_half = Labels(
        ["system", "atom"],
        torch.tensor([[0, a] for a in range(len(atoms)) if a % 2 == 1]),
    )

    # check overall prediction
    requested = {"energy": ModelOutput(per_atom=False)}
    outputs = calculator.run_model(atoms, outputs=requested)
    assert np.allclose(ref.get_potential_energy(), outputs["energy"].block().values)

    # check per atom energy
    requested = {"energy": ModelOutput(per_atom=True)}
    outputs = calculator.run_model(atoms, outputs=requested, selected_atoms=first_half)
    first_energies = outputs["energy"].block().values.numpy().reshape(-1)

    outputs = calculator.run_model(atoms, outputs=requested, selected_atoms=second_half)
    second_energies = outputs["energy"].block().values.numpy().reshape(-1)

    expected = ref.get_potential_energies()
    assert np.allclose(expected[first_mask], first_energies)
    assert np.allclose(expected[second_mask], second_energies)

    # check total energy
    requested = {"energy": ModelOutput(per_atom=False)}
    outputs = calculator.run_model(atoms, outputs=requested, selected_atoms=first_half)
    first_energies = outputs["energy"].block().values.numpy().reshape(-1)

    outputs = calculator.run_model(atoms, outputs=requested, selected_atoms=second_half)
    second_energies = outputs["energy"].block().values.numpy().reshape(-1)

    expected = ref.get_potential_energy()
    assert np.allclose(expected, first_energies + second_energies)

    # check batched prediction
    requested = {"energy": ModelOutput(per_atom=False)}
    outputs = calculator.run_model([atoms, atoms], outputs=requested)
    assert np.allclose(
        ref.get_potential_energy(), outputs["energy"].block().values[[0]]
    )

    # check non-conservative forces and stresses
    requested = {
        "energy": ModelOutput(per_atom=False),
        "non_conservative_forces": ModelOutput(per_atom=True),
        "non_conservative_stress": ModelOutput(per_atom=False),
    }
    outputs = calculator.run_model([atoms, atoms], outputs=requested)
    assert np.allclose(
        ref.get_potential_energy(), outputs["energy"].block().values[[0]]
    )
    assert np.allclose(
        ref.get_potential_energy(), outputs["energy"].block().values[[1]]
    )
    assert "non_conservative_forces" in outputs
    assert outputs["non_conservative_forces"].block().values.shape == (
        2 * len(atoms),
        3,
        1,
    )
    assert "non_conservative_stress" in outputs
    assert outputs["non_conservative_stress"].block().values.shape == (2, 3, 3, 1)


@pytest.mark.parametrize("non_conservative", [True, False])
def test_compute_energy(tmpdir, model, atoms, non_conservative):
    ref = atoms.copy()
    ref.calc = ase.calculators.lj.LennardJones(
        sigma=SIGMA, epsilon=EPSILON, rc=CUTOFF, ro=CUTOFF, smooth=False
    )

    path = os.path.join(tmpdir, "exported-model.pt")
    model.save(path)
    calculator = MetatomicCalculator(
        path,
        check_consistency=True,
        non_conservative=non_conservative,
    )

    energy = calculator.compute_energy(atoms)["energy"]
    assert np.allclose(ref.get_potential_energy(), energy)

    results = calculator.compute_energy(atoms, compute_forces_and_stresses=True)
    assert np.allclose(ref.get_potential_energy(), results["energy"])
    if not non_conservative:
        assert np.allclose(ref.get_forces(), results["forces"])
        assert np.allclose(
            ref.get_stress(), _full_3x3_to_voigt_6_stress(results["stress"])
        )

    energies = calculator.compute_energy([atoms, atoms])["energy"]
    assert np.allclose(ref.get_potential_energy(), energies[0])
    assert np.allclose(ref.get_potential_energy(), energies[1])

    results = calculator.compute_energy(
        [atoms, atoms], compute_forces_and_stresses=True
    )
    assert np.allclose(ref.get_potential_energy(), results["energy"][0])
    assert np.allclose(ref.get_potential_energy(), results["energy"][1])
    if not non_conservative:
        assert np.allclose(ref.get_forces(), results["forces"][0])
        assert np.allclose(ref.get_forces(), results["forces"][1])
        assert np.allclose(
            ref.get_stress(), _full_3x3_to_voigt_6_stress(results["stress"][0])
        )
        assert np.allclose(
            ref.get_stress(), _full_3x3_to_voigt_6_stress(results["stress"][1])
        )

    atoms_no_pbc = atoms.copy()
    atoms_no_pbc.pbc = [False, False, False]
    assert "stress" not in calculator.compute_energy(atoms_no_pbc)


def test_serialize_ase(tmpdir, model, atoms):
    # Run some tests with a different dtype
    model._capabilities.dtype = "float32"

    calculator = MetatomicCalculator(model)

    message = (
        "can not save metatensor model in ASE `todict`, please initialize "
        "`MetatomicCalculator` with a path to a saved model file if you need to use "
        "`todict"
    )
    with pytest.raises(RuntimeError, match=message):
        calculator.todict()

    # save with exported model
    path = os.path.join(tmpdir, "exported-model.pt")
    model.save(path)

    calculator = MetatomicCalculator(path)
    data = calculator.todict()
    _ = MetatomicCalculator.fromdict(data)

    # check the standard trajectory format of ASE, which uses `todict`/`fromdict`
    atoms.calc = MetatomicCalculator(path)
    with tmpdir.as_cwd():
        dyn = ase.md.VelocityVerlet(
            atoms,
            timestep=2 * ase.units.fs,
            trajectory="file.traj",
        )
        dyn.run(10)
        dyn.close()

        atoms = ase.io.read("file.traj", "-1")
        assert atoms.calc is not None


def test_dtype_device(tmpdir, model, atoms):
    ref = atoms.copy()
    ref.calc = ase.calculators.lj.LennardJones(
        sigma=SIGMA, epsilon=EPSILON, rc=CUTOFF, ro=CUTOFF, smooth=False
    )
    expected = ref.get_potential_energy()
    path = os.path.join(tmpdir, "exported-model.pt")

    dtype_device = [
        ("float64", "cpu"),
        ("float32", "cpu"),
    ]

    if _tests_utils.can_use_mps_backend():
        dtype_device.append(("float32", "mps"))

    if torch.cuda.is_available():
        dtype_device.append(("float32", "cuda"))
        dtype_device.append(("float64", "cuda"))

    for dtype, device in dtype_device:
        capabilities = model.capabilities()
        capabilities.dtype = dtype

        # re-create the model with a different dtype
        dtype_model = AtomisticModel(
            model.module.to(STR_TO_DTYPE[dtype]),
            model.metadata(),
            capabilities,
        )

        dtype_model.save(path)
        atoms.calc = MetatomicCalculator(path, check_consistency=True, device=device)
        assert np.allclose(atoms.get_potential_energy(), expected)


def test_model_with_extensions(tmpdir, atoms):
    ref = atoms.copy()
    ref.calc = ase.calculators.lj.LennardJones(
        sigma=SIGMA, epsilon=EPSILON, rc=CUTOFF, ro=CUTOFF, smooth=False
    )

    model_path = os.path.join(tmpdir, "model.pt")
    extensions_directory = os.path.join(tmpdir, "extensions")

    # use forward slash as path separator even on windows. This removes the need to
    # escape the path below
    model_path = model_path.replace("\\", "/")
    extensions_directory = extensions_directory.replace("\\", "/")

    # export the model in a sub-process, to prevent loading of the extension in the
    # current interpreter (until we try to execute the code)
    script = f"""
import metatomic_lj_test

model = metatomic_lj_test.lennard_jones_model(
    atomic_type=28,
    cutoff={CUTOFF},
    sigma={SIGMA},
    epsilon={EPSILON},
    length_unit="Angstrom",
    energy_unit="eV",
    with_extension=True,
)

model.save("{model_path}", collect_extensions="{extensions_directory}")
    """

    subprocess.run([sys.executable, "-c", script], check=True, cwd=tmpdir)

    message = "Unknown builtin op: metatomic_lj_test::lennard_jones"
    with pytest.raises(RuntimeError, match=message):
        MetatomicCalculator(model_path, check_consistency=True)

    # Now actually loading the extensions
    atoms.calc = MetatomicCalculator(
        model_path,
        extensions_directory=extensions_directory,
        check_consistency=True,
    )

    assert np.allclose(ref.get_potential_energy(), atoms.get_potential_energy())


def _read_neighbor_check(path):
    with open(path) as fd:
        data = json.load(fd)

    dtype = torch.float64

    positions = torch.tensor(data["system"]["positions"], dtype=dtype).reshape(-1, 3)
    system = System(
        types=torch.tensor([1] * positions.shape[0], dtype=torch.int32),
        positions=positions,
        cell=torch.tensor(data["system"]["cell"], dtype=dtype),
        pbc=torch.tensor([True, True, True]),
    )

    options = NeighborListOptions(
        cutoff=data["options"]["cutoff"],
        full_list=data["options"]["full_list"],
        # ASE can only compute strict NL
        strict=True,
    )

    samples = torch.tensor(
        data["expected-neighbors"]["samples"], dtype=torch.int32
    ).reshape(-1, 5)
    distances = torch.tensor(
        data["expected-neighbors"]["distances"], dtype=dtype
    ).reshape(-1, 3, 1)

    neighbors = TensorBlock(
        values=distances,
        samples=Labels(
            [
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            samples,
        ),
        components=[Labels.range("xyz", 3)],
        properties=Labels.range("distance", 1),
    )

    return system, options, neighbors


def _check_same_set_of_neighbors(expected, actual, full_list):
    assert expected.samples.names == actual.samples.names
    assert len(expected.samples) == len(actual.samples)

    for sample_i, sample in enumerate(expected.samples):
        sign = 1.0
        position = actual.samples.position(sample)

        if position is None and not full_list:
            # try looking for the inverse pair
            sign = -1.0
            position = actual.samples.position(
                [sample[1], sample[0], -sample[2], -sample[3], -sample[4]]
            )

        if position is None:
            raise AssertionError(f"missing expected neighbors sample: {sample}")

        assert torch.allclose(expected.values[sample_i], sign * actual.values[position])


def test_neighbor_list_adapter():
    HERE = os.path.realpath(os.path.dirname(__file__))
    test_files = os.path.join(
        HERE, "..", "..", "..", "..", "metatensor-torch", "tests", "neighbor-checks"
    )

    for path in glob.glob(os.path.join(test_files, "*.json")):
        system, options, expected_neighbors = _read_neighbor_check(path)

        atoms = ase.Atoms(
            symbols=system.types.numpy(),
            positions=system.positions.numpy(),
            cell=system.cell.numpy(),
            pbc=not torch.all(system.cell == torch.zeros((3, 3))),
        )

        neighbors = _compute_ase_neighbors(
            atoms, options, torch.float64, torch.device("cpu")
        )

        _check_same_set_of_neighbors(expected_neighbors, neighbors, options.full_list)


class MultipleOutputModel(torch.nn.Module):
    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        results = {}
        for name, requested in outputs.items():
            assert not requested.per_atom

            block = TensorBlock(
                values=torch.tensor([[0.0]], dtype=torch.float64),
                samples=Labels("system", torch.tensor([[0]])),
                components=torch.jit.annotate(List[Labels], []),
                properties=Labels(name.split(":")[0], torch.tensor([[0]])),
            )
            tensor = TensorMap(Labels("_", torch.tensor([[0]])), [block])
            results[name] = tensor

        return results


def test_additional_outputs(atoms):
    capabilities = ModelCapabilities(
        outputs={
            "energy": ModelOutput(per_atom=False),
            "test::test": ModelOutput(per_atom=False),
            "another::one": ModelOutput(per_atom=False),
        },
        atomic_types=[28],
        interaction_range=0.0,
        supported_devices=["cpu"],
        dtype="float64",
    )
    model = AtomisticModel(MultipleOutputModel().eval(), ModelMetadata(), capabilities)

    atoms.calc = MetatomicCalculator(model, check_consistency=True)

    assert atoms.get_potential_energy() == 0.0
    assert atoms.calc.additional_outputs == {}

    atoms.calc = MetatomicCalculator(
        model,
        check_consistency=True,
        additional_outputs={
            "test::test": ModelOutput(per_atom=False),
            "another::one": ModelOutput(per_atom=False),
        },
    )
    assert atoms.get_potential_energy() == 0.0

    test = atoms.calc.additional_outputs["test::test"]
    assert test.block().properties.names == ["test"]

    another = atoms.calc.additional_outputs["another::one"]
    assert another.block().properties.names == ["another"]

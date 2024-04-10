import os
import subprocess
import sys

import ase.build
import ase.calculators.lj
import ase.md
import ase.units
import metatensor_lj_test
import numpy as np
import pytest
import torch

from metatensor.torch import Labels
from metatensor.torch.atomistic import MetatensorAtomisticModel, ModelOutput
from metatensor.torch.atomistic.ase_calculator import MetatensorCalculator


STR_TO_DTYPE = {
    "float32": torch.float32,
    "float64": torch.float64,
}


CUTOFF = 5.0
SIGMA = 1.5808
EPSILON = 0.1729


@pytest.fixture
def model():
    return metatensor_lj_test.lennard_jones_model(
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
    return metatensor_lj_test.lennard_jones_model(
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
    check_against_ase_lj(atoms, MetatensorCalculator(model, check_consistency=True))
    check_against_ase_lj(
        atoms, MetatensorCalculator(model_different_units, check_consistency=True)
    )


def test_torch_script_model(model, model_different_units, atoms):
    model = torch.jit.script(model)
    check_against_ase_lj(atoms, MetatensorCalculator(model, check_consistency=True))

    model_different_units = torch.jit.script(model_different_units)
    check_against_ase_lj(
        atoms, MetatensorCalculator(model_different_units, check_consistency=True)
    )


def test_exported_model(tmpdir, model, model_different_units, atoms):
    path = os.path.join(tmpdir, "exported-model.pt")
    model.export(path)
    check_against_ase_lj(atoms, MetatensorCalculator(path, check_consistency=True))

    model_different_units.export(path)
    check_against_ase_lj(atoms, MetatensorCalculator(path, check_consistency=True))


def test_get_properties(model, atoms):
    atoms.calc = MetatensorCalculator(model, check_consistency=True)

    properties = atoms.get_properties(["energy", "energies", "forces", "stress"])

    assert np.all(properties["energies"] == atoms.get_potential_energies())
    assert np.all(properties["energy"] == atoms.get_potential_energy())
    assert np.all(properties["forces"] == atoms.get_forces())
    assert np.all(properties["stress"] == atoms.get_stress())


def test_selected_atoms(tmpdir, model, atoms):
    ref = atoms.copy()
    ref.calc = ase.calculators.lj.LennardJones(
        sigma=SIGMA, epsilon=EPSILON, rc=CUTOFF, ro=CUTOFF, smooth=False
    )

    path = os.path.join(tmpdir, "exported-model.pt")
    model.export(path)
    calculator = MetatensorCalculator(path, check_consistency=True)

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


def test_serialize_ase(tmpdir, model, atoms):
    # Run some tests with a different dtype
    model._capabilities.dtype = "float32"

    calculator = MetatensorCalculator(model)

    message = (
        "can not save metatensor model in ASE `todict`, please initialize "
        "`MetatensorCalculator` with a path to a saved model file if you need to use "
        "`todict"
    )
    with pytest.raises(RuntimeError, match=message):
        calculator.todict()

    # save with exported model
    path = os.path.join(tmpdir, "exported-model.pt")
    model.export(path)

    calculator = MetatensorCalculator(path)
    data = calculator.todict()
    _ = MetatensorCalculator.fromdict(data)

    # check the standard trajectory format of ASE, which uses `todict`/`fromdict`
    atoms.calc = MetatensorCalculator(path)
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

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    ):
        dtype_device.append(("float32", "mps"))

    if torch.cuda.is_available():
        dtype_device.append(("float32", "cuda"))
        dtype_device.append(("float64", "cuda"))

    for dtype, device in dtype_device:
        capabilities = model.capabilities()
        capabilities.dtype = dtype

        # re-create the model with a different dtype
        dtype_model = MetatensorAtomisticModel(
            model._module.to(STR_TO_DTYPE[dtype]),
            model.metadata(),
            capabilities,
        )

        dtype_model.export(path)
        atoms.calc = MetatensorCalculator(path, check_consistency=True, device=device)
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
import metatensor_lj_test

model = metatensor_lj_test.lennard_jones_model(
    atomic_type=28,
    cutoff={CUTOFF},
    sigma={SIGMA},
    epsilon={EPSILON},
    length_unit="Angstrom",
    energy_unit="eV",
    with_extension=True,
)

model.export("{model_path}", collect_extensions="{extensions_directory}")
    """

    subprocess.run([sys.executable, "-c", script], check=True)

    message = "Unknown builtin op: metatensor_lj_test::lennard_jones"
    with pytest.raises(RuntimeError, match=message):
        MetatensorCalculator(model_path, check_consistency=True)

    # Now actually loading the extensions
    atoms.calc = MetatensorCalculator(
        model_path,
        extensions_directory=extensions_directory,
        check_consistency=True,
    )

    assert np.allclose(ref.get_potential_energy(), atoms.get_potential_energy())

import os
import subprocess
import sys
from typing import Dict, List, Optional

import ase.build
import ase.calculators.lj
import ase.md
import ase.units
import numpy as np
import pytest
import torch
from ase.calculators.calculator import PropertyNotImplementedError
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from metatensor.torch import Labels, TensorBlock, TensorMap

import metatomic_lj_test
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    System,
)
from metatomic_ase import MetatomicCalculator
from metatomic_ase._calculator import (
    PER_ATOM_QUANTITIES,
    _full_3x3_to_voigt_6_stress,
)

from ._tests_utils import ALL_DEVICE_DTYPE, STR_TO_DTYPE, prints_to_stderr


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


def check_against_ase_lj(atoms, calculator, dtype):
    ref = atoms.copy()
    ref.calc = ase.calculators.lj.LennardJones(
        sigma=SIGMA, epsilon=EPSILON, rc=CUTOFF, ro=CUTOFF, smooth=False
    )

    atoms.calc = calculator

    if dtype == "float32":
        rtol = 1e-5
        atol = 1e-5
    elif dtype == "float64":
        rtol = 1e-5
        atol = 1e-8
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    assert np.allclose(
        ref.get_potential_energy(), atoms.get_potential_energy(), atol=atol, rtol=rtol
    )
    assert np.allclose(
        ref.get_potential_energies(),
        atoms.get_potential_energies(),
        atol=atol,
        rtol=rtol,
    )
    assert np.allclose(ref.get_forces(), atoms.get_forces(), atol=atol, rtol=rtol)
    assert np.allclose(ref.get_stress(), atoms.get_stress(), atol=atol, rtol=rtol)


def _set_model_dtype(model, dtype):
    model._capabilities.dtype = dtype
    model._model_dtype = STR_TO_DTYPE[dtype]
    return model


@pytest.mark.parametrize("device,dtype", ALL_DEVICE_DTYPE)
def test_python_model(model, model_different_units, atoms, device, dtype):
    model = _set_model_dtype(model, dtype)
    model_different_units = _set_model_dtype(model_different_units, dtype)

    check_against_ase_lj(
        atoms,
        MetatomicCalculator(
            model,
            check_consistency=True,
            uncertainty_threshold=None,
            device=device,
        ),
        dtype=dtype,
    )
    check_against_ase_lj(
        atoms,
        MetatomicCalculator(
            model_different_units,
            check_consistency=True,
            uncertainty_threshold=None,
            device=device,
        ),
        dtype=dtype,
    )


@pytest.mark.parametrize("device,dtype", ALL_DEVICE_DTYPE)
def test_torch_script_model(model, model_different_units, atoms, device, dtype):
    model = _set_model_dtype(model, dtype)
    model_different_units = _set_model_dtype(model_different_units, dtype)

    model = torch.jit.script(model)
    check_against_ase_lj(
        atoms,
        MetatomicCalculator(
            model,
            check_consistency=True,
            uncertainty_threshold=None,
            device=device,
        ),
        dtype=dtype,
    )

    model_different_units = torch.jit.script(model_different_units)
    check_against_ase_lj(
        atoms,
        MetatomicCalculator(
            model_different_units,
            check_consistency=True,
            uncertainty_threshold=None,
            device=device,
        ),
        dtype=dtype,
    )


@pytest.mark.parametrize("device,dtype", ALL_DEVICE_DTYPE)
def test_exported_model(tmpdir, model, model_different_units, atoms, device, dtype):
    model = _set_model_dtype(model, dtype)
    model_different_units = _set_model_dtype(model_different_units, dtype)

    path = os.path.join(tmpdir, "exported-model.pt")
    model.save(path)
    check_against_ase_lj(
        atoms,
        MetatomicCalculator(
            path,
            check_consistency=True,
            uncertainty_threshold=None,
            device=device,
        ),
        dtype=dtype,
    )

    model_different_units.save(path)
    check_against_ase_lj(
        atoms,
        MetatomicCalculator(
            path,
            check_consistency=True,
            uncertainty_threshold=None,
            device=device,
        ),
        dtype=dtype,
    )


@pytest.mark.parametrize("non_conservative", [True, False])
def test_get_properties(model, atoms, non_conservative):
    atoms.calc = MetatomicCalculator(
        model,
        check_consistency=True,
        non_conservative=non_conservative,
        uncertainty_threshold=None,
    )

    properties = atoms.get_properties(["energy", "energies", "forces", "stress"])

    assert np.all(properties["energies"] == atoms.get_potential_energies())
    assert np.all(properties["energy"] == atoms.get_potential_energy())
    assert np.all(properties["forces"] == atoms.get_forces())
    assert np.all(properties["stress"] == atoms.get_stress())

    # check that we can use all of the `.get_xxx` functions independantly
    atoms.calc = MetatomicCalculator(
        model, non_conservative=non_conservative, uncertainty_threshold=None
    )
    atoms.get_potential_energy()

    atoms.calc = MetatomicCalculator(
        model, non_conservative=non_conservative, uncertainty_threshold=None
    )
    atoms.get_potential_energies()

    atoms.calc = MetatomicCalculator(
        model, non_conservative=non_conservative, uncertainty_threshold=None
    )
    atoms.get_forces()

    atoms.calc = MetatomicCalculator(
        model, non_conservative=non_conservative, uncertainty_threshold=None
    )
    atoms.get_stress()


def test_accuracy_warning(model, atoms):
    # our dummy model artificially gives a high uncertainty for large structures
    big_atoms = atoms * (2, 2, 2)
    big_atoms.calc = MetatomicCalculator(model, check_consistency=True)

    with pytest.warns(
        UserWarning,
        match="Some of the atomic energy uncertainties are large",
    ):
        big_atoms.get_forces()


def accuracy_is_zero_error(atoms):
    match = "`uncertainty_threshold` is 0.0 but must be positive"
    with pytest.raises(ValueError, match=match):
        atoms.calc = MetatomicCalculator(model, uncertainty_threshold=0.0)


def test_run_model(tmpdir, model, atoms):
    ref = atoms.copy()
    ref.calc = ase.calculators.lj.LennardJones(
        sigma=SIGMA, epsilon=EPSILON, rc=CUTOFF, ro=CUTOFF, smooth=False
    )

    path = os.path.join(tmpdir, "exported-model.pt")
    model.save(path)
    calculator = MetatomicCalculator(
        path, check_consistency=True, uncertainty_threshold=None
    )

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
    requested = {"energy": ModelOutput(sample_kind="system")}
    outputs = calculator.run_model(atoms, outputs=requested)
    assert np.allclose(
        ref.get_potential_energy(), outputs["energy"].block().values.item()
    )

    # check per atom energy
    requested = {"energy": ModelOutput(sample_kind="atom")}
    outputs = calculator.run_model(atoms, outputs=requested, selected_atoms=first_half)
    first_energies = outputs["energy"].block().values.numpy().reshape(-1)

    outputs = calculator.run_model(atoms, outputs=requested, selected_atoms=second_half)
    second_energies = outputs["energy"].block().values.numpy().reshape(-1)

    expected = ref.get_potential_energies()
    assert np.allclose(expected[first_mask], first_energies)
    assert np.allclose(expected[second_mask], second_energies)

    # check total energy
    requested = {"energy": ModelOutput(sample_kind="system")}
    outputs = calculator.run_model(atoms, outputs=requested, selected_atoms=first_half)
    first_energies = outputs["energy"].block().values.numpy().reshape(-1)

    outputs = calculator.run_model(atoms, outputs=requested, selected_atoms=second_half)
    second_energies = outputs["energy"].block().values.numpy().reshape(-1)

    expected = ref.get_potential_energy()
    assert np.allclose(expected, first_energies + second_energies)

    # check batched prediction
    requested = {"energy": ModelOutput(sample_kind="system")}
    outputs = calculator.run_model([atoms, atoms], outputs=requested)
    assert np.allclose(
        ref.get_potential_energy(), outputs["energy"].block().values[[0]]
    )

    # check non-conservative forces and stresses
    requested = {
        "energy": ModelOutput(sample_kind="system"),
        "non_conservative_force": ModelOutput(sample_kind="atom"),
        "non_conservative_stress": ModelOutput(sample_kind="system"),
    }
    outputs = calculator.run_model([atoms, atoms], outputs=requested)
    assert np.allclose(
        ref.get_potential_energy(), outputs["energy"].block().values[[0]]
    )
    assert np.allclose(
        ref.get_potential_energy(), outputs["energy"].block().values[[1]]
    )
    assert "non_conservative_force" in outputs

    shape = (2 * len(atoms), 3, 1)
    assert outputs["non_conservative_force"].block().values.shape == shape
    assert "non_conservative_stress" in outputs
    assert outputs["non_conservative_stress"].block().values.shape == (2, 3, 3, 1)


@pytest.mark.parametrize("non_conservative", [True, False])
@pytest.mark.parametrize("per_atom", [True, False])
def test_compute_energy(tmpdir, model, atoms, non_conservative, per_atom):
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

    results = calculator.compute_energy(atoms, per_atom=per_atom)
    if per_atom:
        energies = results["energies"]
        assert np.allclose(ref.get_potential_energies(), energies)
    assert np.allclose(ref.get_potential_energy(), results["energy"])

    results = calculator.compute_energy(
        atoms, compute_forces_and_stresses=True, per_atom=per_atom
    )
    assert np.allclose(ref.get_potential_energy(), results["energy"])
    if not non_conservative:
        assert np.allclose(ref.get_forces(), results["forces"])
        assert np.allclose(
            ref.get_stress(), _full_3x3_to_voigt_6_stress(results["stress"])
        )
    if per_atom:
        assert np.allclose(ref.get_potential_energies(), results["energies"])

    results = calculator.compute_energy([atoms, atoms], per_atom=per_atom)
    assert np.allclose(ref.get_potential_energy(), results["energy"][0])
    assert np.allclose(ref.get_potential_energy(), results["energy"][1])
    if per_atom:
        assert np.allclose(ref.get_potential_energies(), results["energies"][0])
        assert np.allclose(ref.get_potential_energies(), results["energies"][1])

    results = calculator.compute_energy(
        [atoms, atoms],
        compute_forces_and_stresses=True,
        per_atom=per_atom,
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
    if per_atom:
        assert np.allclose(ref.get_potential_energies(), results["energies"][0])
        assert np.allclose(ref.get_potential_energies(), results["energies"][1])

    atoms_no_pbc = atoms.copy()
    atoms_no_pbc.pbc = [False, False, False]
    assert "stress" not in calculator.compute_energy(atoms_no_pbc)


def test_serialize_ase(tmpdir, model, atoms):
    calculator = MetatomicCalculator(model, uncertainty_threshold=None)

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

    calculator = MetatomicCalculator(path, uncertainty_threshold=None)
    data = calculator.todict()
    _ = MetatomicCalculator.fromdict(data)

    # check the standard trajectory format of ASE, which uses `todict`/`fromdict`
    atoms.calc = MetatomicCalculator(path, uncertainty_threshold=None)
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


@pytest.mark.parametrize("device,dtype", ALL_DEVICE_DTYPE)
def test_dtype_device(tmpdir, model, atoms, device, dtype):
    ref = atoms.copy()
    ref.calc = ase.calculators.lj.LennardJones(
        sigma=SIGMA, epsilon=EPSILON, rc=CUTOFF, ro=CUTOFF, smooth=False
    )
    expected = ref.get_potential_energy()
    path = os.path.join(tmpdir, "exported-model.pt")

    capabilities = model.capabilities()
    capabilities.dtype = dtype
    # only keep the intial outputs, this is a workaround for the compatibility code in
    # AtomisticModel which adds deprecated duplicate outputs to the capabilities
    capabilities.outputs = {
        name: capabilities.outputs[name]
        for name in model._model_capabilities_outputs_names
    }

    # re-create the model with a different dtype
    dtype_model = AtomisticModel(
        model.module.to(STR_TO_DTYPE[dtype]),
        model.metadata(),
        capabilities,
    )

    dtype_model.save(path)
    atoms.calc = MetatomicCalculator(
        path, check_consistency=True, device=device, uncertainty_threshold=None
    )
    assert np.allclose(atoms.get_potential_energy(), expected)


def test_model_with_extensions(tmpdir, atoms, capfd):
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

    message = (
        "This is likely due to missing TorchScript extensions.\nMake sure to provide "
        "the `extensions_directory` argument if your extensions are not installed "
        "system-wide"
    )
    with pytest.raises(RuntimeError, match=message):
        printed_err = "Warning: failed to load TorchScript extension metatomic_lj_test"
        with prints_to_stderr(capfd, match=printed_err):
            MetatomicCalculator(model_path, check_consistency=True)

    # Now actually loading the extensions
    atoms.calc = MetatomicCalculator(
        model_path,
        extensions_directory=extensions_directory,
        check_consistency=True,
        uncertainty_threshold=None,
    )

    assert np.allclose(ref.get_potential_energy(), atoms.get_potential_energy())


class MultipleOutputModel(torch.nn.Module):
    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        results = {}
        device = systems[0].positions.device
        for name, requested in outputs.items():
            assert requested.sample_kind == "system"

            block = TensorBlock(
                values=torch.tensor([[0.0]], dtype=torch.float64, device=device),
                samples=Labels("system", torch.tensor([[0]], device=device)),
                components=torch.jit.annotate(List[Labels], []),
                properties=Labels(
                    name.split(":")[0], torch.tensor([[0]], device=device)
                ),
            )
            tensor = TensorMap(Labels("_", torch.tensor([[0]], device=device)), [block])
            results[name] = tensor

        return results


def test_additional_outputs(atoms):
    capabilities = ModelCapabilities(
        outputs={
            "energy": ModelOutput(sample_kind="system", unit="eV"),
            "test::test": ModelOutput(sample_kind="system"),
            "another::one": ModelOutput(sample_kind="system"),
        },
        atomic_types=[28],
        interaction_range=0.0,
        supported_devices=["cpu"],
        dtype="float64",
    )
    model = AtomisticModel(MultipleOutputModel().eval(), ModelMetadata(), capabilities)

    atoms.calc = MetatomicCalculator(
        model,
        check_consistency=True,
        uncertainty_threshold=None,
    )

    assert atoms.get_potential_energy() == 0.0
    assert atoms.calc.additional_outputs == {}

    atoms.calc = MetatomicCalculator(
        model,
        additional_outputs={
            "test::test": ModelOutput(sample_kind="system"),
            "another::one": ModelOutput(sample_kind="system"),
        },
        check_consistency=True,
        uncertainty_threshold=None,
    )
    assert atoms.get_potential_energy() == 0.0

    test = atoms.calc.additional_outputs["test::test"]
    assert test.block().properties.names == ["test"]

    another = atoms.calc.additional_outputs["another::one"]
    assert another.block().properties.names == ["another"]


@pytest.mark.parametrize("non_conservative", [True, False])
def test_variants(atoms, model, non_conservative):
    atoms.calc = MetatomicCalculator(
        model,
        check_consistency=True,
        non_conservative=non_conservative,
        uncertainty_threshold=None,
    )

    atoms_variant = atoms.copy()
    atoms_variant.calc = MetatomicCalculator(
        model,
        check_consistency=True,
        non_conservative=non_conservative,
        variants={"energy": "doubled"},
        uncertainty_threshold=None,
    )

    np.allclose(
        2.0 * atoms.get_potential_energy(), atoms_variant.get_potential_energy()
    )
    np.allclose(2.0 * atoms.get_forces(), atoms_variant.get_forces())
    np.allclose(2.0 * atoms.get_stress(), atoms_variant.get_stress())


@pytest.mark.parametrize(
    "default_output",
    [
        "energy",
        "energy_uncertainty",
        "non_conservative_force",
        "non_conservative_stress",
    ],
)
def test_variant_default(atoms, model, default_output):
    """Allow setting a variant explicitly to None to use the default output."""

    atoms.calc = MetatomicCalculator(
        model,
        check_consistency=True,
        non_conservative=True,
        uncertainty_threshold=None,
    )

    variants = {
        v: "doubled"
        for v in [
            "energy",
            "energy_uncertainty",
            "non_conservative_force",
            "non_conservative_stress",
        ]
    }
    variants[default_output] = None

    atoms_variant = atoms.copy()
    atoms_variant.calc = MetatomicCalculator(
        model,
        check_consistency=True,
        non_conservative=True,
        variants={"energy": "doubled"},
        uncertainty_threshold=None,
    )

    if default_output == "energy":
        np.allclose(atoms.get_potential_energy(), atoms_variant.get_potential_energy())
        np.allclose(2.0 * atoms.get_forces(), atoms_variant.get_forces())
        np.allclose(2.0 * atoms.get_stress(), atoms_variant.get_stress())
    elif default_output == "non_conservative_force":
        np.allclose(
            2.0 * atoms.get_potential_energy(), atoms_variant.get_potential_energy()
        )
        np.allclose(atoms.get_forces(), atoms_variant.get_forces())
        np.allclose(2.0 * atoms.get_stress(), atoms_variant.get_stress())
    elif default_output == "non_conservative_stress":
        np.allclose(
            2.0 * atoms.get_potential_energy(), atoms_variant.get_potential_energy()
        )
        np.allclose(2.0 * atoms.get_forces(), atoms_variant.get_forces())
        np.allclose(atoms.get_stress(), atoms_variant.get_stress())


@pytest.mark.parametrize("force_is_None", [True, False])
def test_variant_non_conservative_error(atoms, model, force_is_None):
    variants = {
        "energy": "doubled",
        "non_conservative_force": "doubled",
        "non_conservative_stress": "doubled",
    }

    if force_is_None:
        variants["non_conservative_force"] = None
    else:
        variants["non_conservative_stress"] = None

    match = "must either be both `None` or both not `None`."
    with pytest.raises(ValueError, match=match):
        MetatomicCalculator(
            model,
            check_consistency=True,
            non_conservative=True,
            variants=variants,
            uncertainty_threshold=None,
        )


@pytest.mark.parametrize("non_conservative", ["on", "off", "invalid"])
def test_non_conservative_invalid_raises(model, non_conservative):
    """Passing an invalid non_conservative value raises ValueError."""
    with pytest.raises(ValueError, match="non_conservative must be one of"):
        MetatomicCalculator(model, non_conservative=non_conservative)


@pytest.mark.parametrize("non_conservative", ["forces", "stress"])
def test_non_conservative_mixed_modes(tmpdir, model, atoms, non_conservative):
    """'forces' mode uses NC forces + autograd stress; 'stress' mode is the reverse"""
    ref = atoms.copy()
    ref.calc = ase.calculators.lj.LennardJones(
        sigma=SIGMA, epsilon=EPSILON, rc=CUTOFF, ro=CUTOFF, smooth=False
    )

    path = os.path.join(tmpdir, "exported-model.pt")
    model.save(path)

    # Conservative reference via calculate()
    calc_ref = MetatomicCalculator(
        path, check_consistency=True, non_conservative=False, uncertainty_threshold=None
    )
    atoms.calc = calc_ref
    ref_forces = atoms.get_forces()
    ref_stress = atoms.get_stress()

    # Mixed-mode calculator
    calc = MetatomicCalculator(
        path,
        check_consistency=True,
        non_conservative=non_conservative,
        uncertainty_threshold=None,
    )
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()

    np.testing.assert_allclose(ref.get_potential_energy(), energy, rtol=1e-5)
    assert forces.shape == ref_forces.shape
    assert stress.shape == ref_stress.shape

    if non_conservative == "stress":
        # Forces come from autograd — must match conservative reference
        np.testing.assert_allclose(ref_forces, forces, rtol=1e-5, atol=1e-8)
    elif non_conservative == "forces":
        # Stress comes from autograd — must match conservative reference
        np.testing.assert_allclose(ref_stress, stress, rtol=1e-5, atol=1e-8)


def test_model_without_energy(atoms):
    """
    Test that a MetatomicCalculator can be created with a model without energy
    output.
    """

    # Create a model that only outputs a custom property, no energy
    class NoEnergyModel(torch.nn.Module):
        def forward(
            self,
            systems: List[System],
            outputs: Dict[str, ModelOutput],
            selected_atoms: Optional[Labels] = None,
        ) -> Dict[str, TensorMap]:
            results = {}
            for name in outputs:
                # Return dummy data for each requested output
                block = TensorBlock(
                    values=torch.tensor([[1.0]], dtype=torch.float64),
                    samples=Labels("system", torch.tensor([[0]])),
                    components=torch.jit.annotate(List[Labels], []),
                    properties=Labels(name.split(":")[0], torch.tensor([[0]])),
                )
                tensor = TensorMap(Labels("_", torch.tensor([[0]])), [block])
                results[name] = tensor
            return results

    # Create model capabilities without energy output
    capabilities = ModelCapabilities(
        outputs={
            "feature": ModelOutput(sample_kind="system"),
            "custom::output": ModelOutput(sample_kind="system"),
        },
        atomic_types=[28],
        interaction_range=0.0,
        supported_devices=["cpu"],
        dtype="float64",
    )
    model = AtomisticModel(NoEnergyModel().eval(), ModelMetadata(), capabilities)

    # Should be able to create calculator without error
    calc = MetatomicCalculator(
        model,
        check_consistency=True,
        uncertainty_threshold=None,
    )

    # The calculator should work for additional outputs
    atoms.calc = MetatomicCalculator(
        model,
        additional_outputs={
            "feature": ModelOutput(sample_kind="system"),
        },
        check_consistency=True,
        uncertainty_threshold=None,
    )

    # Should be able to call run_model directly with custom outputs
    outputs = atoms.calc.run_model(
        atoms,
        outputs={"feature": ModelOutput(sample_kind="system")},
    )
    assert "feature" in outputs

    # But trying to get energy should fail with a clear error
    match = "does not support energy-related properties"
    with pytest.raises(PropertyNotImplementedError, match=match):
        atoms.get_potential_energy()

    with pytest.raises(PropertyNotImplementedError, match=match):
        atoms.get_forces()

    # compute_energy should also fail
    match = "does not support energy computation"
    with pytest.raises(ValueError, match=match):
        calc.compute_energy(atoms)


class AdditionalInputModel(torch.nn.Module):
    def __init__(self, inputs: Dict[str, ModelOutput]):
        super().__init__()
        self._requested_inputs = inputs

    def requested_inputs(self) -> Dict[str, ModelOutput]:
        return self._requested_inputs

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        return {
            ("extra::" + input): systems[0].get_data(input)
            for input in self._requested_inputs
        }


def test_additional_input(atoms):
    inputs = {
        "mass": ModelOutput(unit="u", sample_kind="atom"),
        "velocity": ModelOutput(unit="A/fs", sample_kind="atom"),
        "charge": ModelOutput(unit="e", sample_kind="atom"),
        "ase::initial_charges": ModelOutput(unit="e", sample_kind="atom"),
    }
    outputs = {("extra::" + n): inputs[n] for n in inputs}
    capabilities = ModelCapabilities(
        outputs=outputs,
        atomic_types=[28],
        interaction_range=0.0,
        supported_devices=["cpu"],
        dtype="float64",
    )

    model = AtomisticModel(
        AdditionalInputModel(inputs).eval(), ModelMetadata(), capabilities
    )
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0)
    atoms.set_initial_charges([0.0] * len(atoms))
    calculator = MetatomicCalculator(model, check_consistency=True)
    results = calculator.run_model(atoms, outputs)
    for name, tensor in results.items():
        head, name = name.split("::", maxsplit=1)
        assert head == "extra"
        assert name in inputs

        # quantity info is no longer set (deprecated); just check unit is set
        assert tensor.get_info("unit") == inputs[name].unit
        values = tensor[0].values.numpy()

        expected = PER_ATOM_QUANTITIES[name]["getter"](atoms).reshape(values.shape)
        if name == "velocity":
            # ase velocity is in (eV/u)^(1/2) and we requested A/fs
            expected /= ase.units.Angstrom / ase.units.fs

        assert np.allclose(values, expected)


@pytest.mark.parametrize("device,dtype", ALL_DEVICE_DTYPE)
def test_mixed_pbc(model, device, dtype):
    """Test that the calculator works on a mixed-PBC system"""
    atoms = ase.build.fcc111("Ni", size=(2, 2, 3), vacuum=10.0)
    atoms.set_pbc((True, True, False))

    model = _set_model_dtype(model, dtype)

    atoms.calc = MetatomicCalculator(
        model,
        check_consistency=True,
        uncertainty_threshold=None,
        device=device,
    )
    atoms.get_potential_energy()
    atoms.get_forces()

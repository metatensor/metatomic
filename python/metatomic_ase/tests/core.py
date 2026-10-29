"""Core models exercise the ASE boundary independently of the legacy Torch API."""

import os
import subprocess
import sys

import ase
import numpy as np
import pytest
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.lj import LennardJones
from metatensor import Labels, TensorBlock, TensorMap

from metatomic import ModelCapabilities, ModelMetadata, NumpyModel, Quantity
from metatomic.testing import LennardJonesModel
from metatomic_ase import MetatomicCalculator, SymmetrizedCalculator


@pytest.fixture
def atoms():
    return ase.Atoms(
        "Ar4",
        positions=[[0, 0, 0], [1.3, 0.1, 0], [0, 1.4, 0.2], [0.3, 0, 1.5]],
        cell=5 * np.eye(3),
        pbc=True,
    )


def lj(length_unit="angstrom", factor=1.0):
    return LennardJonesModel(
        atomic_type=18,
        cutoff=2.5 / factor,
        sigma=1.0 / factor,
        epsilon=1,
        length_unit=length_unit,
        energy_unit="eV",
        seed=42,
    )


def calculator(model=None, **kwargs):
    return MetatomicCalculator(
        lj() if model is None else model,
        check_consistency=True,
        uncertainty_threshold=None,
        **kwargs,
    )


@pytest.mark.parametrize(
    "length_unit,factor", [("angstrom", 1), ("Bohr", 0.529177210903)]
)
def test_energy_forces_stress(atoms, length_unit, factor):
    reference = atoms.copy()
    reference.calc = LennardJones(sigma=1, epsilon=1, rc=2.5)
    atoms.calc = calculator(lj(length_unit, factor))
    assert isinstance(atoms.calc, MetatomicCalculator)
    assert atoms.get_potential_energy() == pytest.approx(
        reference.get_potential_energy()
    )
    np.testing.assert_allclose(atoms.get_forces(), reference.get_forces(), atol=1e-9)
    np.testing.assert_allclose(atoms.get_stress(), reference.get_stress(), atol=1e-9)
    np.testing.assert_allclose(
        atoms.get_potential_energies(), reference.get_potential_energies(), atol=1e-9
    )
    # Exercise Verlet reuse and rebuilding after displacements.
    for displacement in [0.05, 1.1]:
        atoms.positions[1, 0] += displacement
        reference.positions[1, 0] += displacement
        np.testing.assert_allclose(
            atoms.get_forces(), reference.get_forces(), atol=1e-9
        )


def test_batch_and_selected_atoms(atoms):
    calc = calculator()
    second = atoms.copy()
    second.positions *= 1.03
    batched = calc.compute_energy([atoms, second], True, per_atom=True)
    for i, system in enumerate([atoms, second]):
        single = calc.compute_energy(system, True, per_atom=True)
        for key in single:
            np.testing.assert_allclose(batched[key][i], single[key], atol=1e-12)
    requests = [
        Quantity(name="energy", unit="eV", sample_kind="atom"),
        Quantity(
            name="energy", unit="eV", sample_kind="system", gradients=["positions"]
        ),
    ]
    selected = Labels(["system", "atom"], np.array([[0, 1], [0, 3]], dtype=np.int32))
    per_atom, total = calc.run_model(atoms, requests, selected)
    assert set(per_atom.block().samples.column("atom")) == {1, 3}
    assert per_atom.block().values.sum() == pytest.approx(total.block().values[0, 0])
    assert total.block().gradients_list() == ["positions"]


@pytest.mark.parametrize("mode", [False, True, "forces", "stress"])
def test_non_conservative_and_variants(atoms, mode):
    calc = calculator(non_conservative=mode, variants={"energy": "doubled"})
    result = calc.compute_energy(atoms, True)
    basic = calculator().compute_energy(atoms, True)
    assert result["energy"] == pytest.approx(2 * basic["energy"])
    if mode in (True, "forces"):
        (tensor,) = calc.run_model(
            atoms,
            [
                Quantity(
                    name="non_conservative_force/doubled",
                    unit="eV/angstrom",
                    sample_kind="atom",
                )
            ],
        )
        block = tensor.block()
        values = np.zeros((len(atoms), 3))
        values[block.samples.column("atom")] = block.values[..., 0]
        np.testing.assert_allclose(result["forces"], values - values.mean(axis=0))
    else:
        np.testing.assert_allclose(result["forces"], 2 * basic["forces"])
    if mode in (True, "stress"):
        (tensor,) = calc.run_model(
            atoms,
            [
                Quantity(
                    name="non_conservative_stress/doubled",
                    unit="eV/angstrom^3",
                    sample_kind="system",
                )
            ],
        )
        np.testing.assert_allclose(result["stress"], tensor.block().values[0, ..., 0])
    else:
        np.testing.assert_allclose(result["stress"], 2 * basic["stress"])


def test_additional_outputs_aliases(atoms):
    atoms.calc = calculator(
        additional_outputs={
            "atomic": Quantity(name="energy", unit="eV", sample_kind="atom"),
            "total": Quantity(name="energy", unit="eV", sample_kind="system"),
        }
    )
    energy = atoms.get_potential_energy()
    outputs = atoms.calc.additional_outputs
    assert outputs["atomic"].block().values.sum() == pytest.approx(energy)
    assert outputs["total"].block().values[0, 0] == pytest.approx(energy)


def test_plugin_serialization(atoms, tmp_path):
    path = tmp_path / "model.py"
    path.write_text(
        "from metatomic.testing import LennardJonesModel\n"
        "model = LennardJonesModel(atomic_type=18, cutoff=2.5, sigma=1, "
        "epsilon=1, length_unit='angstrom', energy_unit='eV')\n"
    )
    calc = calculator(
        str(path),
        plugin_name="python",
        model_options={},
        additional_outputs={
            "atomic": Quantity(name="energy", unit="eV", sample_kind="atom"),
        },
    )
    restored = MetatomicCalculator.fromdict(calc.todict())
    assert restored.compute_energy(atoms)["energy"] == pytest.approx(
        calc.compute_energy(atoms)["energy"]
    )
    assert restored.metadata().name == calc.metadata().name


class InputsModel(NumpyModel):
    def __init__(self, name="ase::info::offset", kind="system"):
        self.name = name
        self.kind = kind
        self.calls = 0
        self.requests = []

    def capabilities(self):
        return ModelCapabilities(
            atomic_types=[18],
            interaction_range=0,
            length_unit="angstrom",
            supported_devices=["cpu"],
            dtype="float64",
            outputs=[
                Quantity(name="energy", unit="eV", sample_kind="system"),
                Quantity(name="test::zero", unit="", sample_kind="system"),
            ],
        )

    def metadata(self):
        return ModelMetadata()

    def requested_inputs(self, outputs):
        self.requests.append([q.name for q in outputs])
        if any(q.name == "energy" for q in outputs):
            return [Quantity(name=self.name, unit="", sample_kind=self.kind)]
        return []

    def execute_inner(self, systems, selected_atoms, requested_outputs):
        self.calls += 1
        results = []
        for request in requested_outputs:
            values = [
                s.custom_data(self.name).block().values.sum()
                if request.name == "energy"
                else 0
                for s in systems
            ]
            results.append(
                TensorMap(
                    Labels.range("_", 1),
                    [
                        TensorBlock(
                            np.array(values, dtype=float).reshape(-1, 1),
                            Labels.range("system", len(systems)),
                            [],
                            Labels.range("energy", 1),
                        )
                    ],
                )
            )
        return results


@pytest.mark.parametrize(
    "kind,name", [("system", "ase::info::offset"), ("atom", "ase::arrays::offset")]
)
def test_request_dependent_inputs_and_cache(atoms, kind, name):
    model = InputsModel(name, kind)
    atoms.info["offset"] = 2.0
    atoms.arrays["offset"] = np.ones(len(atoms)) * 2
    atoms.calc = calculator(model)
    expected = 2 * (len(atoms) if kind == "atom" else 1)
    assert atoms.get_potential_energy() == expected
    count = model.calls
    assert atoms.get_potential_energy() == expected
    assert model.calls == count
    if kind == "atom":
        atoms.arrays["offset"] += 1
    else:
        atoms.info["offset"] += 1
    assert atoms.get_potential_energy() == expected * 1.5
    # A different output must not request or require the energy's input.
    del atoms.info["offset"]
    del atoms.arrays["offset"]
    (zero,) = atoms.calc.run_model(
        atoms, [Quantity(name="test::zero", unit="", sample_kind="system")]
    )
    assert zero.block().values[0, 0] == 0


def test_input_provider(atoms):
    def provide(atoms, quantity):
        tensor = TensorMap(
            Labels.range("_", 1),
            [
                TensorBlock(
                    np.array([[atoms.info["offset"]]]),
                    Labels.range("system", 1),
                    [],
                    Labels.range("_", 1),
                )
            ],
        )
        tensor.set_info("unit", "")
        return tensor

    atoms.info["offset"] = 2.0
    atoms.calc = calculator(InputsModel("custom::offset"), input_provider=provide)
    assert atoms.get_potential_energy() == 2
    atoms.info["offset"] = 4.0
    assert atoms.get_potential_energy() == 4


def test_torch_autograd(atoms):
    torch = pytest.importorskip("torch")
    from metatomic import TorchModel

    class Harmonic(TorchModel):
        def capabilities(self):
            return ModelCapabilities(
                atomic_types=[18],
                interaction_range=0,
                length_unit="angstrom",
                supported_devices=["cpu"],
                dtype="float64",
                outputs=[Quantity(name="energy", unit="eV", sample_kind="system")],
            )

        def metadata(self):
            return ModelMetadata()

        def forward(self, systems, selected_atoms, outputs):
            return [
                TensorMap(
                    Labels.range("_", 1),
                    [
                        TensorBlock(
                            torch.stack(
                                [(s.positions**2).sum() for s in systems]
                            ).reshape(-1, 1),
                            Labels.range("system", len(systems)),
                            [],
                            Labels.range("energy", 1),
                        )
                    ],
                )
                for _ in outputs
            ]

    atoms.calc = calculator(Harmonic())
    assert atoms.get_potential_energy() == pytest.approx((atoms.positions**2).sum())
    np.testing.assert_allclose(atoms.get_forces(), -2 * atoms.positions)
    expected = 2 * atoms.positions.T @ atoms.positions / atoms.cell.volume
    np.testing.assert_allclose(atoms.get_stress(voigt=False), expected)


def test_symmetry_and_nonperiodic(atoms):
    base = calculator()
    expected = base.compute_energy(atoms, True)
    message = "SymmetrizedCalculator is deprecated"
    with pytest.warns(DeprecationWarning, match=message):
        atoms.calc = SymmetrizedCalculator(base, l_max=0, include_inversion=True)
    assert atoms.get_potential_energy() == pytest.approx(expected["energy"])
    np.testing.assert_allclose(atoms.get_forces(), expected["forces"], atol=1e-12)
    atoms.pbc = False
    atoms.cell = np.zeros((3, 3))
    atoms.calc = base
    assert np.isfinite(atoms.get_potential_energy())
    assert "stress" not in base.results
    message = "non-zero cell volume"
    with pytest.raises(PropertyNotImplementedError, match=message):
        atoms.get_stress()


def test_without_torch():
    # Reject even attempted imports: importing and running Core models must not
    # require the legacy Torch package or Torch itself.
    code = """
import sys
class NoTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("torch", "metatomic_torch", "metatensor_torch"):
            raise ImportError("Torch deliberately unavailable")
sys.meta_path.insert(0, NoTorch())
from ase import Atoms
from metatomic.testing import LennardJonesModel
from metatomic_ase import MetatomicCalculator
model = LennardJonesModel(atomic_type=18, cutoff=2.5, sigma=1, epsilon=1,
                         length_unit="angstrom", energy_unit="eV")
atoms = Atoms("Ar2", positions=[[0, 0, 0], [1.3, 0, 0]])
atoms.calc = MetatomicCalculator(model, uncertainty_threshold=None)
assert atoms.get_forces().shape == (2, 3)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=os.environ
    )
    assert result.returncode == 0, result.stderr


def test_request_dependent_pair_lists(atoms):
    from metatomic import PairListOptions

    class DynamicPairs(LennardJonesModel):
        def requested_pair_lists(self, outputs):
            if not outputs:
                return []
            options = super().requested_pair_lists(outputs)
            if any(q.sample_kind == "atom" for q in outputs):
                options.append(
                    PairListOptions(cutoff=1.8, full_list=True, strict=False)
                )
            return options

        def execute_inner(self, systems, selected_atoms, requested_outputs):
            expected = self.requested_pair_lists(requested_outputs)
            for system in systems:
                assert set(system.known_pairs()) == set(expected)
            return super().execute_inner(systems, selected_atoms, requested_outputs)

    model = DynamicPairs(
        atomic_type=18,
        cutoff=2.5,
        sigma=1,
        epsilon=1,
        length_unit="angstrom",
        energy_unit="eV",
    )
    calc = calculator(model)
    calc.compute_energy(atoms)
    calc.compute_energy(atoms, per_atom=True)
    calc.compute_energy(atoms)


def test_uncertainty_and_energy_only(atoms):
    atoms.calc = MetatomicCalculator(lj(), uncertainty_threshold=0.001)
    message = "energy uncertainties exceed"
    with pytest.warns(UserWarning, match=message):
        atoms.get_potential_energy()
    atoms.calc = calculator(do_gradients_with_energy=False)
    atoms.get_potential_energy()
    assert set(atoms.calc.results) == {"energy"}
    assert atoms.get_forces().shape == (len(atoms), 3)


def test_subclass(atoms):
    class CustomCalculator(MetatomicCalculator):
        def __init__(self, model, **kwargs):
            super().__init__(model, **kwargs)
            self.was_initialized = True

        def calculate(self, *args, **kwargs):
            super().calculate(*args, **kwargs)
            self.results["energy"] += 1

    atoms.calc = CustomCalculator(lj(), uncertainty_threshold=None)
    assert isinstance(atoms.calc, CustomCalculator)
    assert atoms.calc.was_initialized
    assert atoms.get_potential_energy() == pytest.approx(
        calculator().compute_energy(atoms)["energy"] + 1
    )


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_torch_pairs_autograd(device):
    torch = pytest.importorskip("torch")
    from metatomic import PairListOptions, TorchModel

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    class PairEnergy(TorchModel):
        def capabilities(self):
            return ModelCapabilities(
                atomic_types=[18],
                interaction_range=2,
                length_unit="angstrom",
                supported_devices=["cpu", "cuda", "metal"],
                dtype="float32",
                outputs=[Quantity(name="energy", unit="eV", sample_kind="system")],
            )

        def metadata(self):
            return ModelMetadata()

        def requested_pair_lists(self, outputs):
            return [PairListOptions(cutoff=2, full_list=False)]

        def forward(self, systems, selected_atoms, outputs):
            def labels(name, size):
                return Labels.range(name, size).to(device=device, arrays="torch")

            options = self.requested_pair_lists(outputs)[0]
            values = []
            for system in systems:
                assert system.positions.device.type == device
                values.append((system.pairs(options).values ** 2).sum())
            return [
                TensorMap(
                    labels("_", 1),
                    [
                        TensorBlock(
                            torch.stack(values).reshape(-1, 1),
                            labels("system", len(systems)),
                            [],
                            labels("energy", 1),
                        )
                    ],
                )
                for _ in outputs
            ]

    atoms = ase.Atoms("Ar2", positions=[[0, 0, 0], [1.3, 0.1, 0]])
    atoms.calc = calculator(PairEnergy(), device=device)
    assert atoms.get_potential_energy() == pytest.approx(1.7, abs=1e-6)
    np.testing.assert_allclose(
        atoms.get_forces(), [[2.6, 0.2, 0], [-2.6, -0.2, 0]], atol=1e-6
    )


def test_pair_input_provider(atoms):
    def provide(atoms, quantity):
        tensor = TensorMap(
            Labels.range("_", 1),
            [
                TensorBlock(
                    np.array([[atoms.info["offset"]]]),
                    Labels(
                        [
                            "system",
                            "first_atom",
                            "second_atom",
                            "cell_shift_a",
                            "cell_shift_b",
                            "cell_shift_c",
                        ],
                        np.array([[0, 0, 1, 0, 0, 0]], dtype=np.int32),
                    ),
                    [],
                    Labels.range("_", 1),
                )
            ],
        )
        tensor.set_info("unit", "")
        return tensor

    atoms.info["offset"] = 3.0
    atoms.calc = calculator(
        InputsModel("custom::pair", "atom_pair"), input_provider=provide
    )
    assert atoms.get_potential_energy() == 3.0
    atoms.info["offset"] = 5.0
    assert atoms.get_potential_energy() == 5.0


def test_legacy_subclass(atoms):
    pytest.importorskip("torch")
    import metatomic_lj_test

    class CustomCalculator(MetatomicCalculator):
        def __init__(self, model, **kwargs):
            super().__init__(model, **kwargs)
            self.was_initialized = True

    model = metatomic_lj_test.lennard_jones_model(
        atomic_type=18,
        cutoff=2.5,
        sigma=1,
        epsilon=1,
        length_unit="angstrom",
        energy_unit="eV",
        with_extension=False,
    )
    atoms.calc = CustomCalculator(model, uncertainty_threshold=None)
    assert atoms.calc.was_initialized
    assert isinstance(atoms.calc, CustomCalculator)
    assert atoms.calc.model() is model
    assert atoms.calc.name == "metatomiccalculator"
    assert atoms.get_potential_energy() == pytest.approx(
        calculator().compute_energy(atoms)["energy"]
    )


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("subclass", [False, True])
def test_copy_and_custom_constructor(atoms, legacy, subclass):
    import copy

    model = lj()
    if legacy:
        import metatomic_lj_test

        model = metatomic_lj_test.lennard_jones_model(
            atomic_type=18,
            cutoff=2.5,
            sigma=1,
            epsilon=1,
            length_unit="angstrom",
            energy_unit="eV",
            with_extension=False,
        )

    class CustomCalculator(MetatomicCalculator):
        def __init__(self, *, potential):
            super().__init__(model=potential, uncertainty_threshold=None)
            self.was_initialized = True

        def compute_energy(self, *args, **kwargs):
            result = super().compute_energy(*args, **kwargs)
            result["energy"] += 1
            return result

    if subclass:
        calc = CustomCalculator(potential=model)
        assert calc.was_initialized
        assert isinstance(calc, CustomCalculator)
    else:
        calc = MetatomicCalculator(model, uncertainty_threshold=None)
    cloned = copy.copy(calc)
    assert cloned is not calc
    assert type(cloned) is type(calc)
    assert cloned.model() is model
    expected = calculator().compute_energy(atoms)["energy"] + int(subclass)
    assert cloned.compute_energy(atoms)["energy"] == pytest.approx(expected)

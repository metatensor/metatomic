import sys
import weakref
from typing import Dict, List, Optional, Tuple, Union

import ase.calculators.calculator
import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk, molecule
from ase.cell import Cell
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)
from metatomic_ase import MetatomicCalculator, SymmetrizedCalculator
from metatomic_ase import _symmetry as symmetry_module
from metatomic_ase._symmetry import (
    _average_over_group,
    _choose_quadrature,
    _get_group_operations,
    _get_quadrature,
    _rotate_atoms,
    _RotationalAverageAccumulator,
)


def _body_axis_from_system(system: System) -> torch.Tensor:
    pos = system.positions
    if len(pos) < 2:
        return torch.tensor([0.0, 0.0, 1.0], dtype=pos.dtype, device=pos.device)
    d2 = torch.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=-1)
    idx = torch.argmax(d2)
    b = pos[idx % d2.shape[1]] - pos[idx // d2.shape[1]]
    nrm = torch.linalg.norm(b)
    return (
        b / nrm
        if nrm > 0
        else torch.tensor([0.0, 0.0, 1.0], dtype=pos.dtype, device=pos.device)
    )


def _legendre_0_1_2_3(c: float) -> tuple[float, float, float, float]:
    return 1.0, c, 0.5 * (3 * c**2 - 1.0), 0.5 * (5 * c**3 - 3 * c)


def _compute_rotational_average(results, rotations, weights, store_std):
    batch_size = len(rotations)
    w = weights / weights.sum()

    def _wreshape(x):
        return w.reshape((batch_size,) + (1,) * (x.ndim - 1))

    def _wmean(x):
        return np.sum(_wreshape(x) * x, axis=0)

    def _wstd(x):
        mu = _wmean(x)
        weighted_terms = _wreshape(x) * (x - mu) ** 2
        variance = np.sum(weighted_terms, axis=0)
        scale = np.sum(np.abs(weighted_terms), axis=0)
        dtype = np.result_type(x.dtype, w.dtype, np.float64)
        finfo = np.finfo(dtype)
        tolerance = 64.0 * batch_size * finfo.eps * np.maximum(scale, finfo.tiny)
        if np.any(variance < -tolerance):
            minimum = float(np.min(variance))
            raise ValueError(
                "rotational variance is materially negative "
                f"({minimum}); increase l_max and check quadrature convergence"
            )
        return np.sqrt(np.maximum(variance, 0.0))

    output = {}
    for name, result in results.items():
        values = np.asarray(result, dtype=float)
        if name == "forces":
            values = values @ rotations
        elif name == "stress":
            values = np.swapaxes(rotations, 1, 2) @ values @ rotations
        output[name] = _wmean(values)
        if store_std:
            output[name + "_rot_std"] = _wstd(values)
    return output


def _streaming_average(results, rotations, weights, batch_size, store_std=True):
    accumulator = _RotationalAverageAccumulator(weights, store_std=store_std)
    for start in range(0, len(rotations), batch_size):
        stop = min(start + batch_size, len(rotations))
        accumulator.update(
            {name: values[start:stop] for name, values in results.items()},
            rotations[start:stop],
        )
    return accumulator.finalize()


class MockAnisoModel(torch.nn.Module):
    def __init__(
        self,
        a: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        b: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        c: Tuple[float, float] = (0.0, 0.0),
        p_iso: float = 1.0,
        tensor_forces: bool = False,
        tensor_amp: float = 0.5,
        dtype: torch.dtype = torch.float64,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.a0, self.a1, self.a2, self.a3 = a
        self.b1, self.b2, self.b3 = b
        self.c2, self.c3 = c
        self.p_iso = p_iso
        self.tensor_forces = tensor_forces
        self.tensor_amp = tensor_amp
        self._dtype = dtype
        self._device = torch.device(device)
        self._zhat = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
        self._D = torch.diag(torch.tensor([1.0, -1.0, 0.0], dtype=dtype, device=device))

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        n_sys = len(systems)
        energies: List[torch.Tensor] = []
        stresses: List[torch.Tensor] = []
        forces: List[torch.Tensor] = []

        for system in systems:
            pos = system.positions
            b = _body_axis_from_system(system).to(
                dtype=self._dtype, device=self._device
            )
            cval = float(torch.dot(b, self._zhat))
            P0, P1, P2, P3 = _legendre_0_1_2_3(cval)

            energies.append(
                torch.sum(pos**2)
                + self.a0 * P0
                + self.a1 * P1
                + self.a2 * P2
                + self.a3 * P3
            )

            F_spur = (self.b1 * P1 + self.b2 * P2 + self.b3 * P3) * self._zhat[None, :]
            F = pos.clone() + F_spur
            if self.tensor_forces:
                v = torch.cross(self._zhat, b, dim=0)
                s = torch.norm(v)
                cth = float(torch.dot(self._zhat, b))
                if s < 1e-15:
                    R = (
                        torch.eye(3, dtype=self._dtype, device=self._device)
                        if cth > 0
                        else -torch.eye(3, dtype=self._dtype, device=self._device)
                    )
                else:
                    vx = torch.tensor(
                        [
                            [0.0, -v[2], v[1]],
                            [v[2], 0.0, -v[0]],
                            [-v[1], v[0], 0.0],
                        ],
                        dtype=self._dtype,
                        device=self._device,
                    )
                    R = (
                        torch.eye(3, dtype=self._dtype, device=self._device)
                        + vx
                        + vx @ vx * ((1.0 - cth) / (s**2))
                    )
                F = F + (self.tensor_amp * (R @ self._D @ R.T @ self._zhat))[None, :]
            forces.append(F)

            stresses.append(
                self.p_iso * torch.eye(3, dtype=self._dtype, device=self._device)
                + (self.c2 * P2 + self.c3 * P3) * self._D
            )

        result: Dict[str, TensorMap] = {}
        zero = torch.tensor([[0]], dtype=torch.int64, device=self._device)
        key = Labels("_", zero)
        system_samples = Labels(
            "system",
            torch.arange(n_sys, dtype=torch.int64, device=self._device).unsqueeze(1),
        )
        xyz = torch.arange(3, dtype=torch.int64, device=self._device).reshape(-1, 1)

        if "energy" in outputs:
            result["energy"] = TensorMap(
                key,
                [
                    TensorBlock(
                        torch.stack(energies).unsqueeze(-1),
                        system_samples,
                        [],
                        Labels("energy", zero),
                    )
                ],
            )

        if "non_conservative_force" in outputs:
            force_samples = torch.cat(
                [
                    torch.cartesian_prod(torch.tensor([i]), torch.arange(len(system)))
                    for i, system in enumerate(systems)
                ]
            ).to(dtype=torch.int64, device=self._device)
            result["non_conservative_force"] = TensorMap(
                key,
                [
                    TensorBlock(
                        torch.cat(forces).unsqueeze(-1),
                        Labels(["system", "atom"], force_samples),
                        [Labels("xyz", xyz)],
                        Labels("non_conservative_force", zero),
                    )
                ],
            )

        if "non_conservative_stress" in outputs:
            result["non_conservative_stress"] = TensorMap(
                key,
                [
                    TensorBlock(
                        torch.stack(stresses).unsqueeze(-1),
                        system_samples,
                        [Labels("xyz_1", xyz), Labels("xyz_2", xyz)],
                        Labels("non_conservative_stress", zero),
                    )
                ],
            )

        return result

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return []


def mock_atomistic_model(**options) -> AtomisticModel:
    return AtomisticModel(
        MockAnisoModel(**options).eval(),
        ModelMetadata("mock_aniso", "Mock anisotropic model for testing"),
        ModelCapabilities(
            {
                "energy": ModelOutput(sample_kind="system", unit="eV"),
                "non_conservative_force": ModelOutput(sample_kind="atom", unit="eV/A"),
                "non_conservative_stress": ModelOutput(
                    sample_kind="system", unit="eV/A^3"
                ),
            },
            list(range(1, 102)),
            100,
            "angstrom",
            ["cpu"],
            "float64",
        ),
    )


def mock_calculator(**options) -> MetatomicCalculator:
    return MetatomicCalculator(
        mock_atomistic_model(**options),
        non_conservative=True,
        do_gradients_with_energy=False,
        additional_outputs={
            "energy": ModelOutput(sample_kind="system"),
            "non_conservative_force": ModelOutput(sample_kind="atom"),
            "non_conservative_stress": ModelOutput(sample_kind="system"),
        },
    )


class RequestedInputEnergyModel(torch.nn.Module):
    _requested_inputs: Dict[str, ModelOutput]

    def __init__(self, name: str, unit: str, sample_kind: str):
        super().__init__()
        self.input_name = name
        self._requested_inputs = {name: ModelOutput(unit=unit, sample_kind=sample_kind)}

    def requested_inputs(self) -> Dict[str, ModelOutput]:
        return self._requested_inputs

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        energies: List[torch.Tensor] = []
        for system in systems:
            energies.append(system.get_data(self.input_name).block().values.sum())

        device = systems[0].positions.device
        return {
            "energy": TensorMap(
                Labels("_", torch.tensor([[0]], device=device)),
                [
                    TensorBlock(
                        values=torch.stack(energies).reshape(-1, 1),
                        samples=Labels(
                            "system",
                            torch.arange(len(systems), device=device).reshape(-1, 1),
                        ),
                        components=torch.jit.annotate(List[Labels], []),
                        properties=Labels("energy", torch.tensor([[0]], device=device)),
                    )
                ],
            )
        }


def requested_input_calculator(
    name: str,
    unit: str,
    sample_kind: str = "atom",
    requested_inputs: Optional[Dict[str, ModelOutput]] = None,
) -> MetatomicCalculator:
    model = RequestedInputEnergyModel(name, unit, sample_kind).eval()
    if requested_inputs is not None:
        model._requested_inputs = requested_inputs
    atomistic = AtomisticModel(
        model,
        ModelMetadata(),
        ModelCapabilities(
            {"energy": ModelOutput(sample_kind="system", unit="eV")},
            [1],
            0.0,
            "Angstrom",
            ["cpu"],
            "float64",
        ),
    )
    return MetatomicCalculator(atomistic, check_consistency=True)


def _set_input_dependency(atoms, dependency, value):
    if dependency == "momenta":
        atoms.set_momenta([[value, 0.0, 0.0]])
    elif dependency == "masses":
        atoms.set_masses([value])
    elif dependency == "signal":
        atoms.set_array("signal", np.asarray([value]))
    elif dependency == "bias":
        atoms.info["bias"] = value
    else:  # pragma: no cover - test-table programming error
        raise AssertionError(f"unknown dependency: {dependency}")


@pytest.mark.parametrize(
    ("standard_name", "standard_unit", "custom_name", "info_name", "value"),
    [
        ("charge", "e", "ase::info::charge", "charge", 0.0),
        ("spin_multiplicity", "", "ase::info::spin", "spin", 1),
    ],
)
@pytest.mark.parametrize("custom_first", [False, True])
def test_required_info_cache_watch_dominates_standard_default(
    standard_name,
    standard_unit,
    custom_name,
    info_name,
    value,
    custom_first,
):
    names = (
        [custom_name, standard_name] if custom_first else [standard_name, custom_name]
    )
    units = {standard_name: standard_unit, custom_name: ""}
    requested_inputs = {
        name: ModelOutput(unit=units[name], sample_kind="system") for name in names
    }
    base = requested_input_calculator(
        custom_name,
        "",
        sample_kind="system",
        requested_inputs=requested_inputs,
    )
    calculator = SymmetrizedCalculator(base, l_max=0, include_inversion=False)
    atoms = Atoms("H")
    atoms.info[info_name] = value
    atoms.calc = calculator

    assert atoms.get_potential_energy() == pytest.approx(value)
    del atoms.info[info_name]
    assert info_name in calculator.check_state(atoms)
    with pytest.raises(ValueError, match=f"no info with name '{info_name}'"):
        atoms.get_potential_energy()


@pytest.fixture
def dimer() -> Atoms:
    return Atoms("H2", positions=[[0, 0, 0], [0.3, 0.2, 1.0]])


@pytest.fixture
def fcc_bulk() -> Atoms:
    return bulk("Cu", "fcc", cubic=True)


@pytest.mark.parametrize("Lmax, expect_removed", [(0, False), (3, True)])
def test_energy_L_components_removed(
    dimer: Atoms, Lmax: int, expect_removed: bool
) -> None:
    a = (1.0, 1.0, 1.0, 1.0)
    base = mock_calculator(a=a)
    calc = SymmetrizedCalculator(base, l_max=Lmax)
    dimer.calc = calc
    dimer.get_forces()
    e = dimer.get_potential_energy()
    E_true = float(np.sum(dimer.positions**2))
    if expect_removed:
        assert np.isclose(e, E_true + a[0], atol=1e-10)
    else:
        assert not np.isclose(e, E_true + a[0], atol=1e-10)


def test_tensorial_L2_force_cancellation(dimer: Atoms) -> None:
    base = mock_calculator(tensor_forces=True, tensor_amp=1.0)

    for Lmax in [1, 2, 3]:
        calc = SymmetrizedCalculator(base, l_max=Lmax)
        dimer.calc = calc
        F = dimer.get_forces()
        expected_F = dimer.get_positions()
        expected_F -= np.mean(expected_F, axis=0)
        assert np.allclose(F, expected_F, atol=1e-10)


def test_product_quadrature_cancels_rank_three_vector_integrand() -> None:
    """Exercise the D^3 response times D^1 vector-backrotation product directly."""
    lebedev_order, n_rotations = _choose_quadrature(3)
    rotations, weights = _get_quadrature(
        lebedev_order, n_rotations, include_inversion=False
    )

    body = np.array([0.3, 0.2, 1.0])
    body /= np.linalg.norm(body)
    rotated_body = np.einsum("bij,j->bi", rotations, body)
    cosine = rotated_body[:, 2]
    p3 = 0.5 * (5.0 * cosine**3 - 3.0 * cosine)
    forces = p3[:, None, None] * np.array([0.0, 0.0, 1.0])[None, None, :]

    averaged = _compute_rotational_average(
        {"forces": forces}, rotations, weights, False
    )["forces"]
    assert np.allclose(averaged, np.zeros((1, 3)), atol=1e-12)


def test_stress_trace_is_preserved(fcc_bulk: Atoms) -> None:
    """Averaging preserves the isotropic stress component.

    The traceless part need not vanish: symmetrization makes a rank-2 output
    equivariant, not invariant, so an input-dependent deviatoric stress is valid.
    """
    base = mock_calculator(c=(2.0, 1.0), p_iso=5.0)
    calc = SymmetrizedCalculator(base, l_max=3, include_inversion=True)
    fcc_bulk.calc = calc
    fcc_bulk.get_forces()
    S = fcc_bulk.get_stress(voigt=False)

    assert np.isclose(np.trace(S) / 3.0, 5.0, atol=1e-10)


def test_joint_energy_force_consistency(dimer: Atoms) -> None:
    base = mock_calculator(a=(1, 1, 1, 1), b=(0, 0, 0))
    calc = SymmetrizedCalculator(base, l_max=3)
    dimer.calc = calc
    f = dimer.get_forces()
    e = dimer.get_potential_energy()
    expected_F = dimer.get_positions()
    expected_F -= np.mean(expected_F, axis=0)
    assert np.isclose(e, np.sum(dimer.positions**2) + 1.0, atol=1e-10)
    assert np.allclose(f, expected_F, atol=1e-12)


def _requested_features(atoms, *names):
    class _Model:
        def requested_inputs(self, use_new_names=True):
            assert use_new_names
            return {name: ModelOutput(sample_kind="atom") for name in names}

    class _Base:
        _model = _Model()

    return symmetry_module._requested_per_atom_features(_Base(), atoms)


_MAGMOM_INPUTS = [
    ("ase::initial_magmoms", "e * hbar / (2 * m_e)"),
    ("ase::arrays::initial_magmoms", ""),
]


def test_rotate_atoms_preserves_geometry():
    from scipy.spatial.transform import Rotation

    atoms = Atoms("H2", positions=[[0, 0, 0], [1.2, 0, 0]], cell=np.eye(3), pbc=True)
    atoms.set_momenta([[1, 0, 0], [0, 1, 0]])
    atoms.new_array("custom_vector", np.array([[1, 2, 3], [4, 5, 6]]))
    atoms.new_array("labels3", np.array([["a", "b", "c"], ["d", "e", "f"]]))
    atoms.set_tags([3, 4])
    R = Rotation.from_euler("z", 90, degrees=True).as_matrix()[None, ...]  # 90° about z

    rotated = _rotate_atoms(atoms, R, ["momenta", "custom_vector"])[0]
    assert np.allclose(
        rotated.positions[1] - rotated.positions[0], [0, 1.2, 0], atol=1e-12
    )
    assert np.allclose(rotated.cell[0], [0, 1, 0], atol=1e-12)
    assert np.allclose(
        rotated.get_scaled_positions(wrap=False),
        atoms.get_scaled_positions(wrap=False),
        atol=1e-12,
    )
    d0 = atoms.get_distance(0, 1)
    d1 = rotated.get_distance(0, 1)
    assert np.isclose(d0, d1, atol=1e-12)
    assert np.allclose(rotated.get_momenta(), atoms.get_momenta() @ R[0].T)
    assert np.allclose(
        rotated.arrays["custom_vector"], atoms.arrays["custom_vector"] @ R[0].T
    )
    assert np.array_equal(rotated.get_tags(), atoms.get_tags())
    assert np.array_equal(rotated.arrays["labels3"], atoms.arrays["labels3"])

    inverted = _rotate_atoms(atoms, -np.eye(3)[None, ...], ["momenta"])[0]
    assert np.allclose(inverted.get_momenta(), -atoms.get_momenta())


def test_identity_grid_preserves_unwrapped_periodic_coordinates():
    atoms = Atoms("H", positions=[[1.2, -0.3, 0.5]], cell=np.eye(3), pbc=True)
    atoms.calc = SymmetrizedCalculator(
        mock_calculator(), l_max=0, include_inversion=False
    )

    assert atoms.get_potential_energy() == pytest.approx(
        np.sum(atoms.positions**2), abs=1e-12
    )


def test_space_group_operations_respect_per_atom_arrays():
    atoms = Atoms(
        "H2",
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=np.eye(3),
        pbc=True,
    )
    atoms.set_tags([1, 2])
    features, _ = _requested_features(atoms, "ase::tags")

    rotations, permutations = _get_group_operations(atoms, per_atom_features=features)

    assert len(permutations) > 0
    assert all(
        np.array_equal(atoms.get_tags()[p], atoms.get_tags()) for p in permutations
    )


def test_unrequested_arrays_do_not_reduce_space_group():
    atoms = Atoms("Cu", scaled_positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True)
    atoms.set_momenta([[1.234, 2.345, 3.456]])
    atoms.new_array("labels3", np.array([["a", "b", "c"]]))
    features, vectors = _requested_features(atoms)

    rotations, permutations = _get_group_operations(atoms, per_atom_features=features)

    assert vectors == []
    assert len(rotations) == 48


@pytest.mark.parametrize(("name", "unit"), _MAGMOM_INPUTS)
def test_vector_initial_magmoms_are_rejected_for_improper_quadrature(name, unit):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.set_initial_magnetic_moments([[1.0, 2.0, 3.0]])
    base = requested_input_calculator(name, unit)
    calculator = SymmetrizedCalculator(base, l_max=0, include_inversion=True)

    with pytest.raises(ValueError, match=r"axial O\(3\) parity"):
        calculator.calculate(atoms, ["energy"], [])


@pytest.mark.parametrize(("name", "unit"), _MAGMOM_INPUTS)
def test_vector_initial_magmoms_are_rotated_for_proper_quadrature(name, unit):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    magnetic_moment = np.array([[1.0, 2.0, 3.0]])
    atoms.set_initial_magnetic_moments(magnetic_moment)
    base = requested_input_calculator(name, unit)
    calculator = SymmetrizedCalculator(base, l_max=1, include_inversion=False)

    calculator.calculate(atoms, ["energy"], [])

    rotated = magnetic_moment @ np.swapaxes(calculator.quadrature_rotations, 1, 2)
    expected = np.sum(calculator.quadrature_weights * rotated.sum(axis=(1, 2)))
    assert calculator.results["energy"] == pytest.approx(expected, abs=1e-14)


def test_feature_filtered_space_group_remains_a_projector(fcc_bulk):
    atoms = fcc_bulk.copy()
    atoms.set_initial_charges(np.array([0.0, 0.6, 1.2, 1.8]) * 1.0e-6)
    features, _ = _requested_features(atoms, "ase::initial_charges")
    rotations, permutations = _get_group_operations(atoms, per_atom_features=features)

    rng = np.random.default_rng(123)
    results = {
        "energies": rng.normal(size=len(atoms)),
        "forces": rng.normal(size=(len(atoms), 3)),
        "stress": rng.normal(size=(3, 3)),
    }
    once = _average_over_group(results, rotations, permutations)
    twice = _average_over_group(once, rotations, permutations)
    for name in results:
        assert np.allclose(twice[name], once[name], atol=1e-12)


@pytest.mark.parametrize(
    ("direction", "dtype", "expected_operations"),
    [
        ([1.0e-15, 0.0, 0.0], float, 8),
        ([1.0e20, 1.0, 0.0], float, 2),
        ([1.0, 1.0 + 1.0e-6, 1.0], np.float32, 2),
    ],
    ids=["tiny", "large-component-range", "float32-resolution"],
)
def test_vector_feature_filter_has_componentwise_scale(
    direction, dtype, expected_operations
):
    atoms = Atoms("Cu", scaled_positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True)
    atoms.new_array("direction", np.array([direction], dtype=dtype))
    features, _ = _requested_features(atoms, "ase::arrays::direction")

    rotations, permutations = _get_group_operations(atoms, per_atom_features=features)

    assert len(rotations) == expected_operations


def test_space_group_filter_applies_axial_parity_to_improper_actions():
    atoms = Atoms("Cu", scaled_positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True)
    direction = np.array([[0.0, 0.0, 1.0]])

    polar, _ = _get_group_operations(
        atoms, per_atom_features=[("direction", direction, "polar")]
    )
    axial, _ = _get_group_operations(
        atoms, per_atom_features=[("direction", direction, "axial")]
    )

    polar_improper = [Q for Q in polar if np.linalg.det(Q) < 0]
    axial_improper = [Q for Q in axial if np.linalg.det(Q) < 0]
    assert len(polar) == len(axial) == 8
    assert len(polar_improper) == len(axial_improper) == 4
    assert all(np.isclose(Q[2, 2], 1.0) for Q in polar_improper)
    assert all(np.isclose(Q[2, 2], -1.0) for Q in axial_improper)


def test_group_closure_check_rejects_nonclosed_actions():
    quarter_turn = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert not symmetry_module._group_actions_are_closed(
        [np.eye(3, dtype=int), quarter_turn], [np.array([0]), np.array([0])]
    )


@pytest.mark.parametrize(
    ("direction", "expected_actions"),
    [([1.0, 0.0, 0.0], 6), ([1e20, 0.0, 1.0], 2)],
)
def test_vector_filter_rotated_crystal_little_group(direction, expected_actions):
    axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    transverse = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    cell = np.stack([axis, transverse, np.cross(axis, transverse)]).T
    atoms = Atoms("Cu", scaled_positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=True)
    atoms.new_array("direction", np.array([direction]))
    features, _ = _requested_features(atoms, "ase::arrays::direction")

    rotations, permutations = _get_group_operations(atoms, per_atom_features=features)

    assert len(rotations) == expected_actions


def test_vector_filter_handles_anisotropic_rotated_cell_roundoff():
    from scipy.spatial.transform import Rotation

    orientation = Rotation.random(random_state=972).as_matrix()
    cell = np.diag([1.0, 100.0, 10_000.0]) @ orientation.T
    atoms = Atoms("Cu", scaled_positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=True)
    feature = [("direction", orientation[:, 2][None, :], "polar")]

    rotations, permutations = _get_group_operations(atoms, per_atom_features=feature)

    assert len(rotations) == 4


def test_vector_filter_retains_identity_in_triclinic_cell():
    atoms = Atoms(
        "Cu",
        scaled_positions=[[0.0, 0.0, 0.0]],
        cell=[[2.1, 0.2, 0.1], [0.3, 1.7, 0.2], [0.1, 0.4, 1.9]],
        pbc=True,
    )
    atoms.new_array("direction", np.array([[1.0, 0.0, 0.0]]))
    features, _ = _requested_features(atoms, "ase::arrays::direction")

    rotations, _ = _get_group_operations(atoms, per_atom_features=features)

    assert len(rotations) == 1
    assert np.array_equal(rotations[0], np.eye(3))


def test_choose_quadrature_rules():
    for L in [0, 5, 17, 50]:
        lebedev_order, n_gamma = _choose_quadrature(L)
        assert lebedev_order >= 2 * L
        assert n_gamma == 2 * L + 1

    assert _choose_quadrature(65) == (131, 131)


def test_choose_quadrature_rejects_unsupported_bandwidth():
    with pytest.raises(ValueError, match="largest supported l_max is 65"):
        _choose_quadrature(66)


@pytest.mark.parametrize(
    ("name", "value", "error"),
    [
        ("l_max", -1, ValueError),
        ("l_max", 1.0, TypeError),
        ("l_max", True, TypeError),
        ("batch_size", 0, ValueError),
        ("batch_size", -1, ValueError),
        ("batch_size", 1.0, TypeError),
        ("batch_size", True, TypeError),
    ],
)
def test_constructor_rejects_invalid_discrete_settings(name, value, error):
    arguments = {"l_max": 0, name: value}
    with pytest.raises(error, match=name):
        SymmetrizedCalculator(mock_calculator(), **arguments)


def test_constructor_accepts_numpy_integer_settings():
    calculator = SymmetrizedCalculator(
        mock_calculator(), l_max=np.int64(0), batch_size=np.int64(1)
    )
    assert calculator.l_max == 0
    assert calculator.batch_size == 1


def test_supported_properties_match_base_calculator_contract():
    assert "stresses" not in SymmetrizedCalculator.implemented_properties


def test_calculate_streams_rotations_in_bounded_batches(dimer, monkeypatch):
    class _CountingBase:
        def __init__(self):
            self.calculate_calls = 0
            self.batch_sizes = []
            self.previous_atoms = None
            self.previous_forces = None

        def calculate(self, atoms, properties, system_changes):
            self.calculate_calls += 1

        def compute_energy(
            self, atoms_list, compute_forces_and_stresses=False, per_atom=False
        ):
            if self.previous_atoms is not None:
                assert self.previous_atoms() is None
                assert self.previous_forces() is None
            self.batch_sizes.append(len(atoms_list))
            forces = np.ones((len(atoms_list[0]), 3))
            self.previous_atoms = weakref.ref(atoms_list[0])
            self.previous_forces = weakref.ref(forces)
            return {
                "energy": [float(np.sum(atoms.positions**2)) for atoms in atoms_list],
                "forces": [forces],
            }

    rotation_batch_sizes = []
    original_rotate_atoms = symmetry_module._rotate_atoms

    def counted_rotate_atoms(atoms, rotations, vector_arrays=None):
        rotation_batch_sizes.append(len(rotations))
        return original_rotate_atoms(atoms, rotations, vector_arrays)

    monkeypatch.setattr(symmetry_module, "_rotate_atoms", counted_rotate_atoms)
    base = _CountingBase()
    calculator = SymmetrizedCalculator(
        base,
        l_max=0,
        batch_size=1,
        include_inversion=True,
    )
    calculator.calculate(dimer, ["forces"], [])

    assert base.calculate_calls == 0
    assert base.batch_sizes == [1, 1]
    assert rotation_batch_sizes == [1, 1]
    assert base.previous_atoms() is None
    assert base.previous_forces() is None
    assert np.all(np.isfinite(calculator.results["forces"]))


def test_calculate_rotates_requested_momentum_input():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.set_momenta([[2.0, 0.0, 0.0]])
    base = requested_input_calculator("momentum", "(eV*u)^(1/2)")
    calculator = SymmetrizedCalculator(base, l_max=0, include_inversion=True)

    calculator.calculate(atoms, ["energy"], [])

    assert calculator.results["energy"] == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize(
    (
        "name",
        "unit",
        "sample_kind",
        "dependency",
        "initial",
        "changed",
        "expected",
    ),
    [
        ("momentum", "(eV*u)^(1/2)", "atom", "momenta", 1.0, 2.0, (1.0, 2.0)),
        ("velocity", "(eV/u)^(1/2)", "atom", "masses", 1.0, 2.0, (2.0, 1.0)),
        ("ase::arrays::signal", "", "atom", "signal", 1.0, 2.0, (1.0, 2.0)),
        ("ase::info::bias", "", "system", "bias", 3.0, 9.0, (3.0, 9.0)),
    ],
    ids=["momentum", "velocity-mass", "custom-array", "custom-info"],
)
def test_cache_tracks_every_requested_input_dependency(
    name, unit, sample_kind, dependency, initial, changed, expected
):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    if name == "velocity":
        atoms.set_momenta([[2.0, 0.0, 0.0]])
    _set_input_dependency(atoms, dependency, initial)
    atoms.calc = SymmetrizedCalculator(
        requested_input_calculator(name, unit, sample_kind),
        l_max=0,
        include_inversion=False,
    )

    assert atoms.get_potential_energy() == pytest.approx(expected[0])
    _set_input_dependency(atoms, dependency, changed)
    assert dependency in atoms.calc.check_state(atoms)
    assert atoms.get_potential_energy() == pytest.approx(expected[1])


def test_cache_tracks_in_place_info_mutation():
    atoms = Atoms("H")
    atoms.info["bias"] = np.array(3.0)
    base = requested_input_calculator("ase::info::bias", "", "system")
    atoms.calc = SymmetrizedCalculator(base, l_max=0, include_inversion=False)

    assert atoms.get_potential_energy() == pytest.approx(3.0)
    atoms.info["bias"][...] = 9.0
    assert "bias" in atoms.calc.check_state(atoms)
    assert atoms.get_potential_energy() == pytest.approx(9.0)


def test_calculate_rejects_reserved_custom_positions_input():
    atoms = Atoms("H", positions=[[2.0, 0.0, 0.0]])
    base = requested_input_calculator("ase::arrays::positions", "")
    calculator = SymmetrizedCalculator(base, l_max=0, include_inversion=True)

    with pytest.raises(ValueError, match="must use System.positions"):
        calculator.calculate(atoms, ["energy"], [])


def test_calculate_rotates_boolean_xyz_input_as_converter_labels_it():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.new_array("flags", np.array([[True, False, False]]))
    base = requested_input_calculator("ase::arrays::flags", "")
    calculator = SymmetrizedCalculator(base, l_max=0, include_inversion=True)

    calculator.calculate(atoms, ["energy"], [])

    assert calculator.results["energy"] == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize(
    ("name", "sample_kind", "dependency"),
    [
        ("ase::arrays::signal", "atom", "signal"),
        ("ase::info::bias", "system", "bias"),
    ],
    ids=["custom-array", "custom-info"],
)
@pytest.mark.parametrize(
    ("value", "error"),
    [
        (np.nan, "non-finite"),
        (np.inf, "non-finite"),
        (-np.inf, "non-finite"),
        (1.0 + 2.0j, "real-valued"),
    ],
    ids=["nan", "positive-inf", "negative-inf", "complex"],
)
def test_calculate_rejects_invalid_requested_input(
    name, sample_kind, dependency, value, error
):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    _set_input_dependency(atoms, dependency, value)
    calculator = SymmetrizedCalculator(
        requested_input_calculator(name, "", sample_kind), l_max=0
    )

    with pytest.raises(ValueError, match=f"{dependency}.*{error}"):
        calculator.calculate(atoms, ["energy"], [])


def test_failed_projection_does_not_leave_partial_cached_results(monkeypatch):
    class _EnergyBase:
        def compute_energy(
            self, atoms_list, compute_forces_and_stresses=False, per_atom=False
        ):
            return {
                "energy": [1.0 for _ in atoms_list],
                "forces": [np.ones((len(atoms), 3)) for atoms in atoms_list],
            }

    projection_calls = 0

    def fail_once(atoms, **kwargs):
        nonlocal projection_calls
        projection_calls += 1
        if projection_calls == 1:
            raise RuntimeError("synthetic projection failure")
        return [], []

    monkeypatch.setattr(symmetry_module, "_get_group_operations", fail_once)
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    calculator = SymmetrizedCalculator(
        _EnergyBase(),
        l_max=0,
        include_inversion=False,
        apply_space_group_symmetry=True,
    )
    atoms.calc = calculator

    with pytest.raises(RuntimeError, match="synthetic projection failure"):
        atoms.get_forces()
    assert calculator.results == {}

    assert np.allclose(atoms.get_forces(), 1.0)
    assert projection_calls == 2


def test_scalar_energy_skips_space_group_discovery(fcc_bulk, monkeypatch):
    def unexpected_group_discovery(*args, **kwargs):
        raise AssertionError("scalar total energy needs no space-group projection")

    monkeypatch.setattr(
        symmetry_module, "_get_group_operations", unexpected_group_discovery
    )
    calculator = SymmetrizedCalculator(
        mock_calculator(a=(0.0, 1.0, 0.0, 0.0)),
        l_max=0,
        include_inversion=True,
        apply_space_group_symmetry=True,
        store_rotational_std=True,
    )

    calculator.calculate(fcc_bulk, ["energy"], [])

    assert np.isfinite(calculator.results["energy"])
    assert np.isfinite(calculator.results["energy_rot_std"])


def test_get_quadrature_properties():
    R, w = _get_quadrature(lebedev_order=11, n_rotations=5, include_inversion=False)
    assert np.isclose(np.sum(w), 1.0)
    assert np.allclose([np.dot(r.T, r) for r in R], np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-12)

    R_inv, w_inv = _get_quadrature(
        lebedev_order=11, n_rotations=5, include_inversion=True
    )
    assert len(R_inv) == 2 * len(R)
    dets = np.linalg.det(R_inv)
    assert np.all(np.isin(np.sign(dets).astype(int), [-1, 1]))
    assert np.isclose(np.sum(w_inv), 1.0)


@pytest.mark.parametrize("batch_size", [1, 2, 7, 1000])
def test_streaming_rotational_average_matches_full_grid(batch_size):
    rotations, weights = _get_quadrature(
        lebedev_order=7,
        n_rotations=5,
        include_inversion=True,
    )
    generator = np.random.default_rng(20260711)
    n_rotations = len(rotations)
    n_atoms = 4
    results = {
        "energy": generator.normal(size=n_rotations),
        "energies": generator.normal(size=(n_rotations, n_atoms)),
        "forces": generator.normal(size=(n_rotations, n_atoms, 3)),
        "stress": generator.normal(size=(n_rotations, 3, 3)),
    }
    expected = _compute_rotational_average(results, rotations, weights, True)

    actual = _streaming_average(results, rotations, weights, batch_size)

    assert actual.keys() == expected.keys()
    for name in expected:
        assert np.allclose(actual[name], expected[name], rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize(
    ("excess", "raises"),
    [(1e-15, False), (1e-12, True)],
)
@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_streaming_rotational_std_preserves_signed_weight_guard(
    excess, raises, batch_size
):
    rotations = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
    weights = np.array([1.0, 1.0, -1.0 - excess])
    energy = np.array([0.0, 1.0, 0.0])
    if raises:
        with pytest.raises(ValueError, match="rotational variance.*negative"):
            _streaming_average({"energy": energy}, rotations, weights, batch_size)
    else:
        output = _streaming_average({"energy": energy}, rotations, weights, batch_size)
        assert output["energy_rot_std"] == 0.0


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_streaming_rotational_average_is_stable_around_large_offset(batch_size):
    rotations = np.repeat(np.eye(3)[None, :, :], 5, axis=0)
    weights = np.array([0.1, 0.2, 0.25, 0.15, 0.3])
    energy = 1.0e12 + np.array([0.0, 1.0, -2.0, 3.0, -1.0])
    output = _streaming_average({"energy": energy}, rotations, weights, batch_size)

    normalized = weights / np.sum(weights)
    centered = energy - energy[0]
    expected_mean = energy[0] + np.sum(normalized * centered)
    expected_variance = np.sum(
        normalized * (centered - np.sum(normalized * centered)) ** 2
    )
    assert output["energy"] == expected_mean
    assert output["energy_rot_std"] == pytest.approx(
        np.sqrt(expected_variance), abs=1e-14
    )


def test_streaming_rotational_average_rejects_shape_changes():
    rotations = np.repeat(np.eye(3)[None, :, :], 2, axis=0)
    accumulator = _RotationalAverageAccumulator(np.array([0.5, 0.5]), store_std=False)
    accumulator.update({"forces": np.zeros((1, 2, 3))}, rotations[:1])

    with pytest.raises(ValueError, match="property 'forces' changed shape"):
        accumulator.update({"forces": np.zeros((1, 1, 3))}, rotations[1:])


def test_componentwise_std_uses_complete_backrotated_bandwidth():
    """A bandwidth-3 response times vector back-rotation needs l_max=4 when
    individual squared components, rather than the invariant norm, are requested."""

    def component_variance(l_max):
        order, n_gamma = _choose_quadrature(l_max)
        rotations, weights = _get_quadrature(order, n_gamma, False)
        body_axis = np.array([0.0, 0.0, 1.0])
        lab_z = np.array([0.0, 0.0, 1.0])
        cosine = (rotations @ body_axis)[:, 2]
        response = 0.5 * (5.0 * cosine**3 - 3.0 * cosine)
        forces = response[:, None, None] * lab_z[None, None, :]
        output = _compute_rotational_average(
            {"forces": forces}, rotations, weights, True
        )
        return output["forces_rot_std"][0] ** 2

    under_resolved = component_variance(3)
    resolved = component_variance(4)
    expected = np.array([11.0, 11.0, 23.0]) / 315.0

    # The invariant norm is already exact on both grids, but the components are not.
    assert np.isclose(np.sum(under_resolved), 1.0 / 7.0, atol=1e-12)
    assert np.isclose(np.sum(resolved), 1.0 / 7.0, atol=1e-12)
    assert not np.allclose(under_resolved, expected, atol=1e-3)
    assert np.allclose(resolved, expected, atol=1e-12)


def test_average_over_fcc_group(fcc_bulk: Atoms):
    energy = 0.0
    forces = np.random.default_rng(0).normal(size=(4, 3))
    forces -= np.mean(forces, axis=0)
    stress = np.array([[10.0, 1.0, 0.0], [1.0, 5.0, 0.0], [0.0, 0.0, 1.0]])
    rotations, permutations = _get_group_operations(fcc_bulk)
    out = _average_over_group(
        {"energy": energy, "forces": forces, "stress": stress},
        rotations,
        permutations,
    )

    assert np.isclose(out["energy"], energy)
    assert np.allclose(out["forces"], np.zeros_like(forces))
    assert np.allclose(
        out["stress"], np.eye(3) * np.trace(out["stress"]) / 3.0, atol=1e-8
    )


def test_space_group_average_non_periodic_does_not_import_spglib(monkeypatch):
    atoms = molecule("CH4")
    energy = 0.0
    forces = np.random.default_rng(0).normal(size=(5, 3))
    forces -= np.mean(forces, axis=0)
    monkeypatch.setitem(sys.modules, "spglib", None)

    rotations, permutations = _get_group_operations(atoms)
    assert rotations == []
    assert permutations == []

    out = _average_over_group(
        {"energy": energy, "forces": forces}, rotations, permutations
    )
    assert np.isclose(out["energy"], energy)
    assert np.allclose(out["forces"], forces)


def test_space_group_keeps_translation_distinct_permutations():
    primitive = Atoms(
        numbers=[1, 2, 3],
        scaled_positions=[
            [0.113, 0.227, 0.319],
            [0.367, 0.419, 0.173],
            [0.271, 0.083, 0.449],
        ],
        cell=[[2.1, 0.2, 0.1], [0.3, 1.7, 0.2], [0.1, 0.4, 1.9]],
        pbc=True,
    )
    atoms = primitive.repeat((2, 1, 1))

    Q_list, permutations = _get_group_operations(atoms)
    identity = np.arange(len(atoms))
    identity_rotations = [
        permutation
        for Q, permutation in zip(Q_list, permutations, strict=True)
        if np.allclose(Q, np.eye(3), atol=1e-12)
    ]
    assert len(identity_rotations) >= 2
    assert any(
        not np.array_equal(permutation, identity) for permutation in permutations
    )
    assert all(permutation.shape == (len(atoms),) for permutation in permutations)

    energies = np.arange(len(atoms), dtype=float)
    forces = np.arange(3 * len(atoms), dtype=float).reshape(len(atoms), 3)
    once = _average_over_group(
        {"energies": energies, "forces": forces}, Q_list, permutations
    )
    twice = _average_over_group(once, Q_list, permutations)

    expected_energies = 0.5 * (energies + energies[[3, 4, 5, 0, 1, 2]])
    expected_forces = 0.5 * (forces + forces[[3, 4, 5, 0, 1, 2]])
    assert np.allclose(once["energies"], expected_energies)
    assert np.allclose(once["forces"], expected_forces)
    assert np.allclose(twice["energies"], once["energies"])
    assert np.allclose(twice["forces"], once["forces"])


def test_space_group_matching_uses_cartesian_symprec():
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.positions[0] += [1.0e-4, -2.0e-4, 1.5e-4]

    rotations, permutations = _get_group_operations(atoms, symprec=1.0e-3)

    assert len(rotations) > 1
    assert all(
        len(np.unique(permutation)) == len(atoms) for permutation in permutations
    )


def _skew_site_matching_case(second_site_x):
    frac = np.array(
        [
            [0.9, 0.1, 0.0],
            [second_site_x, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
        ]
    )
    new_frac = frac.copy()
    new_frac[0] = 0.0
    indices = np.arange(len(frac))
    cell = Cell([[1.0, 0.0, 0.0], [0.99, 0.1, 0.0], [0.0, 0.0, 1.0]])
    tolerance = 1.5e-2
    site_indices, radius = symmetry_module._build_periodic_site_indices(
        frac, [indices], cell.array, tolerance
    )
    assert site_indices is not None
    return new_frac, frac, indices, cell, tolerance, site_indices[0], radius


def _assert_group_actions_equal(actual, expected):
    for actual_items, expected_items in zip(actual, expected, strict=True):
        assert len(actual_items) == len(expected_items)
        assert all(
            np.array_equal(left, right)
            for left, right in zip(actual_items, expected_items, strict=True)
        )


@pytest.mark.parametrize(
    "cell,symprec",
    [
        (
            [[3.1, 0.0, 0.0], [2.7, 0.8, 0.0], [2.4, 0.3, 1.1]],
            1.0e-6,
        ),
        (
            [[10.0, 0.0, 0.0], [9.9999, 2.0e-3, 0.0], [0.3, 0.4, 8.0]],
            1.0e-8,
        ),
    ],
    ids=["skew", "ill-conditioned"],
)
def test_periodic_site_index_matches_pairwise_for_adversarial_cells(
    monkeypatch, cell, symprec
):
    atoms = Atoms("Cu", scaled_positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=True).repeat(
        (3, 3, 3)
    )

    indexed_rotations, indexed_permutations = _get_group_operations(
        atoms, symprec=symprec
    )
    monkeypatch.setattr(
        symmetry_module,
        "_build_periodic_site_indices",
        lambda *args, **kwargs: (None, 0.0),
    )
    pairwise_rotations, pairwise_permutations = _get_group_operations(
        atoms, symprec=symprec
    )

    _assert_group_actions_equal(
        (indexed_rotations, indexed_permutations),
        (pairwise_rotations, pairwise_permutations),
    )


@pytest.mark.parametrize(
    ("second_site_x", "expected_index"),
    [(0.98, 0), (0.986, None)],
    ids=["verify-cartesian-nearest", "ambiguous-fallback"],
)
def test_periodic_site_index_verifies_or_falls_back(second_site_x, expected_index):
    # In this skew cell, target 1 is closest in fractional Euclidean distance,
    # while target 0 is closest in Cartesian MIC distance. The index must only
    # generate candidates and leave the final choice to ``find_mic``.
    new_frac, frac, indices, cell, tolerance, site_index, radius = (
        _skew_site_matching_case(second_site_x)
    )

    indexed = symmetry_module._match_species_with_periodic_index(
        new_frac,
        frac,
        indices,
        cell,
        tolerance,
        site_index,
        radius,
    )
    if expected_index is None:
        assert indexed is None
    else:
        pairwise = symmetry_module._match_species_pairwise(
            new_frac, frac, indices, cell
        )
        assert indexed is not None
        assert indexed[0][0] == expected_index
        assert np.array_equal(indexed[0], pairwise[0])
        assert np.allclose(indexed[1], pairwise[1], rtol=0.0, atol=1.0e-15)


@pytest.mark.parametrize(
    ("n_sites", "candidates_per_site"),
    [(8, 8), (1_000, 100)],
    ids=["dense-fraction", "linear-memory-cap"],
)
def test_periodic_site_index_preflights_dense_candidates(n_sites, candidates_per_site):
    class DenseSiteIndex:
        calls = 0

        def query_ball_point(self, points, *, r, return_length=False, **kwargs):
            self.calls += 1
            assert return_length
            return np.full(len(points), candidates_per_site, dtype=np.int64)

    site_index = DenseSiteIndex()
    frac = np.zeros((n_sites, 3))
    matched = symmetry_module._match_species_with_periodic_index(
        frac,
        frac,
        np.arange(len(frac)),
        Cell(np.eye(3)),
        1.0e-6,
        site_index,
        1.0e-6,
    )

    assert matched is None
    assert site_index.calls == 1


def test_space_group_site_matching_falls_back_without_scipy(monkeypatch):
    atoms = Atoms(
        "Cu",
        scaled_positions=[[0.0, 0.0, 0.0]],
        cell=[[3.1, 0.0, 0.0], [2.7, 0.8, 0.0], [2.4, 0.3, 1.1]],
        pbc=True,
    ).repeat((2, 2, 2))
    expected_rotations, expected_permutations = _get_group_operations(atoms)

    monkeypatch.setitem(sys.modules, "scipy.spatial", None)
    actual_rotations, actual_permutations = _get_group_operations(atoms)

    _assert_group_actions_equal(
        (actual_rotations, actual_permutations),
        (expected_rotations, expected_permutations),
    )


def test_space_group_discovery_supports_legacy_spglib_result(monkeypatch):
    import spglib

    def legacy_get_symmetry_dataset(cell, symprec, angle_tolerance):
        return {
            "rotations": np.eye(3, dtype=np.int64)[None, :, :],
            "translations": np.zeros((1, 3)),
        }

    monkeypatch.setattr(spglib, "get_symmetry_dataset", legacy_get_symmetry_dataset)
    atoms = Atoms("Cu", scaled_positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True)

    rotations, permutations = _get_group_operations(atoms)

    assert len(rotations) == 1
    assert np.array_equal(rotations[0], np.eye(3))
    assert np.array_equal(permutations[0], np.array([0]))


def test_symmetrized_calculator_rejects_nonperiodic_3d_stress(dimer):
    calculator = SymmetrizedCalculator(mock_calculator(), l_max=0)
    with pytest.raises(
        ase.calculators.calculator.PropertyNotImplementedError,
        match="3D stress.*periodic",
    ):
        calculator.calculate(dimer, ["stress"], [])


@pytest.mark.parametrize(
    "cell",
    [
        np.zeros((3, 3)),
        np.diag([1.0, 1.0, np.nan]),
        np.diag([1.0, 1.0, np.inf]),
    ],
    ids=["singular", "nan", "inf"],
)
@pytest.mark.parametrize("properties", [["stress"], ["forces"]])
def test_symmetrized_calculator_rejects_invalid_3d_stress_volume(cell, properties):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=True)
    calculator = SymmetrizedCalculator(mock_calculator(), l_max=0)

    with pytest.raises(ValueError, match="singular or non-finite periodic cell"):
        calculator.calculate(atoms, properties, [])


def test_symmetrized_calculator_matches_symmetrized_model(fcc_bulk: Atoms) -> None:
    """SymmetrizedModel and SymmetrizedCalculator implement the same O(3)
    average: evaluated on the same quadrature grid, energy, forces, and stress
    must agree within floating-point tolerance."""
    from metatomic.torch import systems_to_torch
    from metatomic.torch.symmetrized_model import SymmetrizedModel

    a = (1.0, 0.8, 0.6, 0.4)
    b = (0.5, 0.4, 0.3)
    c = (0.7, 0.2)
    p_iso = 2.0

    atoms = fcc_bulk
    # keep every position strictly inside the cell: the calculator wraps
    # positions, and the mock is (deliberately) not translation invariant
    atoms.translate([1.0, 1.0, 1.0])
    atoms.rattle(0.1, seed=0)

    symm_model = SymmetrizedModel(
        mock_atomistic_model(a=a, b=b, c=c, p_iso=p_iso),
        max_o3_lambda_target=2,
        max_o3_lambda_grid=3,
        batch_size=4,
    )
    outputs = {
        "energy": ModelOutput(sample_kind="system"),
        "non_conservative_force": ModelOutput(sample_kind="atom"),
        "non_conservative_stress": ModelOutput(sample_kind="system"),
    }
    systems = systems_to_torch([atoms], dtype=torch.float64)
    low_level = symm_model(systems, outputs)

    ase_calculator = SymmetrizedCalculator(
        mock_calculator(a=a, b=b, c=c, p_iso=p_iso),
        l_max=3,
        batch_size=4,
        include_inversion=True,
    )
    ase_calculator.quadrature_rotations = np.concatenate(
        [symm_model.so3_rotations.numpy(), (-symm_model.so3_rotations).numpy()],
        axis=0,
    )
    weights = symm_model._so3_weights_float64.numpy()
    ase_calculator.quadrature_weights = np.concatenate(
        [0.5 * weights, 0.5 * weights],
        axis=0,
    )

    atoms.calc = ase_calculator
    ase_energy = atoms.get_potential_energy()
    ase_forces = atoms.get_forces()
    ase_stress = atoms.get_stress(voigt=False)

    low_energy = low_level["energy_l0_mean"].block().values.squeeze().item()

    # forces are decomposed to spherical (m=-1, 0, 1) = (y, z, x) order; roll
    # back to Cartesian; the calculator removes the net force
    low_forces = (
        low_level["non_conservative_force_l1_mean"]
        .block()
        .values.roll(1, 1)
        .squeeze(-1)
        .numpy()
    )
    low_forces = low_forces - low_forces.mean(axis=0, keepdims=True)

    l0 = low_level["non_conservative_stress_l0_mean"].block().values.squeeze().item()
    l2 = low_level["non_conservative_stress_l2_mean"].block().values.squeeze().numpy()
    l2_diag = l2[2] / np.sqrt(3.0)
    low_stress = np.array(
        [
            [l2[4] - l2_diag, l2[0], l2[3]],
            [l2[0], -l2[4] - l2_diag, l2[1]],
            [l2[3], l2[1], 2.0 * l2_diag],
        ]
    ) + np.eye(3) * (l0 / 3.0)

    assert np.isclose(ase_energy, low_energy, atol=1e-12)
    assert np.allclose(ase_forces, low_forces, atol=1e-12)
    assert np.allclose(ase_stress, low_stress, atol=1e-12)

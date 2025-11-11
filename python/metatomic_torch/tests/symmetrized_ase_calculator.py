from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk, molecule
from metatensor.torch import Labels, TensorBlock, TensorMap

import metatomic.torch as mta
from metatomic.torch import (
    ModelOutput,
    NeighborListOptions,
    System,
)
from metatomic.torch.ase_calculator import SymmetrizedCalculator, _get_quadrature


def _body_axis_from_system(system: System) -> torch.Tensor:
    """
    Return the normalized vector connecting the two farthest atoms.

    :param system: System.
    :return: Normalized 3D vector defining the body axis.
    """
    pos = system.positions
    if len(pos) < 2:
        return torch.tensor([0.0, 0.0, 1.0], dtype=pos.dtype, device=pos.device)
    d2 = torch.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=-1)
    # i, j = torch.unravel_index(torch.argmax(d2), d2.shape) # for newer PyTorch
    idx = torch.argmax(d2)
    i = idx // d2.shape[1]
    j = idx % d2.shape[1]
    b = pos[j] - pos[i]
    nrm = torch.linalg.norm(b)
    return (
        b / nrm
        if nrm > 0
        else torch.tensor([0.0, 0.0, 1.0], dtype=pos.dtype, device=pos.device)
    )


def _legendre_0_1_2_3(c: float) -> tuple[float, float, float, float]:
    """
    Compute Legendre polynomials P0..P3(c).

    :param c: Cosine between the body axis and the lab z-axis.
    :return: Tuple (P0, P1, P2, P3).
    """
    P0 = 1.0
    P1 = c
    P2 = 0.5 * (3 * c**2 - 1.0)
    P3 = 0.5 * (5 * c**3 - 3 * c)
    return P0, P1, P2, P3


class MockAnisoModel(torch.nn.Module):
    """
    Deterministic, rotation-dependent mock for testing SymmetrizedCalculator.

    Components:
      - Energy:  E_true + a1*P1 + a2*P2 + a3*P3
      - Forces:  F_true + (b1*P1 + b2*P2 + b3*P3)*zhat + optional tensor L=2 term
      - Stress:  p_iso*I + (c2*P2 + c3*P3)*D

    :param a: Coefficients for Legendre P0..P3 in the energy.
    :param b: Coefficients for P1..P3 in the forces (spurious vector parts).
    :param c: Coefficients for P2,P3 in the stress (spurious deviators).
    :param p_iso: Isotropic (true) part of the stress tensor.
    :param tensor_forces: If True, add L=2 tensor-coupled force term.
    :param tensor_amp: Amplitude of the tensor-coupled force component.
    :param dtype: Data type for internal tensors.
    :param device: Device for internal tensors.
    """

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

        # Fixed bases
        self._zhat = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
        self._D = torch.diag(torch.tensor([1.0, -1.0, 0.0], dtype=dtype, device=device))

    @torch.jit.export
    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        n_sys = len(systems)

        # Pre-allocate storages
        energies: List[torch.Tensor] = []
        stresses: List[torch.Tensor] = []
        forces: List[torch.Tensor] = []

        for sys in systems:
            pos = sys.positions

            # Determine body axis and related scalars
            b = _body_axis_from_system(sys).to(dtype=self._dtype, device=self._device)
            cval = float(torch.dot(b, self._zhat))
            P0, P1, P2, P3 = _legendre_0_1_2_3(cval)

            # Energy
            E_true = torch.sum(pos**2)
            E = E_true + self.a0 * P0 + self.a1 * P1 + self.a2 * P2 + self.a3 * P3
            energies.append(E)

            # Forces
            F_true = pos.clone()
            F_spur = (self.b1 * P1 + self.b2 * P2 + self.b3 * P3) * self._zhat[None, :]
            F = F_true + F_spur
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
                T = R @ self._D @ R.T
                F_tensor = self.tensor_amp * (T @ self._zhat)
                F = F + F_tensor[None, :]
            forces.append(F)

            # Stress
            S = (
                self.p_iso * torch.eye(3, dtype=self._dtype, device=self._device)
                + (self.c2 * P2 + self.c3 * P3) * self._D
            )
            stresses.append(S)

        result: Dict[str, TensorMap] = {}
        key = Labels(
            names=["_"],
            values=torch.tensor([[0]], dtype=torch.int64, device=self._device),
        )

        # Energy
        energy_block = TensorBlock(
            values=torch.stack(energies, dim=0)
            .to(dtype=self._dtype, device=self._device)
            .unsqueeze(-1),
            samples=Labels(
                names=["system"],
                values=torch.arange(
                    n_sys, dtype=torch.int64, device=self._device
                ).unsqueeze(1),
            ),
            components=[],
            properties=Labels(
                names=["energy"],
                values=torch.tensor([[0]], dtype=torch.int64, device=self._device),
            ),
        )

        # Forces
        force_block = TensorBlock(
            values=torch.cat(forces, dim=0)
            .to(dtype=self._dtype, device=self._device)
            .unsqueeze(-1),
            samples=Labels(
                names=["system", "atom"],
                values=torch.cat(
                    [
                        torch.cartesian_prod(torch.tensor([i]), torch.arange(len(sys)))
                        for i, sys in enumerate(systems)
                    ]
                ).to(dtype=torch.int64, device=self._device),
            ),
            components=[
                Labels(
                    "xyz",
                    torch.arange(3)
                    .reshape(-1, 1)
                    .to(dtype=torch.int64, device=self._device),
                )
            ],  # vector components
            properties=Labels(
                names=["non_conservative_forces"],
                values=torch.tensor([[0]], dtype=torch.int64, device=self._device),
            ),
        )

        # Stress
        stress_block = TensorBlock(
            values=torch.stack(stresses, dim=0)
            .to(dtype=self._dtype, device=self._device)
            .unsqueeze(-1),
            samples=Labels(
                names=["system"],
                values=torch.arange(
                    n_sys, dtype=torch.int64, device=self._device
                ).unsqueeze(1),
            ),
            components=[
                Labels(
                    "xyz_1",
                    torch.arange(3)
                    .reshape(-1, 1)
                    .to(dtype=torch.int64, device=self._device),
                ),
                Labels(
                    "xyz_2",
                    torch.arange(3)
                    .reshape(-1, 1)
                    .to(dtype=torch.int64, device=self._device),
                ),
            ],
            properties=Labels(
                names=["non_conservative_stress"],
                values=torch.tensor([[0]], dtype=torch.int64, device=self._device),
            ),
        )

        if "energy" in outputs:
            result["energy"] = TensorMap(key, [energy_block])

        if "non_conservative_forces" in outputs:
            result["non_conservative_forces"] = TensorMap(key, [force_block])

        if "non_conservative_stress" in outputs:
            result["non_conservative_stress"] = TensorMap(key, [stress_block])

        return result

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return []


def mock_calculator(
    a: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    b: tuple[float, float, float] = (0.0, 0.0, 0.0),
    c: tuple[float, float] = (0.0, 0.0),
    p_iso: float = 1.0,
    tensor_forces: bool = False,
    tensor_amp: float = 0.5,
) -> mta.ase_calculator.MetatomicCalculator:
    model = MockAnisoModel(
        a=a,
        b=b,
        c=c,
        p_iso=p_iso,
        tensor_forces=tensor_forces,
        tensor_amp=tensor_amp,
    )
    model.eval()

    atomistic_model = mta.AtomisticModel(
        model,
        mta.ModelMetadata("mock_aniso", "Mock anisotropic model for testing"),
        mta.ModelCapabilities(
            {
                "energy": mta.ModelOutput(per_atom=False),
                "non_conservative_forces": mta.ModelOutput(per_atom=True),
                "non_conservative_stress": mta.ModelOutput(per_atom=False),
            },
            list(range(1, 102)),
            100,
            "angstrom",
            ["cpu"],
            "float64",
        ),
    )
    return mta.ase_calculator.MetatomicCalculator(
        atomistic_model,
        non_conservative=True,
        do_gradients_with_energy=False,
        additional_outputs={
            "energy": mta.ModelOutput(per_atom=False),
            "non_conservative_forces": mta.ModelOutput(per_atom=True),
            "non_conservative_stress": mta.ModelOutput(per_atom=False),
        },
    )


@pytest.fixture
def dimer() -> Atoms:
    """
    Create a small asymmetric geometry with a well-defined body axis.

    :return: ASE Atoms object with the H2 molecule.
    """
    return Atoms("H2", positions=[[0, 0, 0], [0.3, 0.2, 1.0]])


@pytest.fixture
def fcc_bulk() -> Atoms:
    """
    Create a small FCC bulk structure.

    :return: ASE Atoms object with FCC Cu.
    """
    return bulk("Cu", "fcc", cubic=True)


def test_quadrature_normalization():
    """Verify normalization and determinant signs of the quadrature."""
    R, w = _get_quadrature(lebedev_order=11, n_rotations=5, include_inversion=True)
    assert np.isclose(np.sum(w), 1.0)
    dets = np.linalg.det(R)
    assert np.all(np.isin(np.round(dets).astype(int), [-1, 1]))


@pytest.mark.parametrize("Lmax, expect_removed", [(0, False), (3, True)])
def test_energy_L_components_removed(
    dimer: Atoms, Lmax: int, expect_removed: bool
) -> None:
    """
    Verify that spurious energy components vanish once rotational averaging is applied.
    For Lmax>0, all use the same minimal Lebedev rule (order=3).
    """
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


def test_force_backrotation_exact(dimer: Atoms) -> None:
    """
    Check that forces are back-rotated exactly when no spurious terms are present.

    :param dimer: Test atomic structure.
    """
    base = mock_calculator(b=(0, 0, 0))
    calc = SymmetrizedCalculator(base, l_max=3)
    dimer.calc = calc
    F = dimer.get_forces()
    expected_F = dimer.get_positions()
    expected_F -= np.mean(expected_F, axis=0)
    assert np.allclose(F, expected_F, atol=1e-12)


def test_tensorial_L2_force_cancellation(dimer: Atoms) -> None:
    """
    Tensor-coupled (L=2) force components must vanish under O(3) averaging.

    Since the minimal Lebedev order used internally is 3, all quadratures
    integrate L=2 components exactly; we only check for correct cancellation.
    """
    base = mock_calculator(tensor_forces=True, tensor_amp=1.0)

    for Lmax in [1, 2, 3]:
        calc = SymmetrizedCalculator(base, l_max=Lmax)
        dimer.calc = calc
        F = dimer.get_forces()
        expected_F = dimer.get_positions()
        expected_F -= np.mean(expected_F, axis=0)
        assert np.allclose(F, expected_F, atol=1e-10)


def test_stress_isotropization(fcc_bulk: Atoms) -> None:
    """
    Check that stress deviatoric parts (L=2,3) vanish under full O(3) averaging.

    :param fcc_bulk: Test atomic structure.
    """
    base = mock_calculator(c=(2.0, 1.0), p_iso=5.0)
    calc = SymmetrizedCalculator(base, l_max=9, include_inversion=True)
    fcc_bulk.calc = calc
    fcc_bulk.get_forces()
    S = fcc_bulk.get_stress(voigt=False)

    fcc_bulk.calc = base
    fcc_bulk.get_forces()

    iso = np.trace(S) / 3.0
    assert np.isclose(iso, 5.0, atol=1e-10)


def test_cancellation_vs_Lmax(dimer: Atoms) -> None:
    """
    Residual anisotropy must vanish once rotational averaging is applied.
    All quadratures with Lmax>0 are equivalent (Lebedev order=3).
    """
    a = (0.0, 0.0, 1.0, 1.0)
    base = mock_calculator(a=a)
    E_true = float(np.sum(dimer.positions**2))

    # No averaging
    calc0 = SymmetrizedCalculator(base, l_max=0)
    dimer.calc = calc0
    dimer.get_forces()
    e0 = dimer.get_potential_energy()

    # Averaged
    calc3 = SymmetrizedCalculator(base, l_max=3)
    dimer.calc = calc3
    dimer.get_forces()
    e3 = dimer.get_potential_energy()

    assert not np.isclose(e0, E_true, atol=1e-10)
    assert np.isclose(e3, E_true, atol=1e-10)


def test_joint_energy_force_consistency(dimer: Atoms) -> None:
    """
    Combined test: both energy and forces are consistent and invariant.

    :param dimer: Test atomic structure.
    """
    base = mock_calculator(a=(1, 1, 1, 1), b=(0, 0, 0))
    calc = SymmetrizedCalculator(base, l_max=3)
    dimer.calc = calc
    f = dimer.get_forces()
    e = dimer.get_potential_energy()
    expected_F = dimer.get_positions()
    expected_F -= np.mean(expected_F, axis=0)
    assert np.isclose(e, np.sum(dimer.positions**2) + 1.0, atol=1e-10)
    assert np.allclose(f, expected_F, atol=1e-12)


def test_rotate_atoms_preserves_geometry(tmp_path):
    """Check that _rotate_atoms applies rotations correctly and preserves distances."""
    from scipy.spatial.transform import Rotation

    from metatomic.torch.ase_calculator import _rotate_atoms

    # Build simple cubic cell with 2 atoms along x
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], cell=np.eye(3))
    R = Rotation.from_euler("z", 90, degrees=True).as_matrix()[None, ...]  # 90° about z

    rotated = _rotate_atoms(atoms, R)[0]
    # Positions should now align along y
    assert np.allclose(
        rotated.positions[1] - rotated.positions[0], [0, 1, 0], atol=1e-12
    )
    # Cell rotated
    assert np.allclose(rotated.cell[0], [0, 1, 0], atol=1e-12)
    # Distances preserved
    d0 = atoms.get_distance(0, 1)
    d1 = rotated.get_distance(0, 1)
    assert np.isclose(d0, d1, atol=1e-12)


def test_choose_quadrature_rules():
    """Check that _choose_quadrature selects appropriate rules."""
    from metatomic.torch.ase_calculator import _choose_quadrature

    for L in [0, 5, 17, 50]:
        lebedev_order, n_gamma = _choose_quadrature(L)
        assert lebedev_order >= L
        assert n_gamma == 2 * L + 1


def test_get_quadrature_properties():
    """Check properties of the quadrature returned by _get_quadrature."""
    from metatomic.torch.ase_calculator import _get_quadrature

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


def test_compute_rotational_average_identity():
    """Check that _compute_rotational_average produces correct averages."""
    from metatomic.torch.ase_calculator import _compute_rotational_average

    R = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
    w = np.ones(3) / 3
    results = {
        "energy": np.array([1.0, 2.0, 3.0]),
        "forces": np.array([[[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]]]),
        "stress": np.array([np.eye(3), 2 * np.eye(3), 3 * np.eye(3)]),
    }
    out = _compute_rotational_average(results, R, w, False)
    assert np.isclose(out["energy"], np.mean(results["energy"]))
    assert np.allclose(out["forces"], np.mean(results["forces"], axis=0))
    assert np.allclose(out["stress"], np.mean(results["stress"], axis=0))

    out = _compute_rotational_average(results, R, w, True)
    assert "energy_rot_std" in out
    assert "forces_rot_std" in out
    assert "stress_rot_std" in out


def test_average_over_fcc_group(fcc_bulk: Atoms):
    """
    Check that averaging over the space group of an FCC crystal
    produces an isotropic (scalar) stress tensor.
    """
    from metatomic.torch.ase_calculator import (
        _average_over_group,
        _get_group_operations,
    )

    # FCC conventional cubic cell (4 atoms)
    atoms = fcc_bulk

    energy = 0.0
    forces = np.random.normal(0, 1, (4, 3))
    forces -= np.mean(forces, axis=0)  # Ensure zero net force

    # Create an intentionally anisotropic stress
    stress = np.array([[10.0, 1.0, 0.0], [1.0, 5.0, 0.0], [0.0, 0.0, 1.0]])
    results = {"energy": energy, "forces": forces, "stress": stress}

    Q_list, P_list = _get_group_operations(atoms)
    out = _average_over_group(results, Q_list, P_list)

    # Energy must be unchanged
    assert np.isclose(out["energy"], energy)

    # Forces must average to zero by symmetry
    F_pg = out["forces"]
    assert np.allclose(F_pg, np.zeros_like(F_pg))

    S_pg = out["stress"]

    # The averaged stress must be isotropic: S_pg = (trace/3)*I
    iso = np.trace(S_pg) / 3.0
    assert np.allclose(S_pg, np.eye(3) * iso, atol=1e-8)


def test_space_group_average_non_periodic():
    """
    Check that averaging over the space group of a non-periodic system leaves the
    results unchanged.
    """
    from metatomic.torch.ase_calculator import (
        _average_over_group,
        _get_group_operations,
    )

    # Methane molecule (Td symmetry)
    atoms = molecule("CH4")

    energy = 0.0
    forces = np.random.normal(0, 1, (5, 3))
    forces -= np.mean(forces, axis=0)  # Ensure zero net force

    results = {"energy": energy, "forces": forces}

    Q_list, P_list = _get_group_operations(atoms)

    # Check that the operation lists are empty
    assert len(Q_list) == 0
    assert len(P_list) == 0

    out = _average_over_group(results, Q_list, P_list)

    # Energy must be unchanged
    assert np.isclose(out["energy"], energy)

    # Forces must be unchanged
    F_pg = out["forces"]
    assert np.allclose(F_pg, forces)

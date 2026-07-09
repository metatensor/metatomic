"""Tests for symmetrized_model.py standalone functions and SymmetrizedModel class."""

from typing import Dict, List, Optional

import metatensor.torch as mts
import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from scipy.spatial.transform import Rotation

from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
    load_atomistic_model,
)
from metatomic.torch.o3._wigner import build_wigner_D_cache
from metatomic.torch.symmetrized_model import (
    SymmetrizedModel,
    get_euler_angles_quadrature,
    per_system_character_fractions,
    per_system_equivariance_rmse,
)
from metatomic.torch.symmetrized_model._decompose import (
    _l0_components_from_matrices,
    _l2_components_from_matrices,
)
from metatomic.torch.symmetrized_model._gradients import _evaluate_with_gradients
from metatomic.torch.symmetrized_model._quadrature import (
    _choose_quadrature,
    _rotations_from_angles,
)


class TestL0Components:
    """Test extraction of L=0 (trace) components from 3x3 matrices."""

    def test_identity_trace(self):
        # Identity matrix has trace 3. The function expects shape (a, 3, 3, b).
        A = torch.eye(3, dtype=torch.float64).unsqueeze(0).unsqueeze(-1)
        result = _l0_components_from_matrices(A)
        assert result.shape == (1, 1, 1)
        assert torch.allclose(result, torch.tensor([[[3.0]]], dtype=torch.float64))

    def test_traceless_matrix(self):
        # A traceless matrix should give L=0 = 0
        M = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        A = M.unsqueeze(0).unsqueeze(-1)
        result = _l0_components_from_matrices(A)
        assert torch.allclose(
            result, torch.tensor([[[0.0]]], dtype=torch.float64), atol=1e-14
        )

    def test_batch_dimensions(self):
        # Test with batch size > 1 and multiple properties
        batch = 5
        n_prop = 3
        A = torch.randn(batch, 3, 3, n_prop, dtype=torch.float64)
        result = _l0_components_from_matrices(A)
        assert result.shape == (batch, 1, n_prop)
        for i in range(batch):
            for p in range(n_prop):
                expected_trace = A[i, 0, 0, p] + A[i, 1, 1, p] + A[i, 2, 2, p]
                assert torch.allclose(result[i, 0, p], expected_trace, atol=1e-14)


class TestL2Components:
    """Test extraction of L=2 (symmetric traceless) components from 3x3 matrices."""

    def test_identity_gives_zero(self):
        # Identity is proportional to L=0 only; L=2 components should be zero.
        A = torch.eye(3, dtype=torch.float64).unsqueeze(0).unsqueeze(-1)
        result = _l2_components_from_matrices(A)
        assert result.shape == (1, 5, 1)
        assert torch.allclose(
            result, torch.zeros(1, 5, 1, dtype=torch.float64), atol=1e-14
        )

    def test_diagonal_traceless(self):
        # diag(1, -1, 0) is traceless and has known L=2 components
        M = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        A = M.unsqueeze(0).unsqueeze(-1)
        result = _l2_components_from_matrices(A)
        assert result.shape == (1, 5, 1)
        # m=0: (2*0 - 1 - (-1)) / (2*sqrt(3)) = 0
        assert torch.allclose(result[0, 2, 0], torch.tensor(0.0, dtype=torch.float64))
        # m=2 (last component): (1 - (-1)) / 2 = 1
        assert torch.allclose(result[0, 4, 0], torch.tensor(1.0, dtype=torch.float64))

    def test_frobenius_norm_relation(self):
        """For a symmetric traceless matrix S, the L=2 decomposition should satisfy
        a norm relation: sum(c_i^2) relates to (1/2) * sum(S_ij * S_ji).
        """
        # Build a symmetric traceless matrix
        S = torch.tensor(
            [[2.0, 1.0, 0.5], [1.0, -1.0, 0.3], [0.5, 0.3, -1.0]],
            dtype=torch.float64,
        )
        A = S.unsqueeze(0).unsqueeze(-1)
        l2 = _l2_components_from_matrices(A)
        l2_norm_sq = (l2**2).sum()

        # The L=2 norm squared should equal half the Frobenius norm of the
        # symmetric part (since the decomposition extracts the symmetric part)
        sym_S = 0.5 * (S + S.T)
        half_frob = 0.5 * (sym_S**2).sum()
        # They won't be exactly equal because S has an L=0 part too.
        # But for a traceless symmetric matrix, L=0 is zero, so they match.
        trace = S[0, 0] + S[1, 1] + S[2, 2]
        assert abs(trace) < 1e-14, "Matrix should be traceless for this test"
        assert torch.allclose(l2_norm_sq, half_frob, atol=1e-12)


class TestDecomposeStressRoundtrip:
    """Test that L=0 + L=2 decomposition covers the symmetric part of a 3x3 tensor."""

    def test_norm_conservation(self):
        """The sum of L=0 and L=2 squared norms should equal
        the Frobenius norm squared of the symmetrized matrix."""
        M = torch.randn(1, 3, 3, 1, dtype=torch.float64)
        sym_M = 0.5 * (M + M.transpose(1, 2))

        l0 = _l0_components_from_matrices(sym_M)
        l2 = _l2_components_from_matrices(sym_M)

        # L=0 norm: trace^2 / 3 (the trace component carries norm trace^2/3
        # in the irrep normalization). Actually, the L=0 extraction returns
        # the raw trace, and L=2 the 5 components. Let's check reconstruction.
        trace_val = l0[0, 0, 0]
        # Reconstruct L=0 part: (trace/3) * I
        l0_matrix = (trace_val / 3.0) * torch.eye(3, dtype=torch.float64)

        # Reconstruct L=2 part from components
        c = l2[0, :, 0]  # 5 components: (m=-2, m=-1, m=0, m=1, m=2)
        l2_matrix = torch.zeros(3, 3, dtype=torch.float64)
        # Reverse of the extraction formulas:
        l2_matrix[0, 1] = c[0]
        l2_matrix[1, 0] = c[0]
        l2_matrix[1, 2] = c[1]
        l2_matrix[2, 1] = c[1]
        l2_matrix[0, 2] = c[3]
        l2_matrix[2, 0] = c[3]
        l2_matrix[0, 0] = c[4] + c[2] * np.sqrt(3) / 3 * (-1)
        l2_matrix[1, 1] = -c[4] + c[2] * np.sqrt(3) / 3 * (-1)
        l2_matrix[2, 2] = c[2] * 2.0 * np.sqrt(3) / 3

        reconstructed = l0_matrix + l2_matrix
        original_sym = sym_M[0, :, :, 0]
        assert torch.allclose(reconstructed, original_sym, atol=1e-12)


class TestWignerD:
    """Test properties of real Wigner D matrices."""

    def test_orthogonality(self):
        """D(R)^T D(R) = I for all ell."""
        rng = np.random.default_rng(42)
        R = Rotation.random(5, random_state=rng)
        l_max = 4
        for i in range(5):
            matrix = torch.tensor(R[i].as_matrix(), dtype=torch.float64)
            wigner = build_wigner_D_cache(
                l_max, matrix, device=matrix.device, dtype=matrix.dtype
            )
            for ell in range(l_max + 1):
                Di = wigner[ell]
                product = Di.T @ Di
                identity = torch.eye(2 * ell + 1, dtype=Di.dtype)
                assert torch.allclose(product, identity, atol=1e-10), (
                    f"D^T D != I for ell={ell}, rotation {i}"
                )

    def test_identity_rotation(self):
        """D(identity) = I for all ell."""
        l_max = 4
        matrix = torch.eye(3, dtype=torch.float64)
        wigner = build_wigner_D_cache(
            l_max, matrix, device=matrix.device, dtype=matrix.dtype
        )
        for ell in range(l_max + 1):
            D = wigner[ell]
            identity = torch.eye(2 * ell + 1, dtype=D.dtype)
            assert torch.allclose(D, identity, atol=1e-10), (
                f"D(identity) != I for ell={ell}"
            )

    def test_composition(self):
        """D(R1) @ D(R2) ≈ D(R1 @ R2) for random rotations."""
        rng = np.random.default_rng(123)
        R1 = Rotation.random(random_state=rng)
        R2 = Rotation.random(random_state=rng)
        R12 = R1 * R2

        l_max = 3
        m1 = torch.tensor(R1.as_matrix(), dtype=torch.float64)
        m2 = torch.tensor(R2.as_matrix(), dtype=torch.float64)
        m12 = torch.tensor(R12.as_matrix(), dtype=torch.float64)

        D1 = build_wigner_D_cache(l_max, m1, device=m1.device, dtype=m1.dtype)
        D2 = build_wigner_D_cache(l_max, m2, device=m2.device, dtype=m2.dtype)
        D12 = build_wigner_D_cache(l_max, m12, device=m12.device, dtype=m12.dtype)

        for ell in range(l_max + 1):
            product = D1[ell] @ D2[ell]
            expected = D12[ell]
            assert torch.allclose(product, expected, atol=1e-10), (
                f"D(R1)D(R2) != D(R1R2) for ell={ell}"
            )


class TestQuadrature:
    """Test quadrature weights and grid properties."""

    def test_weights_sum(self):
        """Quadrature weights should sum to 1 (normalized Haar measure on SO(3))."""
        for L_max in [3, 5, 7]:
            lebedev_order, n_inplane = _choose_quadrature(L_max)
            _, _, _, w = get_euler_angles_quadrature(lebedev_order, n_inplane)
            # The weights are w_i / (4*pi*K) repeated K times, where w_i sum to 4*pi
            # So total sum = sum(w_i)/(4*pi*K) * K = sum(w_i)/(4*pi) = 1
            assert np.allclose(w.sum(), 1.0, atol=1e-12), (
                f"Weights don't sum to 1 for L_max={L_max}: sum={w.sum()}"
            )

    def test_choose_quadrature_monotone(self):
        """Higher L_max should give equal or larger quadrature grids."""
        prev_n = 0
        for L_max in [3, 5, 7, 11, 15]:
            n, K = _choose_quadrature(L_max)
            assert n >= prev_n
            assert K == L_max + 1
            prev_n = n

    def test_rotations_are_proper(self):
        """All rotation matrices from the quadrature should have det = +1."""
        lebedev_order, n_inplane = _choose_quadrature(5)
        alpha, beta, gamma, _ = get_euler_angles_quadrature(lebedev_order, n_inplane)
        R = _rotations_from_angles(alpha, beta, gamma)
        matrices = R.as_matrix()
        dets = np.linalg.det(matrices)
        assert np.allclose(dets, 1.0, atol=1e-10)


class _QuadraticEnergyModel(torch.nn.Module):
    """Minimal model where E = sum(positions^2). Analytical forces = -2*positions."""

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        n_sys = len(systems)
        energies = []
        for sys in systems:
            energies.append(torch.sum(sys.positions**2))

        key = Labels(
            names=["_"],
            values=torch.tensor([[0]], dtype=torch.int64),
        )
        energy_block = TensorBlock(
            values=torch.stack(energies).unsqueeze(-1),
            samples=Labels(
                names=["system"],
                values=torch.arange(n_sys, dtype=torch.int64).unsqueeze(1),
            ),
            components=[],
            properties=Labels(
                names=["energy"],
                values=torch.tensor([[0]], dtype=torch.int64),
            ),
        )
        return {"energy": TensorMap(key, [energy_block])}

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return []


class _RenamedEnergyModel(_QuadraticEnergyModel):
    """Same as _QuadraticEnergyModel, but with a non-standard output name."""

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        out = super().forward(systems, outputs, selected_atoms)
        return {"mtt::energy": out["energy"]}


class _OffsetAnisotropicEnergyModel(torch.nn.Module):
    """Energy = 1e5 + sum(positions^2) + sum(positions[:, 0]).

    Like _AnisotropicEnergyModel (exact O(3) variance = |sum of positions|^2 / 3),
    but with a large invariant offset: in float32, the offset makes the
    second-moment variance estimator cancel catastrophically unless the
    statistics are accumulated in float64.
    """

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        energies = [
            1.0e5 + torch.sum(sys.positions**2) + torch.sum(sys.positions[:, 0])
            for sys in systems
        ]
        block = TensorBlock(
            values=torch.stack(energies).unsqueeze(-1),
            samples=Labels(
                names=["system"],
                values=torch.arange(len(systems), dtype=torch.int64).unsqueeze(1),
            ),
            components=[],
            properties=Labels(
                names=["energy"],
                values=torch.tensor([[0]], dtype=torch.int64),
            ),
        )
        key = Labels(names=["_"], values=torch.tensor([[0]], dtype=torch.int64))
        return {"energy": TensorMap(key, [block])}

    def requested_neighbor_lists(self):
        return []


class _EnergyAndVectorModel(torch.nn.Module):
    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        n_sys = len(systems)
        key = Labels(names=["_"], values=torch.tensor([[0]], dtype=torch.int64))
        result = {}

        if "energy" in outputs:
            energy_block = TensorBlock(
                values=torch.stack(
                    [torch.sum(sys.positions**2) for sys in systems]
                ).unsqueeze(-1),
                samples=Labels(
                    names=["system"],
                    values=torch.arange(n_sys, dtype=torch.int64).unsqueeze(1),
                ),
                components=[],
                properties=Labels(
                    names=["energy"],
                    values=torch.tensor([[0]], dtype=torch.int64),
                ),
            )
            result["energy"] = TensorMap(key, [energy_block])

        if "non_conservative_forces" in outputs:
            values = torch.cat([sys.positions.unsqueeze(-1) for sys in systems], dim=0)
            samples = []
            for i_sys, sys in enumerate(systems):
                for atom in range(len(sys)):
                    samples.append([i_sys, atom])
            force_block = TensorBlock(
                values=values,
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor(samples, dtype=torch.int64),
                ),
                components=[
                    Labels(
                        names=["xyz"],
                        values=torch.arange(3, dtype=torch.int64).reshape(-1, 1),
                    )
                ],
                properties=Labels(
                    names=["p"],
                    values=torch.tensor([[0]], dtype=torch.int64),
                ),
            )
            force_tmap = TensorMap(key, [force_block])
            if selected_atoms is not None:
                force_tmap = mts.slice(
                    force_tmap,
                    axis="samples",
                    selection=selected_atoms,
                )
            result["non_conservative_forces"] = force_tmap

        return result

    def requested_neighbor_lists(self):
        return []


class _AnisotropicEnergyModel(torch.nn.Module):
    """Energy = sum(positions^2) + sum(positions[:, 0]).

    As a function of the O(3) transformation applied to the input positions, the
    output is band-limited: an invariant part (lambda=0) plus a vector part
    (lambda=1, odd under inversion). This makes exact statements about its
    character projections possible.
    """

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        energies = [
            torch.sum(sys.positions**2) + torch.sum(sys.positions[:, 0])
            for sys in systems
        ]
        block = TensorBlock(
            values=torch.stack(energies).unsqueeze(-1),
            samples=Labels(
                names=["system"],
                values=torch.arange(len(systems), dtype=torch.int64).unsqueeze(1),
            ),
            components=[],
            properties=Labels(
                names=["energy"],
                values=torch.tensor([[0]], dtype=torch.int64),
            ),
        )
        key = Labels(names=["_"], values=torch.tensor([[0]], dtype=torch.int64))
        return {"energy": TensorMap(key, [block])}

    def requested_neighbor_lists(self):
        return []


class TestGradientForces:
    """Test conservative forces from autograd via _evaluate_with_gradients."""

    def test_forces_identity_rotation(self):
        """With identity rotation, forces should be -2*positions for E=sum(pos^2)."""
        model = _QuadraticEnergyModel()
        positions = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64
        )
        system = System(
            types=torch.tensor([1, 1]),
            positions=positions,
            cell=torch.zeros(3, 3, dtype=torch.float64),
            pbc=torch.tensor([False, False, False]),
        )
        rotation = torch.eye(3, dtype=torch.float64).unsqueeze(0)  # (1, 3, 3)
        outputs = {"energy": ModelOutput(sample_kind="system")}

        out = _evaluate_with_gradients(
            model,
            system,
            rotation,
            outputs,
            None,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

        assert "forces" in out
        forces = out["forces"].block().values.squeeze(-1)  # (n_atoms, 3) for N=1
        expected = -2.0 * positions
        assert torch.allclose(forces, expected, atol=1e-12)

    def test_forces_with_rotation(self):
        """Forces in rotated frame should equal R @ (forces in lab frame).
        For E=sum(pos^2), forces_lab = -2*pos_lab.
        In rotated frame: forces_rot = -dE/d(pos_rot) where pos_rot = pos_lab @ R.T.
        Since E = sum((pos_rot @ R)^2) = sum(pos_rot^2) (R is orthogonal),
        forces_rot = -2*pos_rot = -2*(pos_lab @ R.T).
        """
        model = _QuadraticEnergyModel()
        positions = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64
        )
        system = System(
            types=torch.tensor([1, 1]),
            positions=positions,
            cell=torch.zeros(3, 3, dtype=torch.float64),
            pbc=torch.tensor([False, False, False]),
        )
        # Random rotation
        rng = np.random.default_rng(42)
        R_scipy = Rotation.random(random_state=rng)
        R = torch.tensor(R_scipy.as_matrix(), dtype=torch.float64)
        outputs = {"energy": ModelOutput(sample_kind="system")}

        out = _evaluate_with_gradients(
            model,
            system,
            R.unsqueeze(0),
            outputs,
            None,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

        forces_rot = out["forces"].block().values.squeeze(-1)
        expected_rot = -2.0 * (positions @ R.T)
        assert torch.allclose(forces_rot, expected_rot, atol=1e-12)

    def test_stress_periodic_system(self):
        """For a periodic system with E=sum(pos^2), check stress via strain trick.

        With strain trick: pos_final = pos_rot @ strain, so
        E = sum((pos_rot @ strain)^2) = sum_i sum_a (sum_b pos_rot_ib * strain_ba)^2
        dE/d(strain_cd) = 2 * sum_i sum_a (pos_rot @ strain)_ia * pos_rot_ic * delta_da
                        = 2 * (pos_rot.T @ (pos_rot @ strain))_{ca}  (at strain=I)
                        = 2 * pos_rot.T @ pos_rot
        stress = (1/V) * dE/d(strain) = (2/V) * pos_rot.T @ pos_rot
        """
        model = _QuadraticEnergyModel()
        positions = torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float64
        )
        cell = torch.eye(3, dtype=torch.float64) * 5.0
        system = System(
            types=torch.tensor([1, 1]),
            positions=positions,
            cell=cell,
            pbc=torch.tensor([True, True, True]),
        )
        R = torch.eye(3, dtype=torch.float64).unsqueeze(0)
        outputs = {"energy": ModelOutput(sample_kind="system")}

        out = _evaluate_with_gradients(
            model,
            system,
            R,
            outputs,
            None,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

        assert "stress" in out
        stress = out["stress"].block().values.squeeze(0).squeeze(-1)  # (3, 3)
        volume = torch.abs(torch.linalg.det(cell))
        expected_stress = 2.0 * positions.T @ positions / volume
        assert torch.allclose(stress, expected_stress, atol=1e-12)

    def test_no_stress_for_nonperiodic(self):
        """Non-periodic systems should not produce stress output."""
        model = _QuadraticEnergyModel()
        system = System(
            types=torch.tensor([1]),
            positions=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64),
            cell=torch.zeros(3, 3, dtype=torch.float64),
            pbc=torch.tensor([False, False, False]),
        )
        R = torch.eye(3, dtype=torch.float64).unsqueeze(0)
        outputs = {"energy": ModelOutput(sample_kind="system")}

        out = _evaluate_with_gradients(
            model,
            system,
            R,
            outputs,
            None,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

        assert "forces" in out
        assert "stress" not in out

    def test_forces_batched_rotations(self):
        """N>1 rotations should produce per-system forces that match what the same
        rotations would produce one at a time. Validates the batched-gradient refactor.
        """
        model = _QuadraticEnergyModel()
        positions = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64
        )
        system = System(
            types=torch.tensor([1, 1]),
            positions=positions,
            cell=torch.zeros(3, 3, dtype=torch.float64),
            pbc=torch.tensor([False, False, False]),
        )

        rng = np.random.default_rng(7)
        rotations = torch.stack(
            [
                torch.tensor(
                    Rotation.random(random_state=rng).as_matrix(),
                    dtype=torch.float64,
                )
                for _ in range(3)
            ],
            dim=0,
        )
        outputs = {"energy": ModelOutput(sample_kind="system")}

        out = _evaluate_with_gradients(
            model,
            system,
            rotations,
            outputs,
            None,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

        forces = out["forces"].block().values.squeeze(-1)  # (3 * n_atoms, 3)
        n_atoms = positions.shape[0]
        for i in range(rotations.shape[0]):
            R = rotations[i]
            expected = -2.0 * (positions @ R.T)
            actual = forces[i * n_atoms : (i + 1) * n_atoms]
            assert torch.allclose(actual, expected, atol=1e-12)

        # also verify sample labels are (system=i, atom=j) in row-major order
        samples = out["forces"].block().samples
        sys_col = samples.column("system")
        atom_col = samples.column("atom")
        for i in range(rotations.shape[0]):
            for j in range(n_atoms):
                row = i * n_atoms + j
                assert int(sys_col[row]) == i
                assert int(atom_col[row]) == j

    def test_stress_batched_rotations(self):
        """Stress under N>1 rotations: rotation-invariant magnitude per system, with
        rotated principal axes."""
        model = _QuadraticEnergyModel()
        positions = torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float64
        )
        cell = torch.eye(3, dtype=torch.float64) * 5.0
        system = System(
            types=torch.tensor([1, 1]),
            positions=positions,
            cell=cell,
            pbc=torch.tensor([True, True, True]),
        )

        rng = np.random.default_rng(11)
        rotations = torch.stack(
            [
                torch.tensor(
                    Rotation.random(random_state=rng).as_matrix(),
                    dtype=torch.float64,
                )
                for _ in range(2)
            ],
            dim=0,
        )
        outputs = {"energy": ModelOutput(sample_kind="system")}

        out = _evaluate_with_gradients(
            model,
            system,
            rotations,
            outputs,
            None,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

        assert "stress" in out
        stress_values = out["stress"].block().values.squeeze(-1)  # (N, 3, 3)
        volume = torch.abs(torch.linalg.det(cell))
        for i in range(rotations.shape[0]):
            R = rotations[i]
            rotated_pos = positions @ R.T
            expected = 2.0 * rotated_pos.T @ rotated_pos / volume
            assert torch.allclose(stress_values[i], expected, atol=1e-12)


class TestSymmetrizedModelForward:
    def _make_system(self, dtype=torch.float64):
        return System(
            types=torch.tensor([1, 1], dtype=torch.int32),
            positions=torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=dtype),
            cell=torch.zeros((3, 3), dtype=dtype),
            pbc=torch.tensor([False, False, False]),
        )

    def _make_second_system(self, dtype=torch.float64):
        return System(
            types=torch.tensor([1, 1], dtype=torch.int32),
            positions=torch.tensor([[0.0, 0.0, 3.0], [4.0, 0.0, 0.0]], dtype=dtype),
            cell=torch.zeros((3, 3), dtype=dtype),
            pbc=torch.tensor([False, False, False]),
        )

    def test_scalar_forward_outputs(self):
        model = SymmetrizedModel(
            _QuadraticEnergyModel(),
            max_o3_lambda_character=1,
            max_o3_lambda_target=0,
            batch_size=2,
        ).to(dtype=torch.float64)

        outputs = {"energy": ModelOutput(sample_kind="system")}
        result = model([self._make_system()], outputs)

        assert "energy_l0_mean" in result
        assert "energy_l0_var" in result
        assert "energy_l0_norm_squared" in result
        assert torch.allclose(
            result["energy_l0_mean"].block().values,
            torch.tensor([[[5.0]]], dtype=torch.float64),
            atol=1e-10,
        )

    def test_forward_character_projections(self):
        model = SymmetrizedModel(
            _QuadraticEnergyModel(),
            max_o3_lambda_character=1,
            max_o3_lambda_target=0,
            batch_size=1,
        ).to(dtype=torch.float64)

        outputs = {"energy": ModelOutput(sample_kind="system")}
        result = model(
            [self._make_system()],
            outputs,
            compute_character_projections=True,
        )

        expected_keys = {
            "energy_l0_mean",
            "energy_l0_var",
            "energy_l0_norm_squared",
            "energy_l0_character_projection",
        }
        assert set(result.keys()) == expected_keys

    def test_character_projections_without_gradients_run_under_no_grad(self):
        class _GradStateModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.recorded_grad_states = []

            def forward(self, systems, outputs, selected_atoms):
                self.recorded_grad_states.append(torch.is_grad_enabled())
                # shape (n_systems=1, n_properties=1): no components, so 2D
                values = torch.tensor(
                    [[1.0]],
                    dtype=systems[0].positions.dtype,
                    device=systems[0].positions.device,
                )
                tensor = TensorMap(
                    Labels(
                        ["_"],
                        torch.tensor([[0]], dtype=torch.int64, device=values.device),
                    ),
                    [
                        TensorBlock(
                            values=values,
                            samples=Labels(
                                ["system"],
                                torch.tensor(
                                    [[0]], dtype=torch.int64, device=values.device
                                ),
                            ),
                            components=[],
                            properties=Labels(
                                ["energy"],
                                torch.tensor(
                                    [[0]], dtype=torch.int64, device=values.device
                                ),
                            ),
                        )
                    ],
                )
                return {"energy": tensor}

        base_model = _GradStateModel()
        model = SymmetrizedModel(
            base_model,
            max_o3_lambda_character=1,
            max_o3_lambda_target=0,
            batch_size=1,
        ).to(dtype=torch.float64)

        outputs = {"energy": ModelOutput(sample_kind="system")}
        result = model(
            [self._make_system()],
            outputs,
            compute_character_projections=True,
            compute_gradients=False,
        )

        assert "energy_l0_character_projection" in result
        assert len(base_model.recorded_grad_states) > 0
        assert all(state is False for state in base_model.recorded_grad_states)

    def test_storage_device_does_not_change_outputs(self):
        # Regression: with `compute_gradients=False`, setting storage_device
        # must only move tensors, not change numerical results.
        outputs = {"energy": ModelOutput(sample_kind="system")}
        systems = [self._make_system()]

        results = {}
        for storage_device in (None, "cpu"):
            base_model = _QuadraticEnergyModel()
            model = SymmetrizedModel(
                base_model,
                max_o3_lambda_character=1,
                max_o3_lambda_target=0,
                batch_size=2,
                storage_device=storage_device,
            ).to(dtype=torch.float64)
            results[storage_device] = model(
                systems, outputs, compute_character_projections=True
            )

        shared_keys = set(results[None].keys()) & set(results["cpu"].keys())
        assert shared_keys, "no shared output keys between storage modes"
        for name in shared_keys:
            tensor_false = results[None][name]
            tensor_true = results["cpu"][name]
            assert tensor_false.keys == tensor_true.keys
            for key in tensor_false.keys:
                block_false = tensor_false.block(key)
                block_true = tensor_true.block(key)
                assert torch.allclose(
                    block_false.values.cpu(),
                    block_true.values.cpu(),
                    atol=1e-12,
                ), f"storage device changed values for '{name}' / key {key}"

    def test_compute_gradients_produces_forces(self):
        # Regression: with `compute_gradients=True`, _evaluate_with_gradients must
        # run autograd through the base model and inject "forces" into the output
        # stream so the symmetrization pipeline produces forces_l1_*. The previous
        # bug coupled autograd enablement to the output offloading policy and could
        # silently break this path.
        outputs = {"energy": ModelOutput(sample_kind="system")}
        model = SymmetrizedModel(
            _QuadraticEnergyModel(),
            max_o3_lambda_character=1,
            max_o3_lambda_target=1,
            batch_size=1,
        ).to(dtype=torch.float64)

        result = model([self._make_system()], outputs, compute_gradients=True)

        assert "forces_l1_mean" in result
        forces = result["forces_l1_mean"].block().values
        # Forces for E=sum(pos^2) at positions [[1,0,0],[0,2,0]] have magnitudes
        # [2, 4]; back-rotated/averaged forces retain non-trivial values for at
        # least one atom (averaging is over O(3), not over atoms).
        assert torch.any(forces.abs() > 0.5)

    def test_vector_like_forward_outputs(self):
        model = SymmetrizedModel(
            _EnergyAndVectorModel(),
            max_o3_lambda_character=1,
            max_o3_lambda_target=1,
            batch_size=2,
        ).to(dtype=torch.float64)

        outputs = {
            "energy": ModelOutput(sample_kind="system"),
            "non_conservative_forces": ModelOutput(sample_kind="atom"),
        }
        result = model([self._make_system()], outputs)

        assert "non_conservative_forces_l1_mean" in result
        assert "non_conservative_forces_l1_var" in result
        assert "non_conservative_forces_l1_norm_squared" in result

    def test_selected_atoms_are_mapped_per_outer_system(self):
        systems = [self._make_system(), self._make_second_system()]
        model = SymmetrizedModel(
            _EnergyAndVectorModel(),
            max_o3_lambda_character=1,
            max_o3_lambda_target=1,
            batch_size=5,
        ).to(dtype=torch.float64)

        outputs = {
            "energy": ModelOutput(sample_kind="system"),
            "non_conservative_forces": ModelOutput(sample_kind="atom"),
        }
        selected_atoms = Labels(
            names=["system", "atom"],
            values=torch.tensor([[0, 0], [1, 1]], dtype=torch.int64),
        )

        result = model(systems, outputs, selected_atoms=selected_atoms)

        energy_block = result["energy_l0_mean"].block()
        assert energy_block.samples.values.tolist() == [[0], [1]]
        assert torch.allclose(
            energy_block.values[:, 0, 0],
            torch.tensor([5.0, 25.0], dtype=torch.float64),
            atol=1e-10,
        )

        force_block = result["non_conservative_forces_l1_mean"].block()
        assert force_block.samples.values.tolist() == [[0, 0], [1, 1]]
        assert torch.allclose(
            force_block.values.roll(1, 1).squeeze(-1),
            torch.stack([systems[0].positions[0], systems[1].positions[1]]),
            atol=1e-10,
        )

    def test_selected_atoms_can_be_empty_for_some_systems(self):
        systems = [self._make_system(), self._make_second_system()]
        model = SymmetrizedModel(
            _EnergyAndVectorModel(),
            max_o3_lambda_character=1,
            max_o3_lambda_target=1,
            batch_size=5,
        ).to(dtype=torch.float64)

        outputs = {
            "non_conservative_forces": ModelOutput(sample_kind="atom"),
        }
        selected_atoms = Labels(
            names=["system", "atom"],
            values=torch.tensor([[1, 1]], dtype=torch.int64),
        )

        result = model(systems, outputs, selected_atoms=selected_atoms)

        force_block = result["non_conservative_forces_l1_mean"].block()
        assert force_block.samples.values.tolist() == [[1, 1]]
        assert torch.allclose(
            force_block.values.roll(1, 1).squeeze(-1),
            systems[1].positions[1].unsqueeze(0),
            atol=1e-10,
        )


class TestCharacterProjectionValidation:
    def _make_system(self):
        return System(
            types=torch.tensor([1, 1], dtype=torch.int32),
            positions=torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float64
            ),
            cell=torch.zeros((3, 3), dtype=torch.float64),
            pbc=torch.tensor([False, False, False]),
        )

    def test_equivariance_only_without_character_lambda(self):
        # the common use case: max_o3_lambda_character can be omitted entirely,
        # and the default grid then follows the target angular momentum
        model = SymmetrizedModel(
            _QuadraticEnergyModel(),
            max_o3_lambda_target=0,
            batch_size=4,
        ).to(dtype=torch.float64)
        assert model.max_o3_lambda_grid == 1  # 2 * max_o3_lambda_target + 1

        outputs = {"energy": ModelOutput(sample_kind="system")}
        result = model([self._make_system()], outputs)
        assert set(result.keys()) == {
            "energy_l0_mean",
            "energy_l0_var",
            "energy_l0_norm_squared",
        }

    def test_character_projections_require_character_lambda(self):
        model = SymmetrizedModel(
            _QuadraticEnergyModel(),
            max_o3_lambda_target=0,
        ).to(dtype=torch.float64)

        outputs = {"energy": ModelOutput(sample_kind="system")}
        with pytest.raises(ValueError, match="max_o3_lambda_character must be set"):
            model([self._make_system()], outputs, compute_character_projections=True)

    def test_character_projections_insufficient_grid_raises(self):
        # a grid unable to integrate the projections exactly produces silently
        # wrong results (isotypical fractions far above 1), so it must be
        # rejected when projections are requested, and only then
        model = SymmetrizedModel(
            _QuadraticEnergyModel(),
            max_o3_lambda_target=0,
            max_o3_lambda_character=2,
            max_o3_lambda_grid=1,  # Lebedev order 3 < 2 * 2
        ).to(dtype=torch.float64)

        outputs = {"energy": ModelOutput(sample_kind="system")}
        with pytest.raises(ValueError, match="too coarse for character projections"):
            model([self._make_system()], outputs, compute_character_projections=True)

        result = model([self._make_system()], outputs)
        assert "energy_l0_mean" in result


class TestPerSystemHelpers:
    def _make_systems(self):
        return [
            System(
                types=torch.tensor([1, 1], dtype=torch.int32),
                positions=positions,
                cell=torch.zeros((3, 3), dtype=torch.float64),
                pbc=torch.tensor([False, False, False]),
            )
            for positions in [
                torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float64),
                torch.tensor([[0.0, 0.0, 3.0], [4.0, 0.0, 0.0]], dtype=torch.float64),
            ]
        ]

    def test_character_fractions_capture_band_limited_output(self):
        # the energy of _AnisotropicEnergyModel has content only at lambda=0 and
        # lambda=1 (both in the proper chi_sigma=+1 sector), so with a sufficient
        # grid the fractions must sum to 1 and vanish everywhere else
        model = SymmetrizedModel(
            _AnisotropicEnergyModel(),
            max_o3_lambda_target=0,
            max_o3_lambda_character=3,
            batch_size=64,
        ).to(dtype=torch.float64)

        outputs = {"energy": ModelOutput(sample_kind="system")}
        result = model(
            self._make_systems(), outputs, compute_character_projections=True
        )

        proper, improper, lambdas = per_system_character_fractions(result, "energy_l0")

        assert lambdas.tolist() == [0, 1, 2, 3]
        assert proper.shape == (2, 4)
        total = (proper + improper).sum(dim=1)
        assert torch.allclose(total, torch.ones_like(total), atol=1e-8)
        assert torch.all(improper.abs() < 1e-8)
        assert torch.all(proper[:, 2:].abs() < 1e-8)

    def test_equivariance_rmse_vanishes_for_equivariant_output(self):
        # forces of _EnergyAndVectorModel are the (rotated) positions, which
        # back-rotate exactly: the equivariance RMSE must be zero per system
        model = SymmetrizedModel(
            _EnergyAndVectorModel(),
            max_o3_lambda_target=1,
            batch_size=8,
        ).to(dtype=torch.float64)

        outputs = {"non_conservative_forces": ModelOutput(sample_kind="atom")}
        result = model(self._make_systems(), outputs)

        rmse = per_system_equivariance_rmse(result, "non_conservative_forces_l1")
        assert rmse.block().values.shape == (2, 1)
        # tolerance above the float64 cancellation floor of the variance,
        # sqrt(eps * max ||x||^2), which summation order can push past 1e-8
        assert torch.all(rmse.block().values.abs() < 1e-7)

    def test_equivariance_rmse_reduction(self):
        # RMSE = sqrt( mean over a system's samples of
        #              (component-summed variance) / (2l+1) ),
        # with 2l+1 read from the components of the matching _mean tensor
        keys = Labels(["_"], torch.tensor([[0]], dtype=torch.int64))
        samples = Labels(
            ["system", "atom"],
            torch.tensor([[0, 0], [0, 1], [1, 0]], dtype=torch.int64),
        )
        properties = Labels(["p"], torch.tensor([[0]], dtype=torch.int64))
        variance = TensorMap(
            keys,
            [
                TensorBlock(
                    values=torch.tensor([[6.0], [15.0], [24.0]], dtype=torch.float64),
                    samples=samples,
                    components=[],
                    properties=properties,
                )
            ],
        )
        mean = TensorMap(
            keys,
            [
                TensorBlock(
                    values=torch.zeros((3, 3, 1), dtype=torch.float64),
                    samples=samples,
                    components=[
                        Labels(
                            ["o3_mu"],
                            torch.tensor([[-1], [0], [1]], dtype=torch.int64),
                        )
                    ],
                    properties=properties,
                )
            ],
        )

        rmse = per_system_equivariance_rmse(
            {"foo_var": variance, "foo_mean": mean}, "foo"
        )

        block = rmse.block()
        assert block.samples.names == ["system"]
        assert block.properties == properties
        # system 0: atoms contribute 6/3 and 15/3 -> mean 3.5; system 1: 24/3
        expected = torch.sqrt(torch.tensor([3.5, 8.0], dtype=torch.float64))
        assert torch.allclose(block.values.squeeze(1), expected, atol=1e-12)

    def test_equivariance_rmse_preserves_blocks_and_properties(self):
        # spherical targets keep their (o3_lambda, o3_sigma) block and property
        # (e.g. radial channel) structure: one RMSE per system, block, and
        # property, each block divided by its own 2l+1, so the values can be
        # aggregated later in whichever way the analysis needs
        keys = Labels(
            ["o3_lambda", "o3_sigma"],
            torch.tensor([[0, 1], [1, 1]], dtype=torch.int64),
        )
        properties = Labels(["n"], torch.tensor([[0], [1]], dtype=torch.int64))
        samples_l0 = Labels(
            ["system", "atom"],
            torch.tensor([[0, 0], [1, 0]], dtype=torch.int64),
        )
        # the l=1 block only has samples in system 0
        samples_l1 = Labels(
            ["system", "atom"], torch.tensor([[0, 0]], dtype=torch.int64)
        )

        variance = TensorMap(
            keys,
            [
                TensorBlock(
                    values=torch.tensor([[1.0, 4.0], [9.0, 16.0]], dtype=torch.float64),
                    samples=samples_l0,
                    components=[],
                    properties=properties,
                ),
                TensorBlock(
                    values=torch.tensor([[3.0, 12.0]], dtype=torch.float64),
                    samples=samples_l1,
                    components=[],
                    properties=properties,
                ),
            ],
        )
        mean = TensorMap(
            keys,
            [
                TensorBlock(
                    values=torch.zeros((2, 1, 2), dtype=torch.float64),
                    samples=samples_l0,
                    components=[
                        Labels(["o3_mu"], torch.tensor([[0]], dtype=torch.int64))
                    ],
                    properties=properties,
                ),
                TensorBlock(
                    values=torch.zeros((1, 3, 2), dtype=torch.float64),
                    samples=samples_l1,
                    components=[
                        Labels(
                            ["o3_mu"],
                            torch.tensor([[-1], [0], [1]], dtype=torch.int64),
                        )
                    ],
                    properties=properties,
                ),
            ],
        )

        rmse = per_system_equivariance_rmse(
            {"spherical_var": variance, "spherical_mean": mean},
            "spherical",
            n_systems=2,
        )

        assert rmse.keys == keys
        # l=0 block: multiplicity 1, one atom per system -> sqrt of the variance
        block_l0 = rmse.block({"o3_lambda": 0, "o3_sigma": 1})
        assert block_l0.properties == properties
        assert torch.allclose(
            block_l0.values,
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64),
            atol=1e-12,
        )
        # l=1 block: multiplicity 3; system 1 has no samples -> zero
        block_l1 = rmse.block({"o3_lambda": 1, "o3_sigma": 1})
        assert torch.allclose(
            block_l1.values,
            torch.tensor([[1.0, 2.0], [0.0, 0.0]], dtype=torch.float64),
            atol=1e-12,
        )


class TestEquivarianceErrorMethod:
    def _make_systems(self):
        return [
            System(
                types=torch.tensor([1, 1], dtype=torch.int32),
                positions=positions,
                cell=torch.zeros((3, 3), dtype=torch.float64),
                pbc=torch.tensor([False, False, False]),
            )
            for positions in [
                torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float64),
                torch.tensor([[0.0, 0.0, 3.0], [4.0, 0.0, 0.0]], dtype=torch.float64),
            ]
        ]

    def test_equivariant_output_has_zero_error(self):
        # forces of _EnergyAndVectorModel back-rotate exactly, so the reported
        # equivariance error must vanish; this is the invariant the metric is for
        model = SymmetrizedModel(
            _EnergyAndVectorModel(),
            max_o3_lambda_target=1,
            batch_size=8,
        ).to(dtype=torch.float64)
        systems = self._make_systems()

        errors = model.equivariance_error(
            systems,
            {
                "energy": ModelOutput(sample_kind="system"),
                "non_conservative_forces": ModelOutput(sample_kind="atom"),
            },
        )

        assert set(errors.keys()) == {"energy_l0", "non_conservative_forces_l1"}
        block = errors["non_conservative_forces_l1"].block()
        assert block.samples.names == ["system"]
        assert block.samples.values[:, 0].tolist() == [0, 1]
        assert block.values.shape == (2, 1)
        # above the float64 cancellation floor of the variance
        assert torch.all(block.values.abs() < 1e-7)

    def test_non_equivariant_output_matches_helper(self):
        # a non-invariant energy must give a strictly positive error, equal to
        # the per_system_equivariance_rmse reduction of the raw forward outputs
        model = SymmetrizedModel(
            _AnisotropicEnergyModel(),
            max_o3_lambda_target=0,
            batch_size=8,
        ).to(dtype=torch.float64)
        systems = self._make_systems()
        outputs = {"energy": ModelOutput(sample_kind="system")}

        errors = model.equivariance_error(systems, outputs)
        raw = model(systems, outputs)
        expected = per_system_equivariance_rmse(raw, "energy_l0", n_systems=2)

        values = errors["energy_l0"].block().values
        assert torch.all(values > 0.1)
        assert torch.allclose(values, expected.block().values, atol=1e-12)


class TestFloat64Accumulation:
    """Statistics are accumulated in float64 regardless of the model dtype."""

    def test_float32_model_with_large_energy_offset(self):
        # positions sum to v = (1, 2, 0), so the exact O(3) variance of the
        # energy is |v|^2 / 3, independent of the invariant 1e5 offset; with
        # float32 accumulation the offset wipes out the variance entirely
        # (second moments ~1e10 quantize at ~1e3)
        system = System(
            types=torch.tensor([1, 1], dtype=torch.int32),
            positions=torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float32
            ),
            cell=torch.zeros((3, 3), dtype=torch.float32),
            pbc=torch.tensor([False, False, False]),
        )
        model = SymmetrizedModel(
            _OffsetAnisotropicEnergyModel(),
            max_o3_lambda_target=1,
            batch_size=8,
        ).to(dtype=torch.float32)

        results = model([system], {"energy": ModelOutput(sample_kind="system")})

        variance = results["energy_l0_var"].block().values
        assert variance.dtype == torch.float64
        expected = 5.0 / 3.0
        # the residual error is the float32 round-off of the model outputs
        # themselves (~1e5 * 1e-7), far below the 5% tolerance
        assert abs(variance.item() - expected) < 0.05 * expected

    """Gradients can be derived from an energy output with a non-standard name."""

    def _make_system(self):
        return System(
            types=torch.tensor([1, 1], dtype=torch.int32),
            positions=torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float64
            ),
            cell=torch.zeros((3, 3), dtype=torch.float64),
            pbc=torch.tensor([False, False, False]),
        )

    def test_gradients_with_custom_energy_name(self):
        systems = [self._make_system()]

        reference = SymmetrizedModel(
            _QuadraticEnergyModel(), max_o3_lambda_target=1, batch_size=4
        ).to(dtype=torch.float64)
        renamed = SymmetrizedModel(
            _RenamedEnergyModel(), max_o3_lambda_target=1, batch_size=4
        ).to(dtype=torch.float64)

        reference_results = reference(
            systems,
            {"energy": ModelOutput(sample_kind="system")},
            compute_gradients=True,
        )
        renamed_results = renamed(
            systems,
            {"mtt::energy": ModelOutput(sample_kind="system")},
            compute_gradients=True,
            energy_name="mtt::energy",
        )

        # the derived forces are identical; only the energy naming differs
        # (non-standard energies pass through without the _l0 relabeling)
        for suffix in ("mean", "var", "norm_squared"):
            assert mts.allclose(
                reference_results[f"forces_l1_{suffix}"],
                renamed_results[f"forces_l1_{suffix}"],
                atol=1e-12,
            )
            assert torch.allclose(
                reference_results[f"energy_l0_{suffix}"].block().values.squeeze(1),
                renamed_results[f"mtt::energy_{suffix}"].block().values,
                atol=1e-12,
            )

    def test_missing_energy_name_raises(self):
        model = SymmetrizedModel(
            _RenamedEnergyModel(), max_o3_lambda_target=1, batch_size=4
        ).to(dtype=torch.float64)

        with pytest.raises(ValueError, match="requires 'energy' in outputs"):
            model(
                [self._make_system()],
                {"mtt::energy": ModelOutput(sample_kind="system")},
                compute_gradients=True,
            )

    def test_equivariance_error_with_custom_energy_name(self):
        model = SymmetrizedModel(
            _RenamedEnergyModel(), max_o3_lambda_target=1, batch_size=4
        ).to(dtype=torch.float64)

        errors = model.equivariance_error(
            [self._make_system()],
            {"mtt::energy": ModelOutput(sample_kind="system")},
            compute_gradients=True,
            energy_name="mtt::energy",
        )

        assert set(errors.keys()) == {"mtt::energy", "forces_l1"}
        # the underlying model is exactly equivariant; tolerance above the
        # float64 cancellation floor of the variance
        assert torch.all(errors["forces_l1"].block().values.abs() < 1e-7)


class TestAtomisticBaseModel:
    """SymmetrizedModel accepts exported AtomisticModel base models."""

    def _make_system(self):
        return System(
            types=torch.tensor([1, 1], dtype=torch.int32),
            positions=torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float64
            ),
            cell=torch.zeros((3, 3), dtype=torch.float64),
            pbc=torch.tensor([False, False, False]),
        )

    def _export(self):
        return AtomisticModel(
            _QuadraticEnergyModel().eval(),
            ModelMetadata(),
            ModelCapabilities(
                outputs={
                    "energy": ModelOutput(
                        sample_kind="system", unit="eV", description="energy"
                    )
                },
                atomic_types=[1],
                interaction_range=0.0,
                length_unit="Angstrom",
                supported_devices=["cpu"],
                dtype="float64",
            ),
        )

    def _symmetrize(self, base_model):
        return SymmetrizedModel(
            base_model,
            max_o3_lambda_target=1,
            batch_size=4,
        ).to(dtype=torch.float64)

    def _assert_same_results(self, reference, results):
        assert set(results.keys()) == set(reference.keys())
        for key in reference:
            assert mts.allclose(reference[key], results[key], atol=1e-12)

    def test_forward_matches_raw_module(self):
        outputs = {"energy": ModelOutput(sample_kind="system")}
        systems = [self._make_system()]

        reference = self._symmetrize(_QuadraticEnergyModel())(systems, outputs)
        results = self._symmetrize(self._export())(systems, outputs)

        self._assert_same_results(reference, results)

    def test_gradients_through_exported_model(self):
        outputs = {"energy": ModelOutput(sample_kind="system")}
        systems = [self._make_system()]

        reference = self._symmetrize(_QuadraticEnergyModel())(
            systems, outputs, compute_gradients=True
        )
        results = self._symmetrize(self._export())(
            systems, outputs, compute_gradients=True
        )

        assert "forces_l1_mean" in results
        self._assert_same_results(reference, results)

    def test_loaded_model_matches_raw_module(self, tmp_path):
        outputs = {"energy": ModelOutput(sample_kind="system")}
        systems = [self._make_system()]

        path = str(tmp_path / "exported.pt")
        self._export().save(path)
        loaded = load_atomistic_model(path)

        reference = self._symmetrize(_QuadraticEnergyModel())(systems, outputs)
        results = self._symmetrize(loaded)(systems, outputs)

        self._assert_same_results(reference, results)

"""Tests for symmetrized_model.py standalone functions and SymmetrizedModel class."""

from pathlib import Path
from typing import Dict, List, Optional
import warnings

import metatensor.torch as mts
import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from scipy.spatial.transform import Rotation

from metatomic.torch import ModelOutput, System, systems_to_torch
from metatomic.torch.symmetrized_model import (
    SymmetrizedModel,
    _choose_quadrature,
    _compute_real_wigner_matrices,
    _evaluate_with_gradients,
    _l0_components_from_matrices,
    _l2_components_from_matrices,
    _rotations_from_angles,
    _transform_system,
    get_euler_angles_quadrature,
)


REAL_CHECKPOINT = (
    Path(__file__).resolve().parents[3]
    / "SYMMOD_EXAMPLE"
    / "pet-mad-xs-v1.5.0.ckpt"
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
        angles = (
            np.zeros(5),
            np.zeros(5),
            np.zeros(5),
        )
        # Use actual rotation angles
        euler = R.as_euler("ZYZ")
        angles = (euler[:, 0], euler[:, 1], euler[:, 2])

        l_max = 4
        wigner = _compute_real_wigner_matrices(l_max, angles)
        for ell in range(l_max + 1):
            D = wigner[ell]  # shape (5, 2l+1, 2l+1)
            for i in range(5):
                Di = D[i]
                product = Di.T @ Di
                identity = torch.eye(2 * ell + 1, dtype=Di.dtype)
                assert torch.allclose(product, identity, atol=1e-10), (
                    f"D^T D != I for ell={ell}, rotation {i}"
                )

    def test_identity_rotation(self):
        """D(identity) = I for all ell."""
        angles = (np.array([0.0]), np.array([0.0]), np.array([0.0]))
        l_max = 4
        wigner = _compute_real_wigner_matrices(l_max, angles)
        for ell in range(l_max + 1):
            D = wigner[ell][0]
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
        e1 = np.atleast_2d(R1.as_euler("ZYZ"))
        e2 = np.atleast_2d(R2.as_euler("ZYZ"))
        e12 = np.atleast_2d(R12.as_euler("ZYZ"))

        D1 = _compute_real_wigner_matrices(l_max, (e1[:, 0], e1[:, 1], e1[:, 2]))
        D2 = _compute_real_wigner_matrices(l_max, (e2[:, 0], e2[:, 1], e2[:, 2]))
        D12 = _compute_real_wigner_matrices(l_max, (e12[:, 0], e12[:, 1], e12[:, 2]))

        for ell in range(l_max + 1):
            product = D1[ell][0] @ D2[ell][0]
            expected = D12[ell][0]
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
        rotation = torch.eye(3, dtype=torch.float64)
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
        forces = out["forces"].block().values.squeeze(-1)  # (2, 3)
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
            R,
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
        R = torch.eye(3, dtype=torch.float64)
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
        R = torch.eye(3, dtype=torch.float64)
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

    def test_forward_project_tokens(self):
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
            project_tokens=True,
        )

        expected_keys = {
            "energy_l0",
            "energy_l0_mean",
            "energy_l0_var",
            "energy_l0_norm_squared",
            "energy_l0_character_projection",
            "energy_l0_character_projection_plus",
            "energy_l0_character_projection_minus",
        }
        assert expected_keys.issubset(result.keys())

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

        assert "non_conservative_forces_l1" in result
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


@pytest.mark.skipif(
    not REAL_CHECKPOINT.is_file(),
    reason="requires local SYMMOD_EXAMPLE checkpoint",
)
def test_real_checkpoint_energy_variance_matches_explicit_o3_reference(capfd):
    pytest.importorskip("ase")
    load_model = pytest.importorskip("metatrain.utils.io").load_model
    get_system_with_neighbor_lists = pytest.importorskip(
        "metatrain.utils.neighbor_lists"
    ).get_system_with_neighbor_lists
    from ase.build import bulk

    dtype = torch.float64

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="the 'features' output name is deprecated, please update the model to use 'feature' instead",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="the 'non_conservative_forces' output name is deprecated, please update the model to use 'non_conservative_force' instead",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Lebedev order may be insufficient for character projections\.",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"`per_atom` is deprecated, please use `sample_kind` instead.*",
            category=DeprecationWarning,
        )

        model = load_model(REAL_CHECKPOINT)
        model.eval()
        model = model.to(dtype=dtype, device="cpu")

        atoms = bulk("Si", cubic=True)
        atoms.rattle(0.1, seed=0)
        system = systems_to_torch([atoms], device="cpu", dtype=dtype)[0]
        system = get_system_with_neighbor_lists(
            system.to(dtype=dtype, device="cpu"),
            model.model.requested_neighbor_lists(),
        )

        outputs = {"energy": ModelOutput(sample_kind="system")}
        symm_model = SymmetrizedModel(
            model,
            max_o3_lambda_grid=3,
            max_o3_lambda_target=2,
            max_o3_lambda_character=2,
            batch_size=1,
        ).to(device="cpu", dtype=dtype)

        with torch.no_grad():
            result = symm_model([system], outputs)

            weights = []
            energies = []
            for inversion in [1, -1]:
                for weight, rotation in zip(
                    symm_model.so3_weights, symm_model.so3_rotations
                ):
                    transformed = _transform_system(
                        system,
                        (inversion * rotation).to(
                            dtype=system.positions.dtype,
                            device=system.positions.device,
                        ),
                    )
                    energy = model([transformed], outputs, None)["energy"].block().values.squeeze()
                    weights.append(0.5 * weight.to(dtype=dtype, device="cpu"))
                    energies.append(energy.to(dtype=dtype, device="cpu"))

    captured = capfd.readouterr()
    assert captured.out == ""
    allowed_stderr_fragments = [
        "`per_atom` is deprecated, please use `sample_kind` instead",
        "output 'energy' has an empty unit. Consider adding a unit to ensure correct unit conversion.",
        "ModelOutput.quantity is deprecated and will be removed in a future version",
        "the 'features' quantity is deprecated, please update this code to use 'feature' instead.",
    ]
    unexpected_stderr = [
        line
        for line in captured.err.splitlines()
        if line != ""
        and not any(fragment in line for fragment in allowed_stderr_fragments)
    ]
    assert unexpected_stderr == []

    weights_tensor = torch.stack(weights)
    energies_tensor = torch.stack(energies)
    mean_reference = torch.sum(weights_tensor * energies_tensor)
    norm_squared_reference = torch.sum(weights_tensor * energies_tensor.square())
    variance_reference = norm_squared_reference - mean_reference.square()

    mean_value = result["energy_l0_mean"].block().values.squeeze()
    norm_squared_value = result["energy_l0_norm_squared"].block().values.squeeze()
    variance_value = result["energy_l0_var"].block().values.squeeze()

    # The current checkpoint is not exactly O(3)-equivariant, so validate against
    # the explicit quadrature reference instead of assuming near-zero variance.
    assert torch.allclose(mean_value, mean_reference, atol=1e-12, rtol=0.0)
    assert torch.allclose(
        norm_squared_value,
        norm_squared_reference,
        atol=1e-12,
        rtol=0.0,
    )
    assert torch.allclose(variance_value, variance_reference, atol=1e-12, rtol=0.0)
